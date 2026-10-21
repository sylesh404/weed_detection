"""Recommendation Service for AgriSight AI.

Provides thread-safe asynchronous execution, persistence, and audit logging
for the Hybrid AI Weed Risk Assessment & Recommendation Engine.

Storage Directory:
- recommendation_data/recommendations.json
- recommendation_data/recommendation_history.csv
- recommendation_data/knowledge_sources.json

Completely decoupled from the real-time camera ingestion and detection loops.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import csv

from .farmer_advisor import FarmerAdvisor
from .knowledge_base import AgriculturalKnowledgeBase


class RecommendationService:
    """Thread-safe service managing recommendation generation, storage, and retrieval."""

    CSV_FIELDS = [
        "assessment_id",
        "timestamp",
        "field_id",
        "weed_count",
        "count_type",
        "visible_cotton_count",
        "infestation_level",
        "crop_risk",
        "action_priority",
        "recommended_action",
        "requires_field_verification",
        "farmer_action_taken",
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        self.root = base_dir or Path(__file__).resolve().parent.parent
        self.data_dir = self.root / "recommendation_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.json_file = self.data_dir / "recommendations.json"
        self.csv_file = self.data_dir / "recommendation_history.csv"
        self.sources_file = self.data_dir / "knowledge_sources.json"

        self.advisor = FarmerAdvisor()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rec_worker")
        self.lock = threading.RLock()

        self._latest_recommendation: Optional[Dict[str, Any]] = None
        self._is_analyzing = False
        self._last_error = ""

        # Initialize persistent knowledge sources index
        self._init_sources_file()

        # Load latest from disk if available
        self._load_latest_from_disk()

    def _init_sources_file(self):
        """Write verified citations index to disk for transparency."""
        try:
            if not self.sources_file.exists():
                citations = self.advisor.kb.get_source_citations()
                self.sources_file.write_text(json.dumps(citations, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_latest_from_disk(self):
        """Restore latest recommendation from disk on startup."""
        if self.json_file.exists():
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        self._latest_recommendation = data[0]
                    elif isinstance(data, dict):
                        self._latest_recommendation = data
            except Exception:
                pass

    def analyze(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        field_context: Optional[Dict[str, Any]] = None,
        cumulative_tracked_ids: Optional[Dict[str, set]] = None,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        """Perform weed risk assessment and recommendation generation.

        Args:
            boxes: Current verified detection bounding boxes.
            image_width: Frame pixel width.
            image_height: Frame pixel height.
            field_context: Optional agronomic inputs (field_id, growth_stage, etc.).
            cumulative_tracked_ids: Track IDs observed across session.
            async_mode: If True, queues analysis in worker pool and returns immediately.

        Returns:
            Structured recommendation dictionary or queued status.
        """
        if async_mode:
            with self.lock:
                self._is_analyzing = True
            self.executor.submit(
                self._execute_and_store,
                boxes,
                image_width,
                image_height,
                field_context,
                cumulative_tracked_ids,
            )
            return {
                "ok": True,
                "status": "ANALYSIS_QUEUED",
                "message": "Weed risk assessment is processing asynchronously.",
            }

        return self._execute_and_store(
            boxes,
            image_width,
            image_height,
            field_context,
            cumulative_tracked_ids,
        )

    def _execute_and_store(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        field_context: Optional[Dict[str, Any]],
        cumulative_tracked_ids: Optional[Dict[str, set]],
    ) -> Dict[str, Any]:
        """Internal execution and atomic persistence helper."""
        with self.lock:
            self._is_analyzing = True
            self._last_error = ""

        try:
            rec = self.advisor.advise(
                boxes=boxes,
                image_width=image_width,
                image_height=image_height,
                field_context=field_context,
                cumulative_tracked_ids=cumulative_tracked_ids,
                force_refresh=True,
            )

            with self.lock:
                self._latest_recommendation = rec
                self._save_record(rec)
            return {"ok": True, "recommendation": rec}

        except Exception as e:
            with self.lock:
                self._last_error = str(e)
            return {"ok": False, "error": f"Recommendation generation failed: {str(e)}"}

        finally:
            with self.lock:
                self._is_analyzing = False

    def _save_record(self, rec: Dict[str, Any]):
        """Persist recommendation to JSON and append to audit CSV."""
        # 1. Update JSON store (capped at 50 recent recommendations)
        try:
            records = []
            if self.json_file.exists():
                try:
                    with open(self.json_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]
                except Exception:
                    records = []
            records.insert(0, rec)
            records = records[:50]
            self.json_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 2. Append to CSV History Audit Log
        try:
            is_new = not self.csv_file.exists() or self.csv_file.stat().st_size == 0
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(
                    {
                        "assessment_id": rec.get("assessment_id"),
                        "timestamp": rec.get("timestamp"),
                        "field_id": rec.get("field_id"),
                        "weed_count": rec.get("weed_count"),
                        "count_type": rec.get("count_type"),
                        "visible_cotton_count": rec.get("visible_cotton_count"),
                        "infestation_level": rec.get("infestation_level"),
                        "crop_risk": rec.get("crop_risk"),
                        "action_priority": rec.get("action_priority"),
                        "recommended_action": rec.get("recommended_action"),
                        "requires_field_verification": rec.get("requires_field_verification"),
                        "farmer_action_taken": rec.get("farmer_action_taken", "NONE_RECORDED"),
                    }
                )
        except Exception:
            pass

    def get_latest(self) -> Dict[str, Any]:
        """Return the most recently generated recommendation."""
        with self.lock:
            if self._latest_recommendation is None:
                return {
                    "ok": True,
                    "available": False,
                    "status": "NOT_ANALYZED_YET",
                    "message": "No field recommendation has been generated yet.",
                }
            return {
                "ok": True,
                "available": True,
                "status": "RECOMMENDATION_AVAILABLE",
                "recommendation": self._latest_recommendation,
                "is_analyzing": self._is_analyzing,
            }

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        """Return stored historical recommendation records."""
        with self.lock:
            if not self.json_file.exists():
                return {"ok": True, "records": []}
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    if isinstance(records, list):
                        return {"ok": True, "records": records[:limit]}
                    return {"ok": True, "records": [records]}
            except Exception as e:
                return {"ok": False, "error": str(e), "records": []}

    def record_farmer_action(self, assessment_id: str, action_taken: str, notes: str = "") -> Dict[str, Any]:
        """Record what action the farmer took based on the recommendation."""
        with self.lock:
            updated = False
            if self._latest_recommendation and self._latest_recommendation.get("assessment_id") == assessment_id:
                self._latest_recommendation["farmer_action_taken"] = action_taken
                self._latest_recommendation["farmer_action_notes"] = notes
                self._latest_recommendation["farmer_action_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated = True

            # Update in JSON history
            if self.json_file.exists():
                try:
                    with open(self.json_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    if isinstance(records, list):
                        for r in records:
                            if r.get("assessment_id") == assessment_id:
                                r["farmer_action_taken"] = action_taken
                                r["farmer_action_notes"] = notes
                                r["farmer_action_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                updated = True
                        self.json_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
                except Exception:
                    pass

            return {
                "ok": True,
                "updated": updated,
                "message": f"Farmer action recorded for assessment {assessment_id}.",
            }

    def get_status(self) -> Dict[str, Any]:
        """Return operational status of recommendation service."""
        with self.lock:
            return {
                "ok": True,
                "service": "Hybrid AI Weed Risk Assessment & Recommendation Engine",
                "status": "OPERATIONAL",
                "is_analyzing": self._is_analyzing,
                "last_error": self._last_error,
                "has_latest_recommendation": self._latest_recommendation is not None,
                "data_directory": str(self.data_dir),
                "history_csv": str(self.csv_file),
            }

    def export_csv(self) -> Optional[Path]:
        """Return Path to CSV history file if it exists."""
        with self.lock:
            return self.csv_file if self.csv_file.exists() else None

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer farmer query using grounded agronomic knowledge."""
        with self.lock:
            current_rec = self._latest_recommendation
        res = self.advisor.answer_question(question, current_rec)
        res["ok"] = True
        return res
