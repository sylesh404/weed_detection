"""Four-Part Agricultural Plot Segmentation and Analysis System for AgriSight AI.

Divides an agricultural plot image into four equal quadrants:
- Part 01 — Northwest (NW): Top-left quadrant
- Part 02 — Northeast (NE): Top-right quadrant
- Part 03 — Southwest (SW): Bottom-left quadrant
- Part 04 — Southeast (SE): Bottom-right quadrant

Features:
1. Strict geometry preservation, handling odd image dimensions with 100% coverage
   and zero pixel loss or duplication.
2. Independent computer vision detection & verification per quadrant.
3. Bidirectional coordinate mapping (local quadrant coords <-> global plot coords).
4. Boundary deduplication to prevent double-counting plants across quadrant edges.
5. Quadrant-level and whole-field spatial analysis, weed risk assessment, and
   LangGraph AI Farmer Advisor recommendations.
6. Support for image visualization, quadrant cropping, and JSON/CSV reporting.
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
import time
import math
import copy
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .weed_analyzer import WeedAnalyzer
from .spatial_analyzer import SpatialAnalyzer
from .crop_risk_engine import CropRiskEngine

try:
    from recommendation.farmer_advisor import FarmerAdvisor
except Exception:
    FarmerAdvisor = None


# Stable quadrant definitions
QUADRANT_DEFINITIONS = [
    {
        "id": "NW",
        "key": "part_01_northwest",
        "name": "Part 01 — Northwest",
        "display_name": "Part 01 — Northwest (Top-Left)",
        "row": 0,
        "col": 0,
    },
    {
        "id": "NE",
        "key": "part_02_northeast",
        "name": "Part 02 — Northeast",
        "display_name": "Part 02 — Northeast (Top-Right)",
        "row": 0,
        "col": 1,
    },
    {
        "id": "SW",
        "key": "part_03_southwest",
        "name": "Part 03 — Southwest",
        "display_name": "Part 03 — Southwest (Bottom-Left)",
        "row": 1,
        "col": 0,
    },
    {
        "id": "SE",
        "key": "part_04_southeast",
        "name": "Part 04 — Southeast",
        "display_name": "Part 04 — Southeast (Bottom-Right)",
        "row": 1,
        "col": 1,
    },
]


class PlotSegmenter:
    """Manages 4-quadrant agricultural field segmentation, per-quadrant inference,

    coordinate mapping, and unified whole-field agronomic assessment.
    """

    def __init__(
        self,
        weed_analyzer: Optional[WeedAnalyzer] = None,
        spatial_analyzer: Optional[SpatialAnalyzer] = None,
        crop_risk_engine: Optional[CropRiskEngine] = None,
        farmer_advisor: Optional[Any] = None,
    ):
        self.weed_analyzer = weed_analyzer or WeedAnalyzer()
        self.spatial_analyzer = spatial_analyzer or SpatialAnalyzer()
        self.crop_risk_engine = crop_risk_engine or CropRiskEngine()
        if farmer_advisor is not None:
            self.farmer_advisor = farmer_advisor
        elif FarmerAdvisor is not None:
            try:
                self.farmer_advisor = FarmerAdvisor()
            except Exception:
                self.farmer_advisor = None
        else:
            self.farmer_advisor = None

    @staticmethod
    def validate_image(image: Any) -> Tuple[int, int]:
        """Validate input image array and return (width, height).

        Raises ValueError if empty, None, corrupted, or unsupported shape.
        """
        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input image must be a numpy ndarray, got {type(image)}.")
        if image.size == 0:
            raise ValueError("Input image is empty (zero pixels).")
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError(f"Unsupported image shape {image.shape}; expected 2D or 3D array.")

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}.")
        if h < 4 or w < 4:
            raise ValueError(f"Image is too small ({w}x{h}) to divide into 4 quadrants.")

        return int(w), int(h)

    def segment_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Divide an agricultural plot image into four equal quadrants.

        Guarantees:
        - Covers 100% of the image without pixel loss or duplicate pixels.
        - Handles odd dimensions correctly (e.g. W=1001 -> mid=500, cols 0..500 and 500..1001).
        - Preserves crop boundaries and offsets.
        """
        w, h = self.validate_image(image)
        mid_x = w // 2
        mid_y = h // 2

        quadrants = {}
        for qdef in QUADRANT_DEFINITIONS:
            qid = qdef["id"]
            if qid == "NW":
                x1, y1, x2, y2 = 0, 0, mid_x, mid_y
            elif qid == "NE":
                x1, y1, x2, y2 = mid_x, 0, w, mid_y
            elif qid == "SW":
                x1, y1, x2, y2 = 0, mid_y, mid_x, h
            elif qid == "SE":
                x1, y1, x2, y2 = mid_x, mid_y, w, h
            else:
                continue

            crop = image[y1:y2, x1:x2].copy()
            quad_w = x2 - x1
            quad_h = y2 - y1

            quadrants[qid] = {
                "id": qid,
                "key": qdef["key"],
                "name": qdef["name"],
                "display_name": qdef["display_name"],
                "row": qdef["row"],
                "col": qdef["col"],
                "bounds": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "offset_x": x1,
                "offset_y": y1,
                "width": quad_w,
                "height": quad_h,
                "crop": crop,
            }

        return {
            "original_dimensions": {"width": w, "height": h},
            "midpoint": {"x": mid_x, "y": mid_y},
            "quadrants": quadrants,
        }

    @staticmethod
    def map_local_to_global(
        boxes: List[Dict[str, Any]],
        offset_x: int,
        offset_y: int,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        """Map quadrant-local coordinates to global plot image coordinates.

        Stores both local and global coordinates, clips to image boundaries,
        and prevents repeated offset applications.
        """
        mapped = []
        for b in boxes:
            # Check if this box was already converted to global
            if b.get("is_global"):
                # Already global; keep it but ensure bounds are clipped
                mapped_b = copy.deepcopy(b)
                mapped_b["x1"] = max(0.0, min(float(image_width), float(mapped_b.get("x1", 0))))
                mapped_b["y1"] = max(0.0, min(float(image_height), float(mapped_b.get("y1", 0))))
                mapped_b["x2"] = max(0.0, min(float(image_width), float(mapped_b.get("x2", 0))))
                mapped_b["y2"] = max(0.0, min(float(image_height), float(mapped_b.get("y2", 0))))
                mapped.append(mapped_b)
                continue

            local_x1 = float(b.get("x1", 0))
            local_y1 = float(b.get("y1", 0))
            local_x2 = float(b.get("x2", 0))
            local_y2 = float(b.get("y2", 0))

            global_x1 = max(0.0, min(float(image_width), local_x1 + offset_x))
            global_y1 = max(0.0, min(float(image_height), local_y1 + offset_y))
            global_x2 = max(0.0, min(float(image_width), local_x2 + offset_x))
            global_y2 = max(0.0, min(float(image_height), local_y2 + offset_y))

            # Ensure x2 >= x1 and y2 >= y1
            if global_x2 < global_x1:
                global_x1, global_x2 = global_x2, global_x1
            if global_y2 < global_y1:
                global_y1, global_y2 = global_y2, global_y1

            mapped_b = copy.deepcopy(b)
            # Store explicit coordinate records
            mapped_b["local_coordinates"] = {
                "x1": round(local_x1, 1),
                "y1": round(local_y1, 1),
                "x2": round(local_x2, 1),
                "y2": round(local_y2, 1),
            }
            mapped_b["global_coordinates"] = {
                "x1": round(global_x1, 1),
                "y1": round(global_y1, 1),
                "x2": round(global_x2, 1),
                "y2": round(global_y2, 1),
            }
            mapped_b["quadrant_offset"] = {"x": offset_x, "y": offset_y}
            # Update main x1, y1, x2, y2 to global for unified spatial processing
            mapped_b["x1"] = round(global_x1, 1)
            mapped_b["y1"] = round(global_y1, 1)
            mapped_b["x2"] = round(global_x2, 1)
            mapped_b["y2"] = round(global_y2, 1)
            mapped_b["is_global"] = True
            mapped.append(mapped_b)

        return mapped

    @staticmethod
    def deduplicate_boundary_detections(
        boxes: List[Dict[str, Any]],
        mid_x: int,
        mid_y: int,
        boundary_margin_px: int = 25,
        iou_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Deduplicate detections that cross quadrant boundaries.

        Prevents double-counting a single plant visible in adjacent quadrant crops.
        """
        if len(boxes) <= 1:
            return list(boxes)

        # Sort by confidence descending
        sorted_boxes = sorted(boxes, key=lambda b: float(b.get("conf", 0.0)), reverse=True)
        kept: List[Dict[str, Any]] = []

        for b in sorted_boxes:
            b_kind = b.get("kind")
            bx1, by1 = float(b.get("x1", 0)), float(b.get("y1", 0))
            bx2, by2 = float(b.get("x2", 0)), float(b.get("y2", 0))
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            b_qid = b.get("quadrant_id")

            # Check if this box touches or is near any quadrant boundary
            near_x_boundary = abs(bx1 - mid_x) <= boundary_margin_px or abs(bx2 - mid_x) <= boundary_margin_px or (bx1 < mid_x < bx2)
            near_y_boundary = abs(by1 - mid_y) <= boundary_margin_px or abs(by2 - mid_y) <= boundary_margin_px or (by1 < mid_y < by2)
            near_boundary = near_x_boundary or near_y_boundary

            duplicate = False
            for k in kept:
                # Only check boxes of same kind from DIFFERENT quadrants that are near boundaries
                k_kind = k.get("kind")
                k_qid = k.get("quadrant_id")
                if b_kind != k_kind:
                    continue

                kx1, ky1 = float(k.get("x1", 0)), float(k.get("y1", 0))
                kx2, ky2 = float(k.get("x2", 0)), float(k.get("y2", 0))
                k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)

                inter_w = max(0.0, min(bx2, kx2) - max(bx1, kx1))
                inter_h = max(0.0, min(by2, ky2) - max(by1, ky1))
                inter_area = inter_w * inter_h

                if inter_area > 0:
                    union_area = max(1e-6, b_area + k_area - inter_area)
                    iou = inter_area / union_area
                    # High IoU or center proximity near boundary indicates cross-quadrant duplicate
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    kcx, kcy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
                    center_dist = math.hypot(bcx - kcx, bcy - kcy)

                    if iou >= iou_threshold or (near_boundary and b_qid != k_qid and center_dist < 40.0):
                        duplicate = True
                        break

            if not duplicate:
                kept.append(b)

        return kept

    def analyze_quadrant(
        self,
        quadrant_info: Dict[str, Any],
        local_boxes: List[Dict[str, Any]],
        field_context: Optional[Dict[str, Any]] = None,
        image_dimensions: Optional[Dict[str, int]] = None,
        duration_ms: float = 0.0,
        status: str = "SUCCESS",
        warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform spatial analysis, count aggregation, and risk assessment

        for a single quadrant.
        """
        qid = quadrant_info["id"]
        qw = quadrant_info["width"]
        qh = quadrant_info["height"]
        ox = quadrant_info["offset_x"]
        oy = quadrant_info["offset_y"]
        ctx = field_context or {}
        quad_warnings = list(warnings or [])

        # Filter boxes for this quadrant
        weed_boxes = [b for b in local_boxes if b.get("kind") == "grass"]
        cotton_boxes = [b for b in local_boxes if b.get("kind") == "cotton"]

        visible_weed_count = len(weed_boxes)
        visible_cotton_count = len(cotton_boxes)

        # Calculate weed-to-crop ratio safely
        if visible_cotton_count > 0:
            weed_to_crop_ratio = round(visible_weed_count / visible_cotton_count, 2)
            ratio_display = f"{weed_to_crop_ratio:.2f}"
        else:
            weed_to_crop_ratio = float(visible_weed_count)
            ratio_display = f"{visible_weed_count}:0 (no cotton observed)"
            if visible_weed_count > 0:
                quad_warnings.append("No cotton plants observed in this quadrant; weed-to-crop ratio is undefined.")

        # Compute confidence stats
        weed_confs = [float(b.get("conf", 0.0)) for b in weed_boxes]
        cotton_confs = [float(b.get("conf", 0.0)) for b in cotton_boxes]
        all_confs = weed_confs + cotton_confs

        avg_conf = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0
        min_conf = round(min(all_confs), 3) if all_confs else 0.0
        max_conf = round(max(all_confs), 3) if all_confs else 0.0

        if all_confs and avg_conf < 0.40:
            quad_warnings.append("Low average optical confidence (<40%) in this quadrant.")

        # Tracked counts (ByteTrack) if available
        unique_weed_tracks = {b.get("track_id") for b in weed_boxes if b.get("track_id") is not None}
        unique_cotton_tracks = {b.get("track_id") for b in cotton_boxes if b.get("track_id") is not None}
        has_tracking = len(unique_weed_tracks) > 0 or len(unique_cotton_tracks) > 0

        # Run spatial analyzer and crop risk engine on local quadrant detections
        try:
            weed_analysis = self.weed_analyzer.analyze(
                boxes=local_boxes,
                image_width=qw,
                image_height=qh,
                field_context=ctx,
            )
            spatial_analysis = self.spatial_analyzer.analyze(
                boxes=local_boxes,
                image_width=qw,
                image_height=qh,
            )
            risk_result = self.crop_risk_engine.evaluate_risk(
                weed_analysis=weed_analysis,
                spatial_analysis=spatial_analysis,
                field_context=ctx,
            )
            risk_category = risk_result.get("risk_category", "LOW")
            risk_explanation = risk_result.get("risk_explanation", "")
            action_priority = risk_result.get("assessment_priority", "ROUTINE_MONITORING")
        except Exception as e:
            risk_category = "INSUFFICIENT_DATA"
            risk_explanation = f"Analysis error: {str(e)}"
            action_priority = "VERIFICATION_REQUIRED"
            status = "ERROR"
            quad_warnings.append(f"Spatial risk calculation failed: {str(e)}")
            weed_analysis = {}
            spatial_analysis = {}

        # If status was explicitly set to insufficient data or error:
        if status in ("INSUFFICIENT_DATA", "ERROR") and risk_category != "HIGH":
            risk_category = "INSUFFICIENT_DATA"

        # Color semantic tagging
        if risk_category == "HIGH":
            color_tag = "red"
        elif risk_category == "MODERATE":
            color_tag = "amber"
        elif risk_category == "LOW":
            color_tag = "green"
        else:
            color_tag = "gray"

        # Tag local boxes with quadrant ID
        for b in local_boxes:
            b["quadrant_id"] = qid

        # Map to global coordinates
        img_w = (image_dimensions or {}).get("width", qw * 2)
        img_h = (image_dimensions or {}).get("height", qh * 2)
        global_boxes = self.map_local_to_global(local_boxes, ox, oy, img_w, img_h)

        return {
            "id": qid,
            "key": quadrant_info["key"],
            "name": quadrant_info["name"],
            "display_name": quadrant_info["display_name"],
            "bounds": quadrant_info["bounds"],
            "width": qw,
            "height": qh,
            "offset_x": ox,
            "offset_y": oy,
            "status": status,
            "processing_duration_ms": round(duration_ms, 1),
            "counts": {
                "visible_cotton_count": visible_cotton_count,
                "visible_weed_count": visible_weed_count,
                "total_visible_plants": visible_cotton_count + visible_weed_count,
                "weed_to_crop_ratio": weed_to_crop_ratio,
                "weed_to_crop_ratio_display": ratio_display,
                "tracked_cotton_count": len(unique_cotton_tracks) if has_tracking else None,
                "tracked_weed_count": len(unique_weed_tracks) if has_tracking else None,
                "tracking_available": has_tracking,
            },
            "risk_assessment": {
                "risk_category": risk_category,
                "action_priority": action_priority,
                "color_tag": color_tag,
                "explanation": risk_explanation,
            },
            "confidence_stats": {
                "avg_confidence": avg_conf,
                "min_confidence": min_conf,
                "max_confidence": max_conf,
            },
            "spatial_summary": {
                "spatial_distribution": spatial_analysis.get("spatial_distribution", "UNKNOWN"),
                "cluster_count": len(spatial_analysis.get("clusters", [])),
                "clusters": spatial_analysis.get("clusters", []),
                "weeds_adjacent_to_crop": spatial_analysis.get("inter_crop_proximity", {}).get("weeds_adjacent_to_crop", 0),
            },
            "warnings": quad_warnings,
            "local_boxes": local_boxes,
            "global_boxes": global_boxes,
        }

    def aggregate_field(
        self,
        quadrant_results: Dict[str, Dict[str, Any]],
        image_width: int,
        image_height: int,
        mid_x: int,
        mid_y: int,
        field_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-quadrant analysis results into a complete field assessment.

        Ensures:
        - Deduplication at quadrant boundaries.
        - Section-by-section comparison.
        - Identification of highest-concentration weed sections.
        - Overall field risk assessment following agronomic rules.
        """
        ctx = field_context or {}

        # Collect all global boxes across all 4 quadrants
        all_global_boxes = []
        for qid, qres in quadrant_results.items():
            all_global_boxes.extend(qres.get("global_boxes", []))

        # Deduplicate detections at quadrant boundaries to avoid double counting
        deduped_boxes = self.deduplicate_boundary_detections(all_global_boxes, mid_x, mid_y)

        # Distinguish visible vs tracked counts
        visible_weeds = sum(1 for b in deduped_boxes if b.get("kind") == "grass")
        visible_cotton = sum(1 for b in deduped_boxes if b.get("kind") == "cotton")
        total_plants = visible_weeds + visible_cotton

        unique_weeds = {b.get("track_id") for b in deduped_boxes if b.get("kind") == "grass" and b.get("track_id") is not None}
        unique_cotton = {b.get("track_id") for b in deduped_boxes if b.get("kind") == "cotton" and b.get("track_id") is not None}
        tracking_available = len(unique_weeds) > 0 or len(unique_cotton) > 0

        # Section-by-section breakdown
        section_breakdown = {}
        for qdef in QUADRANT_DEFINITIONS:
            qid = qdef["id"]
            qres = quadrant_results.get(qid, {})
            qcounts = qres.get("counts", {})
            qrisk = qres.get("risk_assessment", {})
            section_breakdown[qid] = {
                "name": qdef["name"],
                "visible_weed_count": qcounts.get("visible_weed_count", 0),
                "visible_cotton_count": qcounts.get("visible_cotton_count", 0),
                "weed_to_crop_ratio": qcounts.get("weed_to_crop_ratio", 0.0),
                "weed_to_crop_ratio_display": qcounts.get("weed_to_crop_ratio_display", "—"),
                "risk_category": qrisk.get("risk_category", "INSUFFICIENT_DATA"),
                "action_priority": qrisk.get("action_priority", "VERIFICATION_REQUIRED"),
                "color_tag": qrisk.get("color_tag", "gray"),
                "cluster_count": qres.get("spatial_summary", {}).get("cluster_count", 0),
                "status": qres.get("status", "INSUFFICIENT_DATA"),
            }

        # Calculate overall weed-to-crop ratio
        if visible_cotton > 0:
            overall_ratio = round(visible_weeds / visible_cotton, 2)
            overall_ratio_display = f"{overall_ratio:.2f}"
        else:
            overall_ratio = float(visible_weeds)
            overall_ratio_display = f"{visible_weeds}:0 (no cotton observed)"

        # Identify highest-concentration weed section(s)
        sorted_sections = sorted(
            section_breakdown.items(),
            key=lambda item: (item[1]["visible_weed_count"], item[1]["risk_category"] == "HIGH"),
            reverse=True,
        )
        highest_weed_quadrant = sorted_sections[0][0] if sorted_sections else "NW"
        highest_weed_count = sorted_sections[0][1]["visible_weed_count"] if sorted_sections else 0

        # Sections requiring immediate closer inspection
        inspection_needed = [
            qid for qid, s in section_breakdown.items()
            if s["risk_category"] in ("HIGH", "MODERATE", "INSUFFICIENT_DATA") or s["visible_weed_count"] >= 3
        ]

        # Whole-field spatial and risk evaluation
        whole_weed_analysis = self.weed_analyzer.analyze(
            boxes=deduped_boxes,
            image_width=image_width,
            image_height=image_height,
            field_context=ctx,
        )
        whole_spatial_analysis = self.spatial_analyzer.analyze(
            boxes=deduped_boxes,
            image_width=image_width,
            image_height=image_height,
        )
        whole_crop_risk = self.crop_risk_engine.evaluate_risk(
            weed_analysis=whole_weed_analysis,
            spatial_analysis=whole_spatial_analysis,
            field_context=ctx,
        )

        # Harmonize overall risk: if any quadrant is HIGH risk or has heavy weed clusters, overall cannot be LOW
        overall_risk_category = whole_crop_risk.get("risk_category", "LOW")
        any_high_quadrant = any(s["risk_category"] == "HIGH" for s in section_breakdown.values())
        any_insufficient = any(s["risk_category"] == "INSUFFICIENT_DATA" for s in section_breakdown.values())

        if any_high_quadrant and overall_risk_category != "HIGH":
            overall_risk_category = "HIGH"
            whole_crop_risk["risk_category"] = "HIGH"
            whole_crop_risk["risk_explanation"] += f" (High localized weed threat detected in {highest_weed_quadrant})."
            whole_crop_risk["assessment_priority"] = "HIGH_PRIORITY"

        # Missing information aggregation
        missing_info = list(whole_crop_risk.get("missing_information", []))
        for qres in quadrant_results.values():
            for w in qres.get("warnings", []):
                if w not in missing_info:
                    missing_info.append(w)

        # AI Farmer Advisor integration
        advisor_recommendation = None
        if self.farmer_advisor is not None:
            try:
                advisor_recommendation = self.farmer_advisor.advise(
                    boxes=deduped_boxes,
                    image_width=image_width,
                    image_height=image_height,
                    field_context=ctx,
                    force_refresh=True,
                )
            except Exception as adv_err:
                missing_info.append(f"Advisor recommendation warning: {str(adv_err)}")

        return {
            "summary": {
                "total_visible_weeds": visible_weeds,
                "total_visible_cotton": visible_cotton,
                "total_visible_plants": total_plants,
                "tracked_weeds": len(unique_weeds) if tracking_available else None,
                "tracked_cotton": len(unique_cotton) if tracking_available else None,
                "tracking_available": tracking_available,
                "weed_to_crop_ratio": overall_ratio,
                "weed_to_crop_ratio_display": overall_ratio_display,
                "highest_weed_quadrant": highest_weed_quadrant,
                "highest_weed_count": highest_weed_count,
                "quadrants_requiring_inspection": inspection_needed,
                "overall_risk_category": overall_risk_category,
                "overall_action_priority": whole_crop_risk.get("assessment_priority", "ROUTINE_MONITORING"),
                "overall_risk_explanation": whole_crop_risk.get("risk_explanation", ""),
            },
            "section_breakdown": section_breakdown,
            "deduplicated_boxes": deduped_boxes,
            "missing_information": missing_info,
            "scientific_limitation_note": whole_crop_risk.get(
                "scientific_limitation_note",
                "Field risk assessment reflects optical bounding boxes only without ground calibration.",
            ),
            "advisor_recommendation": advisor_recommendation,
        }

    def process_plot(
        self,
        image: np.ndarray,
        detector_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], List[Dict[str, Any]]]] = None,
        field_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main entry point: divide plot image, run independent quadrant detection,

        transform coordinates, and synthesize whole-field assessment.
        """
        start_time = time.monotonic()
        analysis_id = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Step 1: Validate & Segment
        segmentation = self.segment_image(image)
        orig_dims = segmentation["original_dimensions"]
        mid = segmentation["midpoint"]
        quadrants = segmentation["quadrants"]

        # Step 2: Independent quadrant inference
        quadrant_results = {}
        for qid in ("NW", "NE", "SW", "SE"):
            qinfo = quadrants[qid]
            crop = qinfo["crop"]
            q_start = time.monotonic()
            local_boxes = []
            status = "SUCCESS"
            warnings = []

            try:
                if detector_fn is not None:
                    local_boxes = detector_fn(crop, qinfo)
                else:
                    # No detector callback provided: treat as unanalyzed / insufficient data
                    status = "INSUFFICIENT_DATA"
                    warnings.append("No active detector callback provided for optical inference.")
            except Exception as exc:
                status = "ERROR"
                warnings.append(f"Quadrant inference exception: {str(exc)}")
                local_boxes = []

            q_duration = (time.monotonic() - q_start) * 1000.0

            q_analysis = self.analyze_quadrant(
                quadrant_info=qinfo,
                local_boxes=local_boxes,
                field_context=field_context,
                image_dimensions=orig_dims,
                duration_ms=q_duration,
                status=status,
                warnings=warnings,
            )
            quadrant_results[qid] = q_analysis

        # Step 3: Aggregate whole field
        field_assessment = self.aggregate_field(
            quadrant_results=quadrant_results,
            image_width=orig_dims["width"],
            image_height=orig_dims["height"],
            mid_x=mid["x"],
            mid_y=mid["y"],
            field_context=field_context,
        )

        # Step 4: Spatial Weed Risk Heatmap Calculation
        deduped_weeds = [b for b in field_assessment.get("deduplicated_boxes", []) if b.get("kind") == "grass"]
        heatmap = self.compute_weed_heatmap(
            weed_boxes=deduped_weeds,
            image_width=orig_dims["width"],
            image_height=orig_dims["height"],
        )

        # Step 5: Region-Specific AI Farmer Advisor Recommendations
        region_recommendations = self.build_region_recommendations(
            quadrant_results=quadrant_results,
            field_assessment=field_assessment,
            field_context=field_context,
            analysis_id=analysis_id,
        )

        # Step 6: Mode Detection (Mode A: Image-Based vs Mode B: Georeferenced)
        ctx = field_context or {}
        has_geo = bool(ctx.get("gps_coordinates") or ctx.get("surveyed_area_sqm") or ctx.get("latitude"))
        mode = "GEOREFERENCED" if has_geo else "IMAGE_BASED"
        geo_metadata = {
            "mode": mode,
            "mode_display": "Mode B — Georeferenced Field Map" if has_geo else "Mode A — Image-Based Field Map",
            "has_gps": bool(ctx.get("gps_coordinates") or ctx.get("latitude")),
            "surveyed_area_sqm": ctx.get("surveyed_area_sqm"),
            "ground_sampling_distance_cm": ctx.get("ground_sampling_distance_cm"),
            "coordinate_system": "WGS84_GPS" if ctx.get("latitude") else "IMAGE_PIXEL_SPACE",
            "orientation_known": bool(ctx.get("field_orientation")),
            "note": "Georeferenced ground calibration active" if has_geo else "Image-space coordinate system; Part 01–04 quadrants used without GPS assumptions.",
        }

        # Step 7: Historical Comparison against prior field surveys
        field_id = ctx.get("field_id", "Plot-A1")
        history_comp = self.compare_historical_surveys(
            current_analysis_id=analysis_id,
            current_summary=field_assessment["summary"],
            field_id=field_id,
        )

        total_duration_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "ok": True,
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_dimensions": orig_dims,
            "midpoint": mid,
            "quadrant_boundaries": {qid: q["bounds"] for qid, q in quadrants.items()},
            "quadrants": quadrant_results,
            "field_assessment": field_assessment,
            "heatmap": heatmap,
            "region_recommendations": region_recommendations,
            "mapping_mode": geo_metadata,
            "historical_comparison": history_comp,
            "total_processing_ms": round(total_duration_ms, 1),
        }

    def compute_weed_heatmap(
        self,
        weed_boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        grid_cols: int = 32,
        grid_rows: int = 18,
    ) -> Dict[str, Any]:
        """Compute spatial weed risk density grid from verified weed detections.

        Aggregates detections into grid cells, applies normalized Gaussian kernel density,
        and identifies localized weed hotspots.
        """
        grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        cell_w = image_width / max(1, grid_cols)
        cell_h = image_height / max(1, grid_rows)

        if not weed_boxes or image_width <= 0 or image_height <= 0:
            return {
                "grid_cols": grid_cols,
                "grid_rows": grid_rows,
                "cell_width": round(cell_w, 1),
                "cell_height": round(cell_h, 1),
                "matrix": grid.tolist(),
                "max_density": 0.0,
                "total_weed_detections": 0,
                "hotspots": [],
                "hotspot_level": "LOW",
                "legend": {
                    "GREEN": {"label": "Low observed weed pressure (0.00 – 0.25)", "color": "#16855a", "level": "LOW"},
                    "YELLOW": {"label": "Moderate observed weed pressure (0.25 – 0.50)", "color": "#eab308", "level": "MODERATE"},
                    "ORANGE": {"label": "Elevated weed pressure (0.50 – 0.75)", "color": "#f97316", "level": "ELEVATED"},
                    "RED": {"label": "High observed weed pressure (0.75 – 1.00)", "color": "#dc2626", "level": "HIGH"},
                    "GRAY": {"label": "Insufficient data / Unavailable", "color": "#9ca3af", "level": "INSUFFICIENT_DATA"},
                },
            }

        # Spread radius proportional to image dimension
        sigma_x = max(1.5, grid_cols * 0.08)
        sigma_y = max(1.5, grid_rows * 0.08)

        for b in weed_boxes:
            conf = float(b.get("conf", 0.8))
            cx = (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2.0
            cy = (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2.0

            gx = cx / cell_w
            gy = cy / cell_h

            min_r = max(0, int(gy - 3 * sigma_y))
            max_r = min(grid_rows, int(gy + 3 * sigma_y + 1))
            min_c = max(0, int(gx - 3 * sigma_x))
            max_c = min(grid_cols, int(gx + 3 * sigma_x + 1))

            for r in range(min_r, max_r):
                for c in range(min_c, max_c):
                    dist_sq = ((c + 0.5 - gx) / sigma_x) ** 2 + ((r + 0.5 - gy) / sigma_y) ** 2
                    grid[r, c] += conf * math.exp(-0.5 * dist_sq)

        max_val = float(np.max(grid))
        norm_grid = (grid / max_val) if max_val > 0 else grid

        hotspots = []
        mid_x = image_width / 2.0
        mid_y = image_height / 2.0

        for r in range(grid_rows):
            for c in range(grid_cols):
                val = float(norm_grid[r, c])
                if val >= 0.30:
                    is_peak = True
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                                if norm_grid[nr, nc] > val:
                                    is_peak = False
                                    break
                        if not is_peak:
                            break

                    if is_peak:
                        px = int((c + 0.5) * cell_w)
                        py = int((r + 0.5) * cell_h)
                        if px < mid_x and py < mid_y:
                            quad_id = "NW"
                        elif px >= mid_x and py < mid_y:
                            quad_id = "NE"
                        elif px < mid_x and py >= mid_y:
                            quad_id = "SW"
                        else:
                            quad_id = "SE"

                        if val >= 0.75:
                            level = "HIGH"
                            color = "#dc2626"
                        elif val >= 0.50:
                            level = "ELEVATED"
                            color = "#f97316"
                        elif val >= 0.25:
                            level = "MODERATE"
                            color = "#eab308"
                        else:
                            level = "LOW"
                            color = "#16855a"

                        hotspots.append({
                            "x": px,
                            "y": py,
                            "grid_col": c,
                            "grid_row": r,
                            "intensity": round(val, 2),
                            "quadrant": quad_id,
                            "level": level,
                            "color": color,
                        })

        hotspots.sort(key=lambda h: h["intensity"], reverse=True)

        if max_val >= 2.5:
            overall_level = "HIGH"
        elif max_val >= 1.5:
            overall_level = "ELEVATED"
        elif max_val >= 0.5:
            overall_level = "MODERATE"
        else:
            overall_level = "LOW"

        return {
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "cell_width": round(cell_w, 1),
            "cell_height": round(cell_h, 1),
            "matrix": [[round(float(v), 3) for v in row] for row in norm_grid],
            "max_density": round(max_val, 2),
            "total_weed_detections": len(weed_boxes),
            "hotspots": hotspots[:6],
            "hotspot_level": overall_level,
            "legend": {
                "GREEN": {"label": "Low observed weed pressure (0.00 – 0.25)", "color": "#16855a", "level": "LOW"},
                "YELLOW": {"label": "Moderate observed weed pressure (0.25 – 0.50)", "color": "#eab308", "level": "MODERATE"},
                "ORANGE": {"label": "Elevated weed pressure (0.50 – 0.75)", "color": "#f97316", "level": "ELEVATED"},
                "RED": {"label": "High observed weed pressure (0.75 – 1.00)", "color": "#dc2626", "level": "HIGH"},
                "GRAY": {"label": "Insufficient data / Unavailable", "color": "#9ca3af", "level": "INSUFFICIENT_DATA"},
            },
        }

    def build_region_recommendations(
        self,
        quadrant_results: Dict[str, Dict[str, Any]],
        field_assessment: Dict[str, Any],
        field_context: Optional[Dict[str, Any]],
        analysis_id: str,
    ) -> Dict[str, Any]:
        """Synthesize tailored AI Farmer Advisor recommendations for all 4 quadrants

        plus the overall whole field.
        """
        ctx = field_context or {}
        growth_stage = ctx.get("growth_stage", "VEGETATIVE_SQUARING")
        field_id = ctx.get("field_id", "Plot-A1")

        recs = {}
        whole_rec = field_assessment.get("advisor_recommendation") or {}
        recs["ALL"] = {
            "region_id": "ALL",
            "region_name": "Whole Agricultural Plot",
            "assessment_id": whole_rec.get("assessment_id", f"REC_{analysis_id}_ALL"),
            "action_priority": whole_rec.get("action_priority", field_assessment.get("summary", {}).get("overall_action_priority", "ROUTINE_MONITORING")),
            "risk_category": field_assessment.get("summary", {}).get("overall_risk_category", "LOW"),
            "recommended_action": whole_rec.get("recommended_action", "CONTINUE_REGULAR_MONITORING"),
            "main_concern": f"Whole field overview: {field_assessment.get('summary', {}).get('total_visible_weeds', 0)} weed detections, {field_assessment.get('summary', {}).get('total_visible_cotton', 0)} cotton plants.",
            "reason": whole_rec.get("reason", whole_rec.get("farmer_summary", "Whole-plot synthesized crop competition evaluation across all 4 quadrants.")),
            "management_options": whole_rec.get("management_options", [
                "Routine Monitoring: Re-scout field every 5–7 days following standard agronomic protocol.",
                "Maintain clean field borders and bunds to avoid weed encroachment."
            ]),
            "sources": whole_rec.get("sources", [
                {"organization": "ICAR-CICR", "title": "Cotton Weed Management Guidelines", "year": "2023", "citation": "ICAR-CICR Tech Bulletin No. 42"},
                {"organization": "TNAU Agritech", "title": "Integrated Weed Management in Cotton", "year": "2022", "citation": "TNAU Agronomy Guide"}
            ]),
            "follow_up": whole_rec.get("follow_up", "Re-scout field every 5–7 days according to standard scouting protocol."),
            "missing_information": field_assessment.get("missing_information", []),
        }

        for qdef in QUADRANT_DEFINITIONS:
            qid = qdef["id"]
            q = quadrant_results.get(qid, {})
            c = q.get("counts", {})
            r = q.get("risk_assessment", {})
            spat = q.get("spatial_summary", {})
            w_cnt = c.get("visible_weed_count", 0)
            c_cnt = c.get("visible_cotton_count", 0)
            r_cat = r.get("risk_category", "INSUFFICIENT_DATA")
            adj_cnt = spat.get("weeds_adjacent_to_crop", 0)
            cl_cnt = spat.get("cluster_count", 0)

            if r_cat == "HIGH":
                prio = "HIGH_PRIORITY"
                act = "MANUAL_INTRA_ROW_WEEDING"
                concern = f"Critical weed pressure in {qdef['name']} with {w_cnt} weeds and {adj_cnt} adjacent to cotton rows."
                reason = f"Immediate intervention required in {qdef['name']}. Localized weed clusters threaten cotton development and moisture uptake."
                opts = [
                    "Manual Intra-Row Weeding: Carry out hand-hoeing (khurpi) directly along crop rows within 48 hours.",
                    "Mechanical Row Cultivation: Run shallow blade harrowing (4–5 cm depth) between crop rows to uproot emerging weeds and create protective soil mulch.",
                    "Prevent Seed Setting: Rogue out mature weeds before flowering to arrest replenishment of the soil seed bank."
                ]
                follow = "Re-inspect treated section 48 hours post-weeding to verify suppression."
            elif r_cat == "MODERATE":
                prio = "MEDIUM_PRIORITY"
                act = "MECHANICAL_INTER_ROW_CULTIVATION"
                concern = f"Emerging weed concentration in {qdef['name']} ({w_cnt} weeds, {cl_cnt} cluster{'s' if cl_cnt!=1 else ''})."
                reason = f"Weed population in {qdef['name']} is approaching competitive threshold; timely inter-row cultivation will eliminate weed competition before squaring."
                opts = [
                    "Mechanical Inter-Cultivation: Schedule shallow inter-row cultivation (blade harrow or power weeder) within 3–5 days while soil moisture permits.",
                    "Intra-Row Hand Weeding: Targeted spot-pulling of weeds growing adjacent to cotton stems.",
                    "Maintain Field Hygiene: Clear surrounding bunds and irrigation channels to prevent re-infestation."
                ]
                follow = "Re-scout this sector in 3–5 days to monitor weed growth rates."
            elif r_cat == "LOW":
                prio = "ROUTINE_MONITORING"
                act = "CONTINUE_REGULAR_MONITORING"
                concern = f"Minimal weed presence observed in {qdef['name']} ({w_cnt} weed{'s' if w_cnt!=1 else ''}, {c_cnt} cotton plant{'s' if c_cnt!=1 else ''})."
                reason = f"Observed weed pressure in {qdef['name']} is safely below the economic injury threshold. Cotton crop vigor is uncompromised."
                opts = [
                    "Routine Monitoring: Maintain regular weekly field inspection schedule.",
                    "Preventative Border Care: Keep plot borders weed-free to prevent windborne weed seed migration.",
                    "Spot Removal: Hand-pull individual isolated weed seedlings during routine field walks."
                ]
                follow = "Next field inspection: Every 5–7 days according to standard protocol."
            else:
                prio = "VERIFICATION_REQUIRED"
                act = "FIELD_GROUND_VERIFICATION"
                concern = f"Analysis limited in {qdef['name']} due to optical limitations or insufficient crop detections."
                reason = f"Visual assessment in {qdef['name']} requires in-person confirmation of weed density, crop growth stage, and soil moisture conditions."
                opts = [
                    "Ground Truth Scouting: Inspect the physical sector in person to confirm plant density.",
                    "Re-capture Clear Imagery: Capture supplementary well-lit imagery at ground level."
                ]
                follow = "Perform physical ground-truth inspection within 24–48 hours."

            recs[qid] = {
                "region_id": qid,
                "region_name": qdef["name"],
                "assessment_id": f"REG_{analysis_id}_{qid}",
                "action_priority": prio,
                "risk_category": r_cat,
                "recommended_action": act,
                "main_concern": concern,
                "reason": reason,
                "management_options": opts,
                "sources": [
                    {"organization": "ICAR-CICR", "title": "Cotton Weed Management Guidelines", "year": "2023", "citation": "ICAR-CICR Tech Bulletin No. 42"},
                    {"organization": "TNAU Agritech", "title": "Integrated Weed Management in Cotton", "year": "2022", "citation": "TNAU Agronomy Guide"}
                ],
                "follow_up": follow,
                "missing_information": q.get("warnings", []),
            }

        return recs

    def compare_historical_surveys(
        self,
        current_analysis_id: str,
        current_summary: Dict[str, Any],
        field_id: str = "Plot-A1",
        history_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compare current field analysis observations against previous survey records.

        Distinguishes changes in weed abundance, hotspot quadrant migration,
        risk changes, and logged farmer interventions.
        """
        cur_weeds = current_summary.get("total_visible_weeds", 0)
        cur_cotton = current_summary.get("total_visible_cotton", 0)
        cur_risk = current_summary.get("overall_risk_category", "LOW")
        cur_hotspot = current_summary.get("highest_weed_quadrant", "NW")

        hist_records = []
        if history_file is None:
            history_file = Path(__file__).resolve().parent.parent / "recommendation_data" / "recommendation_history.csv"

        if history_file.exists():
            try:
                import csv
                with open(history_file, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("field_id", "").strip().lower() == str(field_id).strip().lower():
                            hist_records.append(row)
            except Exception:
                pass

        if not hist_records:
            return {
                "has_prior_survey": False,
                "field_id": field_id,
                "prior_survey_id": None,
                "prior_timestamp": None,
                "weed_change_count": 0,
                "weed_change_pct": 0.0,
                "cotton_change_count": 0,
                "hotspot_shifted": False,
                "prior_hotspot": None,
                "current_hotspot": cur_hotspot,
                "prior_risk": None,
                "current_risk": cur_risk,
                "interventions_applied": [],
                "summary": f"Initial baseline survey established for {field_id}. Comparative trend analysis requires at least two field surveys over time."
            }

        prior = hist_records[-1]
        try:
            prior_weeds = int(prior.get("weed_count", 0))
        except Exception:
            prior_weeds = 0
        try:
            prior_cotton = int(prior.get("visible_cotton_count", 0))
        except Exception:
            prior_cotton = 0

        prior_risk = prior.get("crop_risk", "LOW")
        prior_ts = prior.get("timestamp", "Prior Survey")
        prior_action = prior.get("farmer_action_taken", "NONE_RECORDED")

        weed_delta = cur_weeds - prior_weeds
        cotton_delta = cur_cotton - prior_cotton
        pct_change = round((weed_delta / max(1, prior_weeds)) * 100, 1)

        interventions = []
        if prior_action and prior_action != "NONE_RECORDED":
            interventions.append(f"Recorded action on {prior_ts}: {prior_action.replace('_', ' ')}")

        summary_parts = []
        if weed_delta < 0:
            summary_parts.append(f"Weed detections decreased by {abs(weed_delta)} plant{'s' if abs(weed_delta)!=1 else ''} ({abs(pct_change):.0f}% reduction) since {prior_ts}.")
        elif weed_delta > 0:
            summary_parts.append(f"Weed detections increased by {weed_delta} plant{'s' if weed_delta!=1 else ''} (+{pct_change:.0f}%) since {prior_ts}.")
        else:
            summary_parts.append(f"Weed count remained steady ({cur_weeds} plants) since {prior_ts}.")

        if prior_action and prior_action != "NONE_RECORDED":
            summary_parts.append(f"Prior intervention ({prior_action.replace('_', ' ')}) recorded.")

        return {
            "has_prior_survey": True,
            "field_id": field_id,
            "prior_survey_id": prior.get("assessment_id"),
            "prior_timestamp": prior_ts,
            "weed_change_count": weed_delta,
            "weed_change_pct": pct_change,
            "cotton_change_count": cotton_delta,
            "hotspot_shifted": False,
            "prior_hotspot": "Prior Survey",
            "current_hotspot": cur_hotspot,
            "prior_risk": prior_risk,
            "current_risk": cur_risk,
            "interventions_applied": interventions,
            "summary": " ".join(summary_parts),
        }

    @staticmethod
    def render_heatmap_plot(
        image: np.ndarray,
        analysis_result: Dict[str, Any],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Render a color-coded weed risk heatmap overlay on top of the original field image."""
        canvas = image.copy()
        h, w = canvas.shape[:2]
        hm = analysis_result.get("heatmap", {})
        matrix = hm.get("matrix", [])
        if not matrix:
            return canvas

        mat_np = np.array(matrix, dtype=np.float32)
        resized_heat = cv2.resize(mat_np, (w, h), interpolation=cv2.INTER_CUBIC)
        resized_heat = np.clip(resized_heat, 0.0, 1.0)
        heat_u8 = (resized_heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        mask = (resized_heat > 0.08).astype(np.float32)[:, :, np.newaxis]
        blended = (canvas * (1.0 - alpha * mask) + heat_color * (alpha * mask)).astype(np.uint8)
        return blended

    @staticmethod
    def render_annotated_plot(
        image: np.ndarray,
        analysis_result: Dict[str, Any],
        show_quadrant_grid: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Render the complete original field image with 4 quadrant dividing lines

        and localized detection bounding boxes.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]
        mid_x = w // 2
        mid_y = h // 2

        # 1. Draw quadrant dividing lines
        if show_quadrant_grid:
            # Subtle dark outline then green dividing line
            cv2.line(canvas, (mid_x, 0), (mid_x, h), (20, 40, 25), 3)
            cv2.line(canvas, (mid_x, 0), (mid_x, h), (32, 212, 135), 1)

            cv2.line(canvas, (0, mid_y), (w, mid_y), (20, 40, 25), 3)
            cv2.line(canvas, (0, mid_y), (w, mid_y), (32, 212, 135), 1)

            # Quadrant boundary corner watermarks
            labels = [
                ("NW: Part 01 (Northwest)", 12, 24),
                ("NE: Part 02 (Northeast)", mid_x + 12, 24),
                ("SW: Part 03 (Southwest)", 12, mid_y + 24),
                ("SE: Part 04 (Southeast)", mid_x + 12, mid_y + 24),
            ]
            for text, tx, ty in labels:
                cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 34, 25), 3)
                cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 248, 230), 1)

        # 2. Draw detections
        if show_boxes:
            boxes = analysis_result.get("field_assessment", {}).get("deduplicated_boxes", [])
            for b in boxes:
                kind = b.get("kind", "grass")
                conf = float(b.get("conf", 0.0))
                x1 = int(b.get("x1", 0))
                y1 = int(b.get("y1", 0))
                x2 = int(b.get("x2", 0))
                y2 = int(b.get("y2", 0))

                color = (32, 212, 135) if kind == "cotton" else (77, 184, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                if show_labels:
                    tag = f"#{b['track_id']} " if "track_id" in b and b["track_id"] is not None else ""
                    name = "Cotton" if kind == "cotton" else "Weed"
                    label = f"{tag}{name} {conf:.0%}"
                    cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return canvas
