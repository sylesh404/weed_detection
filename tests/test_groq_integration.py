"""Comprehensive Unit and Integration Tests for Groq API Recommendation Engine.

Covers:
1. Groq client initialization with and without API key.
2. Missing API-key handling and fallback behavior.
3. Successful model responses using mocked API calls.
4. Invalid/malformed JSON responses.
5. Unsupported risk categories (ensuring Groq cannot override deterministic crop-risk).
6. Missing field data (no fabricated values).
7. Agricultural source attribution (ICAR-CICR, TNAU, PAU, CIB&RC).
8. Safety validation (spray hoods, PPE, drift limits, dosage prohibition).
9. Provider failures (timeouts, rate limits, network errors) with clean fallback.
10. Analyze Field frontend/API integration.
11. Recommendation history compatibility (JSON & CSV stores).
12. Preservation of all existing API endpoints.
"""
import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys
from pathlib import Path

# Ensure package root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommendation.groq_service import GroqAdvisorService, GroqServiceError
from recommendation.graph import RecommendationWorkflow, AdvisorState
from recommendation.recommendation_engine import SafetyValidator, HybridRecommendationEngine
from recommendation.knowledge_base import AgriculturalKnowledgeBase
from app import app


class TestGroqIntegration(unittest.TestCase):
    """Test suite for Groq API integration and guardrails."""

    def setUp(self):
        self.mock_boxes = [
            {"kind": "cotton", "conf": 0.92, "x1": 100, "y1": 100, "x2": 250, "y2": 250, "track_id": 1, "class_name": "cotton"},
            {"kind": "grass", "conf": 0.88, "x1": 180, "y1": 150, "x2": 240, "y2": 210, "track_id": 2, "class_name": "grass"},
            {"kind": "grass", "conf": 0.79, "x1": 320, "y1": 400, "x2": 390, "y2": 480, "track_id": 3, "class_name": "grass"},
        ]
        self.field_context = {
            "field_id": "Plot-UnitTest-1",
            "growth_stage": "VEGETATIVE_SQUARING",
        }

    # 1. Groq client initialization
    def test_client_initialization_with_and_without_key(self):
        # With explicit key
        service_with_key = GroqAdvisorService(api_key="gsk_mock_valid_test_key_12345", model="openai/gpt-oss-120b")
        self.assertTrue(service_with_key.is_available)
        self.assertEqual(service_with_key.model, "openai/gpt-oss-120b")
        status = service_with_key.get_status()
        self.assertTrue(status["configured"])
        self.assertTrue(status["has_api_key"])
        # Ensure secret key is NOT leaked in status
        self.assertNotIn("gsk_mock_valid_test_key_12345", str(status))

        # Without key
        service_no_key = GroqAdvisorService(api_key="")
        self.assertFalse(service_no_key.is_available)
        no_key_status = service_no_key.get_status()
        self.assertFalse(no_key_status["configured"])
        self.assertFalse(no_key_status["has_api_key"])

    # 2. Missing API-key handling & fallback
    def test_missing_api_key_fallback(self):
        service = GroqAdvisorService(api_key="")
        fallback_rec = {
            "crop_risk": "MODERATE",
            "action_priority": "MEDIUM_PRIORITY",
            "recommended_action": "SCOUT_AND_SCHEDULE_INTER_CULTIVATION",
            "reason": "Deterministic rule evaluation.",
            "management_options": ["Mechanical inter-row cultivation."],
        }
        res = service.generate_recommendation(
            field_id="Plot-A1",
            growth_stage="VEGETATIVE_SQUARING",
            weed_analysis={"counts": {"visible_weed_count": 2}},
            spatial_analysis={},
            crop_risk={"risk_category": "MODERATE"},
            knowledge_entries=[],
            missing_info=[],
            deterministic_fallback=fallback_rec,
        )
        self.assertEqual(res["crop_risk"], "MODERATE")
        self.assertEqual(res["ai_provider"], "Deterministic Fallback")
        self.assertEqual(res["ai_status"], "API_KEY_NOT_CONFIGURED")

    # 3. Successful model responses using mocked API calls
    def test_successful_mocked_model_response(self):
        service = GroqAdvisorService(api_key="mock_key")
        mock_groq_json = {
            "field_id": "Plot-UnitTest-1",
            "summary": "Moderate weed presence detected near developing cotton seedlings.",
            "risk_level": "MODERATE",
            "observations": ["2 grass weeds identified with 1 adjacent to cotton stem"],
            "recommended_action": "Targeted Inter-Row Cultivation",
            "reasoning": "Inter-row cultivation at vegetative stage reduces weed competition before flowering.",
            "management_options": [
                {
                    "title": "Mechanical Inter-Cultivation",
                    "description": "Shallow blade harrowing at 4-5 cm depth between rows.",
                    "source": "ICAR-CICR Cotton Weed Guide (2023)"
                }
            ],
            "missing_information": [],
            "safety_warnings": ["Avoid cultivating wet soil to prevent compaction"],
            "sources": [
                {
                    "title": "Cotton Weed Management",
                    "organization": "ICAR-CICR",
                    "publication": "Technical Bulletin",
                    "year": "2023",
                    "citation": "ICAR-CICR Technical Bulletin (2023)",
                    "key_takeaway": "Shallow cultivation preserves lateral feeder roots"
                }
            ],
            "limitations": ["Camera viewing angle limited to center row"]
        }

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_groq_json)
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(service._client.chat.completions, "create", return_value=mock_completion):
            res = service.generate_recommendation(
                field_id="Plot-UnitTest-1",
                growth_stage="VEGETATIVE_SQUARING",
                weed_analysis={"counts": {"visible_weed_count": 2, "visible_cotton_count": 1}},
                spatial_analysis={"inter_crop_proximity": {"weeds_adjacent_to_crop": 1}},
                crop_risk={"risk_category": "MODERATE"},
                knowledge_entries=[{"title": "Cotton Weed Guide", "source": {"organization": "ICAR-CICR"}}],
                missing_info=[],
                deterministic_fallback={"crop_risk": "MODERATE", "reason": "Fallback reason"},
            )

        self.assertEqual(res["ai_provider"], "Groq")
        self.assertEqual(res["risk_level"], "MODERATE")
        self.assertEqual(res["summary"], "Moderate weed presence detected near developing cotton seedlings.")
        self.assertIn("Mechanical Inter-Cultivation", res["management_options"][0])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["organization"], "ICAR-CICR")

    # 4. Invalid JSON responses handling
    def test_invalid_json_response_triggers_fallback(self):
        service = GroqAdvisorService(api_key="mock_key")
        mock_choice = MagicMock()
        mock_choice.message.content = "Malformed plain text without JSON..."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        fallback = {
            "crop_risk": "HIGH",
            "recommended_action": "INSPECT_AND_INITIATE_MANAGEMENT",
            "reason": "Fallback deterministic reason",
            "management_options": ["Hand weeding"],
        }

        with patch.object(service._client.chat.completions, "create", return_value=mock_completion):
            res = service.generate_recommendation(
                field_id="Plot-1",
                growth_stage="VEGETATIVE_SQUARING",
                weed_analysis={},
                spatial_analysis={},
                crop_risk={"risk_category": "HIGH"},
                knowledge_entries=[],
                missing_info=[],
                deterministic_fallback=fallback,
            )

        self.assertEqual(res["ai_provider"], "Deterministic Fallback")
        self.assertEqual(res["ai_status"], "FALLBACK_ON_ERROR")
        self.assertEqual(res["crop_risk"], "HIGH")

    # 5. Unsupported risk categories (Groq cannot override deterministic crop-risk engine)
    def test_groq_cannot_override_deterministic_risk(self):
        service = GroqAdvisorService(api_key="mock_key")
        # Groq hallucinates 'LOW' risk when the deterministic engine computed 'HIGH'
        mock_groq_json = {
            "field_id": "Plot-1",
            "summary": "Fields look clear.",
            "risk_level": "LOW",  # Mismatched!
            "observations": [],
            "recommended_action": "Do nothing",
            "reasoning": "Weeds seem low.",
            "management_options": [],
        }

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_groq_json)
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(service._client.chat.completions, "create", return_value=mock_completion):
            res = service.generate_recommendation(
                field_id="Plot-1",
                growth_stage="EMERGENCE_SEEDLING",
                weed_analysis={"counts": {"visible_weed_count": 10}},
                spatial_analysis={},
                crop_risk={"risk_category": "HIGH"},  # Deterministic authority
                knowledge_entries=[],
                missing_info=[],
                deterministic_fallback={"crop_risk": "HIGH", "reason": "High weed density"},
            )

        # The validator MUST enforce the deterministic risk level
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertEqual(res["crop_risk"], "HIGH")

    # 6. Missing field data handling (no fabricated values)
    def test_missing_field_data_communicates_limitations(self):
        service = GroqAdvisorService(api_key="mock_key")
        payload = service.build_structured_prompt_context(
            field_id="",
            growth_stage="UNKNOWN",
            weed_analysis={"counts": {"visible_weed_count": 0, "visible_cotton_count": 0}},
            spatial_analysis={},
            crop_risk={"risk_category": "INSUFFICIENT_DATA"},
            knowledge_entries=[],
            missing_info=["Crop growth stage is unspecified", "Tracked unique counts unavailable"],
        )
        self.assertEqual(payload["cotton_growth_stage"], "Not Specified")
        self.assertNotIn("unique_tracked_weed_count", payload)  # Must not invent tracked counts
        self.assertIn("missing_information_warnings", payload)
        self.assertEqual(len(payload["missing_information_warnings"]), 2)

    # 7. Agricultural source attribution (ICAR, TNAU, PAU, CIB&RC)
    def test_agricultural_source_attribution_integrity(self):
        kb = AgriculturalKnowledgeBase()
        entries = kb.retrieve(weed_type="grass", crop_stage="VEGETATIVE_SQUARING", limit=3)
        self.assertGreater(len(entries), 0)
        organizations = [e["source"]["organization"] for e in entries]
        valid_keywords = ["ICAR", "CICR", "DWR", "TNAU", "PAU", "CIB&RC", "Cotton"]
        for org in organizations:
            self.assertTrue(any(v in org for v in valid_keywords), f"Unknown org: {org}")

    # 8. Safety validation enforcement
    def test_safety_validator_enforces_spray_shield_and_ppe(self):
        validator = SafetyValidator()
        unsafe_rec = {
            "management_options": [
                "Apply post-emergence herbicide spray across the field.",
                "Shallow hoeing."
            ],
            "safety_disclaimers": []
        }
        validated = validator.validate(unsafe_rec, {"growth_stage": "VEGETATIVE_SQUARING"})
        # Herbicide option must be augmented with mandatory spray hood and CIB&RC check
        chem_option = validated["management_options"][0]
        self.assertIn("MANDATORY SAFETY", chem_option)
        self.assertIn("protective hoods", chem_option)
        self.assertIn("CIB&RC", chem_option)
        # Disclaimers must enforce wind and PPE
        disclaimers = " ".join(validated["safety_disclaimers"]).lower()
        self.assertTrue("personal protective equipment" in disclaimers or "ppe" in disclaimers)
        self.assertIn("wind", disclaimers)

    # 9. Provider failures (rate limits, timeouts) with fallback
    def test_provider_failure_fallback(self):
        from groq import RateLimitError
        service = GroqAdvisorService(api_key="mock_key")
        fallback = {
            "crop_risk": "MODERATE",
            "recommended_action": "SCOUT_AND_SCHEDULE_INTER_CULTIVATION",
            "reason": "Rule fallback",
        }

        with patch.object(service._client.chat.completions, "create", side_effect=RateLimitError("Rate limit reached", response=MagicMock(), body=None)):
            res = service.generate_recommendation(
                field_id="Plot-1",
                growth_stage="VEGETATIVE_SQUARING",
                weed_analysis={},
                spatial_analysis={},
                crop_risk={"risk_category": "MODERATE"},
                knowledge_entries=[],
                missing_info=[],
                deterministic_fallback=fallback,
            )

        self.assertEqual(res["ai_provider"], "Deterministic Fallback")
        self.assertEqual(res["ai_status"], "FALLBACK_ON_ERROR")

    # 10. LangGraph 9-node workflow integration
    def test_langgraph_workflow_node_execution_with_groq(self):
        mock_groq_json = {
            "field_id": "Plot-9Node",
            "summary": "Cotton plot shows low weed interference with good seedling spacing.",
            "risk_level": "LOW",
            "observations": ["1 grass weed", "3 cotton plants"],
            "recommended_action": "CONTINUE_REGULAR_MONITORING",
            "reasoning": "Weed pressure is below CPWC economic threshold.",
            "management_options": [{"title": "Scouting", "description": "Weekly monitoring", "source": "PAU Guide"}],
            "sources": [{"title": "PAU Weed Guide", "organization": "PAU", "publication": "Package of Practices", "year": "2023"}],
        }

        service = GroqAdvisorService(api_key="mock_key")
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_groq_json)
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(service._client.chat.completions, "create", return_value=mock_completion):
            workflow = RecommendationWorkflow(groq_service=service)
            output = workflow.run(
                boxes=self.mock_boxes,
                image_width=1280,
                image_height=720,
                field_context=self.field_context,
            )

        self.assertIn("crop_risk", output)
        self.assertIn("summary", output)
        self.assertEqual(output.get("ai_provider"), "Groq")
        self.assertIn("workflow_audit", output)
        self.assertTrue(output["workflow_audit"]["is_valid_input"])

    # 11. Recommendation history compatibility
    def test_recommendation_history_compatibility(self):
        from recommendation.recommendation_service import RecommendationService
        service = RecommendationService()
        status = service.get_status()
        self.assertTrue(status["ok"])
        self.assertEqual(status["status"], "OPERATIONAL")

    # 12. Preservation of all existing API endpoints (mocking external API calls)
    def test_all_recommendation_api_endpoints_functional(self):
        app.config["TESTING"] = True
        client = app.test_client()

        # 1. /api/recommendations/status
        res = client.get("/api/recommendations/status")
        self.assertEqual(res.status_code, 200)

        # 2. /api/recommendations/latest
        res = client.get("/api/recommendations/latest")
        self.assertEqual(res.status_code, 200)

        # 3. /api/recommendations/history
        res = client.get("/api/recommendations/history")
        self.assertEqual(res.status_code, 200)

        # 4. /api/recommendations/ask
        res = client.post("/api/recommendations/ask", json={"question": "When is the critical period of weed competition?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("grounded"))
        hl = data.get("headline", "").lower()
        self.assertTrue("critical" in hl and "period" in hl)

        # 5. /api/recommendations/analyze with mocked Groq call
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "field_id": "Plot-UnitTest-1",
            "summary": "Mocked field assessment summary",
            "risk_level": "MODERATE",
            "observations": ["Observed detections"],
            "recommended_action": "INTER_CULTIVATION",
            "reasoning": "Mocked reasoning",
            "management_options": [{"title": "Hoeing", "description": "Inter-row hoeing", "source": "ICAR"}],
            "sources": [{"title": "ICAR Guide", "organization": "ICAR-CICR"}],
        })
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("groq.resources.chat.completions.Completions.create", return_value=mock_completion):
            res = client.post("/api/recommendations/analyze", json={
                "boxes": self.mock_boxes,
                "width": 1280,
                "height": 720,
                "field_context": self.field_context,
            })
            self.assertEqual(res.status_code, 200)
            rec_data = res.get_json()
            self.assertTrue(rec_data.get("ok"))
            self.assertIn("recommendation", rec_data)


if __name__ == "__main__":
    unittest.main()
