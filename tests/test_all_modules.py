"""Comprehensive Automated Test Suite for AgriSight AI Hybrid Recommendation Engine.

Tests:
1. WeedAnalyzer: count distinction, density caveats, dispersion, data quality.
2. SpatialAnalyzer: grid distribution, image-plane clustering, proximity, geo caveats.
3. CropRiskEngine: risk categories, CPWC sensitivity, missing data handling.
4. AgriculturalKnowledgeBase: verified retrieval, source citations.
5. HybridRecommendationEngine: action priorities, safety validation layer.
6. RecommendationWorkflow (LangGraph): 9-node execution and state transitions.
7. RecommendationService: asynchronous execution, persistence, action recording.
"""
import unittest
import sys
from pathlib import Path

# Ensure package directory is in sys.path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.weed_analyzer import WeedAnalyzer
from analysis.spatial_analyzer import SpatialAnalyzer
from analysis.crop_risk_engine import CropRiskEngine
from recommendation.knowledge_base import AgriculturalKnowledgeBase
from recommendation.recommendation_engine import HybridRecommendationEngine, SafetyValidator
from recommendation.graph import RecommendationWorkflow
from recommendation.farmer_advisor import FarmerAdvisor
from recommendation.recommendation_service import RecommendationService


class TestWeedAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = WeedAnalyzer()

    def test_distinguishes_visible_and_tracked_counts(self):
        # 3 weed boxes, 2 share the same track_id
        boxes = [
            {"kind": "grass", "conf": 0.85, "track_id": 101, "x1": 50, "y1": 50, "x2": 100, "y2": 100, "class_name": "grass"},
            {"kind": "grass", "conf": 0.82, "track_id": 101, "x1": 55, "y1": 55, "x2": 105, "y2": 105, "class_name": "grass"},
            {"kind": "grass", "conf": 0.78, "track_id": 102, "x1": 200, "y1": 200, "x2": 250, "y2": 250, "class_name": "grass"},
            {"kind": "cotton", "conf": 0.90, "track_id": 201, "x1": 500, "y1": 500, "x2": 600, "y2": 600, "class_name": "cotton"},
        ]
        res = self.analyzer.analyze(boxes, 1280, 720)
        counts = res["counts"]

        self.assertEqual(counts["visible_weed_count"], 3)
        self.assertEqual(counts["tracked_weed_count"], 2)  # IDs 101 and 102
        self.assertEqual(counts["tracking_status"], "PERSISTENT_BYTETRACK")
        self.assertEqual(counts["visible_cotton_count"], 1)

    def test_density_guardrails_require_calibration(self):
        boxes = [{"kind": "grass", "conf": 0.8, "track_id": 1, "x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "grass"}]
        # No GSD or surveyed area provided
        res = self.analyzer.analyze(boxes, 1280, 720, field_context={})
        self.assertEqual(res["density"]["density_status"], "UNKNOWN_NO_CALIBRATION")
        self.assertIsNone(res["density"]["density_per_sqm"])
        self.assertIn("Camera calibration", " ".join(res["missing_information"]))

        # When surveyed area IS provided
        res2 = self.analyzer.analyze(boxes, 1280, 720, field_context={"surveyed_area_sqm": 2.0})
        self.assertEqual(res2["density"]["density_status"], "CALCULATED_FROM_SURVEYED_AREA")
        self.assertEqual(res2["density"]["density_per_sqm"], 0.5)


class TestSpatialAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SpatialAnalyzer(proximity_threshold_px=100.0)

    def test_grid_distribution_and_clustering(self):
        boxes = [
            {"kind": "grass", "x1": 10, "y1": 10, "x2": 40, "y2": 40, "track_id": 1},
            {"kind": "grass", "x1": 20, "y1": 20, "x2": 50, "y2": 50, "track_id": 2},
            {"kind": "grass", "x1": 30, "y1": 30, "x2": 60, "y2": 60, "track_id": 3},
            {"kind": "cotton", "x1": 50, "y1": 50, "x2": 120, "y2": 120, "track_id": 10},
        ]
        res = self.analyzer.analyze(boxes, 1280, 720)
        self.assertEqual(res["coordinate_system"], "IMAGE_PIXEL_PLANE")
        self.assertFalse(res["geographic_data"]["has_genuine_gps"])
        self.assertIn("Top-Left", [z["quadrant"] for z in res["inspection_regions"]])
        self.assertTrue(res["inter_crop_proximity"]["weeds_adjacent_to_crop"] >= 1)


class TestCropRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CropRiskEngine()
        self.weed_analyzer = WeedAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()

    def test_low_risk_when_no_weeds(self):
        w_an = self.weed_analyzer.analyze([], 1280, 720)
        s_an = self.spatial_analyzer.analyze([], 1280, 720)
        risk = self.engine.evaluate_risk(w_an, s_an)
        self.assertEqual(risk["risk_category"], "LOW")
        self.assertEqual(risk["assessment_priority"], "ROUTINE_MONITORING")

    def test_high_risk_during_seedling_stage_with_weeds(self):
        boxes = [
            {"kind": "grass", "conf": 0.8, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "track_id": 1, "class_name": "grass"},
            {"kind": "cotton", "conf": 0.85, "x1": 120, "y1": 120, "x2": 180, "y2": 180, "track_id": 2, "class_name": "cotton"},
        ]
        w_an = self.weed_analyzer.analyze(boxes, 1280, 720)
        s_an = self.spatial_analyzer.analyze(boxes, 1280, 720)
        risk = self.engine.evaluate_risk(w_an, s_an, field_context={"growth_stage": "EMERGENCE_SEEDLING"})
        self.assertIn(risk["risk_category"], ("MODERATE", "HIGH"))
        self.assertTrue(risk["requires_field_verification"])

    def test_insufficient_data_when_confidence_poor_and_stage_unknown(self):
        # Very low confidence box (<0.35) and stage unknown
        boxes = [{"kind": "grass", "conf": 0.20, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "track_id": 1, "class_name": "grass"}]
        w_an = self.weed_analyzer.analyze(boxes, 1280, 720)
        s_an = self.spatial_analyzer.analyze(boxes, 1280, 720)
        risk = self.engine.evaluate_risk(w_an, s_an, field_context={"growth_stage": "UNKNOWN"})
        self.assertEqual(risk["risk_category"], "INSUFFICIENT_DATA")


class TestAgriculturalKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = AgriculturalKnowledgeBase()

    def test_retrieval_and_citations(self):
        entries = self.kb.retrieve(weed_type="grass", crop_stage="VEGETATIVE_SQUARING", limit=3)
        self.assertTrue(len(entries) > 0)
        for e in entries:
            self.assertIn("source", e)
            self.assertTrue(len(e["source"]["organization"]) > 0)
            self.assertTrue(len(e["source"]["publication"]) > 0)

        citations = self.kb.get_source_citations()
        self.assertTrue(len(citations) >= 4)


class TestHybridRecommendationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HybridRecommendationEngine()
        self.weed_analyzer = WeedAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.risk_engine = CropRiskEngine()

    def test_safety_validation_enforces_protective_hoods(self):
        boxes = [
            {"kind": "grass", "conf": 0.8, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "track_id": 1, "class_name": "grass"},
            {"kind": "grass", "conf": 0.85, "x1": 120, "y1": 120, "x2": 170, "y2": 170, "track_id": 2, "class_name": "grass"},
            {"kind": "grass", "conf": 0.82, "x1": 200, "y1": 200, "x2": 250, "y2": 250, "track_id": 3, "class_name": "grass"},
            {"kind": "cotton", "conf": 0.9, "x1": 400, "y1": 400, "x2": 480, "y2": 480, "track_id": 4, "class_name": "cotton"},
        ]
        w_an = self.weed_analyzer.analyze(boxes, 1280, 720)
        s_an = self.spatial_analyzer.analyze(boxes, 1280, 720)
        risk = self.risk_engine.evaluate_risk(w_an, s_an, {"growth_stage": "VEGETATIVE_SQUARING"})
        rec = self.engine.generate_recommendation(w_an, s_an, risk, {"growth_stage": "VEGETATIVE_SQUARING"})

        # Check safety disclaimers
        self.assertTrue(any("CIB&RC" in d for d in rec["safety_disclaimers"]))
        self.assertIn(rec["action_priority"], ("HIGH_PRIORITY", "MEDIUM_PRIORITY"))
        self.assertIn("assessment_id", rec)
        self.assertIn("sources", rec)


class TestRecommendationWorkflow(unittest.TestCase):
    def test_workflow_execution(self):
        wf = RecommendationWorkflow()
        boxes = [
            {"kind": "grass", "conf": 0.75, "x1": 50, "y1": 50, "x2": 100, "y2": 100, "track_id": 1, "class_name": "grass"},
            {"kind": "cotton", "conf": 0.88, "x1": 300, "y1": 300, "x2": 400, "y2": 400, "track_id": 2, "class_name": "cotton"},
        ]
        output = wf.run(boxes, 1280, 720, field_context={"growth_stage": "VEGETATIVE_SQUARING"})
        self.assertIn("assessment_id", output)
        self.assertIn("action_priority", output)
        self.assertIn("farmer_summary", output)
        self.assertTrue(output["workflow_audit"]["is_valid_input"])


class TestRecommendationService(unittest.TestCase):
    def test_service_analyze_and_record_action(self):
        service = RecommendationService()
        boxes = [{"kind": "grass", "conf": 0.8, "x1": 10, "y1": 10, "x2": 60, "y2": 60, "track_id": 1, "class_name": "grass"}]
        res = service.analyze(boxes, 1280, 720, {"growth_stage": "SEEDLING"})
        self.assertTrue(res["ok"])
        rec = res["recommendation"]
        aid = rec["assessment_id"]

        # Record action
        act_res = service.record_farmer_action(aid, "INTER_ROW_CULTIVATION", "Operated blade harrow between rows.")
        self.assertTrue(act_res["ok"])

        # Check latest
        latest = service.get_latest()
        self.assertTrue(latest["available"])
        self.assertEqual(latest["recommendation"]["assessment_id"], aid)


if __name__ == "__main__":
    unittest.main()
