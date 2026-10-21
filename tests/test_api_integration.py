"""Integration tests for AgriSight AI Flask Application & Recommendation Endpoints."""
import unittest
import sys
import json
from pathlib import Path

# Add project root to path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from app import app


class TestApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_existing_endpoints_preserved(self):
        # 1. Root route
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        # Existing elements must be present
        self.assertIn("Cotton & Grass Detection", html)
        self.assertIn("Live camera feed", html)
        self.assertIn("Detection controls", html)
        self.assertIn("Detection history", html)
        # New additive advisor element must also be present
        self.assertIn("AI Farmer Advisor & Crop Risk Assessment", html)

        # 2. Existing /api/status endpoint
        status_res = self.client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        data = status_res.get_json()
        self.assertIn("active", data)
        self.assertIn("cotton_count", data)
        self.assertIn("grass_count", data)

        # 3. Existing /api/logs endpoint
        logs_res = self.client.get("/api/logs")
        self.assertEqual(logs_res.status_code, 200)
        logs_data = logs_res.get_json()
        self.assertTrue(logs_data.get("ok"))

    def test_recommendation_status_endpoint(self):
        res = self.client.get("/api/recommendations/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("status"), "OPERATIONAL")

    def test_recommendation_analyze_endpoint(self):
        payload = {
            "boxes": [
                {"kind": "grass", "conf": 0.85, "x1": 50, "y1": 50, "x2": 120, "y2": 120, "track_id": 1, "class_name": "grass"},
                {"kind": "cotton", "conf": 0.90, "x1": 200, "y1": 200, "x2": 300, "y2": 300, "track_id": 2, "class_name": "cotton"},
            ],
            "width": 1280,
            "height": 720,
            "field_context": {
                "growth_stage": "VEGETATIVE_SQUARING",
                "field_id": "PLOT_TEST_101"
            }
        }
        res = self.client.post("/api/recommendations/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("ok"))
        rec = data.get("recommendation", {})
        self.assertIn("assessment_id", rec)
        self.assertEqual(rec.get("field_id"), "PLOT_TEST_101")
        self.assertIn(rec.get("action_priority"), ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "ROUTINE_MONITORING"))
        self.assertTrue(len(rec.get("management_options", [])) > 0)

        # Check that latest endpoint now returns this recommendation
        latest_res = self.client.get("/api/recommendations/latest")
        self.assertEqual(latest_res.status_code, 200)
        latest_data = latest_res.get_json()
        self.assertTrue(latest_data.get("available"))
        self.assertEqual(latest_data["recommendation"]["assessment_id"], rec["assessment_id"])

        # Test recording farmer action
        action_payload = {
            "assessment_id": rec["assessment_id"],
            "action_taken": "MECHANICAL_INTER_CULTIVATION",
            "notes": "Harrowed inter-row weeds cleanly."
        }
        act_res = self.client.post("/api/recommendations/action", json=action_payload)
        self.assertEqual(act_res.status_code, 200)
        self.assertTrue(act_res.get_json().get("ok"))

    def test_recommendation_ask_endpoint(self):
        q_payload = {"question": "Can I spray herbicide without protective hoods?"}
        res = self.client.post("/api/recommendations/ask", json=q_payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("ok"))
        self.assertIn("hood", data.get("answer", "").lower())


if __name__ == "__main__":
    unittest.main()
