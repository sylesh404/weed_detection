"""Unit and Integration Tests for Field Map & Weed Risk Intelligence in AgriSight AI."""
import unittest
import json
import base64
import numpy as np
import cv2
from pathlib import Path

from analysis.plot_segmenter import PlotSegmenter, QUADRANT_DEFINITIONS
import app as flask_app


class TestFieldMapWeedRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segmenter = PlotSegmenter()
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def test_01_compute_weed_heatmap_empty(self):
        """Heatmap with zero weeds should return a zeroed matrix without errors."""
        hm = self.segmenter.compute_weed_heatmap([], 1280, 720)
        self.assertIn("matrix", hm)
        self.assertEqual(len(hm["matrix"]), 18)
        self.assertEqual(len(hm["matrix"][0]), 32)
        self.assertEqual(len(hm["hotspots"]), 0)
        self.assertEqual(hm["max_density"], 0.0)

    def test_02_compute_weed_heatmap_hotspots_and_quadrants(self):
        """Heatmap with clustered weeds should compute Gaussian density and pinpoint hotspots."""
        # Clustered weeds in Northeast quadrant (x ~ 900, y ~ 150)
        weeds = [
            {"x1": 880, "y1": 130, "x2": 920, "y2": 170, "conf": 0.90},
            {"x1": 890, "y1": 140, "x2": 930, "y2": 180, "conf": 0.88},
            {"x1": 870, "y1": 120, "x2": 910, "y2": 160, "conf": 0.85},
            {"x1": 910, "y1": 150, "x2": 950, "y2": 190, "conf": 0.92},
            {"x1": 860, "y1": 110, "x2": 900, "y2": 150, "conf": 0.82},
        ]
        hm = self.segmenter.compute_weed_heatmap(weeds, 1280, 720)
        self.assertGreater(hm["max_density"], 0.0)
        self.assertGreater(len(hm["hotspots"]), 0)
        # Hotspot should be correctly identified in the Northeast quadrant
        primary_hotspot = hm["hotspots"][0]
        self.assertEqual(primary_hotspot["quadrant"], "NE")
        self.assertIn(primary_hotspot["level"], ["HIGH", "ELEVATED", "MODERATE"])
        self.assertIn("legend", hm)
        self.assertIn("GREEN", hm["legend"])
        self.assertIn("RED", hm["legend"])

    def test_03_region_recommendations_generation(self):
        """Region recommendations must be synthesized for all 4 quadrants and the whole field."""
        quadrants = {
            "NW": {
                "name": "Part 01 — Northwest",
                "counts": {"visible_cotton_count": 5, "visible_weed_count": 1, "weed_to_crop_ratio_display": "0.20"},
                "risk_assessment": {"risk_category": "LOW", "action_priority": "ROUTINE_MONITORING"},
                "spatial_summary": {"cluster_count": 0, "weeds_adjacent_to_crop": 0},
                "warnings": []
            },
            "NE": {
                "name": "Part 02 — Northeast",
                "counts": {"visible_cotton_count": 2, "visible_weed_count": 8, "weed_to_crop_ratio_display": "4.00"},
                "risk_assessment": {"risk_category": "HIGH", "action_priority": "IMMEDIATE_ACTION"},
                "spatial_summary": {"cluster_count": 2, "weeds_adjacent_to_crop": 3},
                "warnings": []
            },
            "SW": {
                "name": "Part 03 — Southwest",
                "counts": {"visible_cotton_count": 4, "visible_weed_count": 3, "weed_to_crop_ratio_display": "0.75"},
                "risk_assessment": {"risk_category": "MODERATE", "action_priority": "MEDIUM_PRIORITY"},
                "spatial_summary": {"cluster_count": 1, "weeds_adjacent_to_crop": 1},
                "warnings": []
            },
            "SE": {
                "name": "Part 04 — Southeast",
                "counts": {"visible_cotton_count": 6, "visible_weed_count": 0, "weed_to_crop_ratio_display": "0.00"},
                "risk_assessment": {"risk_category": "LOW", "action_priority": "ROUTINE_MONITORING"},
                "spatial_summary": {"cluster_count": 0, "weeds_adjacent_to_crop": 0},
                "warnings": []
            }
        }
        field_assessment = {
            "summary": {"overall_risk_category": "HIGH", "overall_action_priority": "IMMEDIATE_ACTION"},
            "advisor_recommendation": {"farmer_summary": "High weed pressure in NE sector."}
        }
        recs = self.segmenter.build_region_recommendations(
            quadrant_results=quadrants,
            field_assessment=field_assessment,
            field_context={"field_id": "Plot-A1", "growth_stage": "VEGETATIVE_SQUARING"},
            analysis_id="TEST_ANALYSIS_01"
        )
        for qid in ["ALL", "NW", "NE", "SW", "SE"]:
            self.assertIn(qid, recs)
            r = recs[qid]
            self.assertIn("risk_category", r)
            self.assertIn("main_concern", r)
            self.assertIn("recommended_action", r)
            self.assertIn("reason", r)
            self.assertIn("management_options", r)
            self.assertIn("sources", r)
            self.assertGreater(len(r["management_options"]), 0)

        # NE quadrant should have HIGH risk recommendations
        self.assertEqual(recs["NE"]["risk_category"], "HIGH")
        self.assertIn("WEEDING", recs["NE"]["recommended_action"])

    def test_04_historical_comparison_handling(self):
        """Historical comparison should honestly distinguish prior records vs no prior survey."""
        # 1. Non-existent field ID
        comp_none = self.segmenter.compare_historical_surveys(
            current_analysis_id="CURR_01",
            current_summary={"total_visible_weeds": 10, "overall_risk_category": "HIGH", "highest_weed_quadrant": "NE"},
            field_id="FIELD_DOES_NOT_EXIST_XYZ"
        )
        self.assertFalse(comp_none["has_prior_survey"])
        self.assertIsNone(comp_none["prior_survey_id"])

        # 2. Existing field ID from recommendation_history.csv (PLOT_TEST_101)
        comp_exist = self.segmenter.compare_historical_surveys(
            current_analysis_id="CURR_02",
            current_summary={"total_visible_weeds": 4, "total_visible_cotton": 5, "overall_risk_category": "MODERATE", "highest_weed_quadrant": "SW"},
            field_id="PLOT_TEST_101"
        )
        self.assertTrue(comp_exist["has_prior_survey"])
        self.assertIsNotNone(comp_exist["prior_timestamp"])
        self.assertIn("summary", comp_exist)

    def test_05_api_field_map_analyze_endpoint(self):
        """POST /api/field-map/analyze endpoint must execute and return heatmap and region recommendations."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        _, img_buf = cv2.imencode(".jpg", img)
        b64_str = base64.b64encode(img_buf.tobytes()).decode("utf-8")

        res = self.client.post("/api/field-map/analyze", json={
            "image_base64": b64_str,
            "test_boxes": [
                {"kind": "cotton", "conf": 0.90, "x1": 50, "y1": 50, "x2": 150, "y2": 150},
                {"kind": "grass", "conf": 0.85, "x1": 350, "y1": 50, "x2": 420, "y2": 120},
                {"kind": "grass", "conf": 0.80, "x1": 380, "y1": 80, "x2": 440, "y2": 140},
            ]
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("ok", True))
        self.assertIn("heatmap", data)
        self.assertIn("region_recommendations", data)
        self.assertIn("historical_comparison", data)
        self.assertIn("image_urls", data)
        self.assertIn("heatmap", data["image_urls"])

    def test_06_dashboard_html_field_map_section(self):
        """Dashboard HTML must feature Field Map & Weed Risk with filters, legend, and Area A + Area B."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("Field Map & Weed Risk", html)
        self.assertIn("Visualize weed hotspots, assess field conditions, and identify where action is needed.", html)
        self.assertIn("chkPlotHeatmap", html)
        self.assertIn("chkPlotCotton", html)
        self.assertIn("chkPlotWeeds", html)
        self.assertIn("chkPlotGrid", html)
        self.assertIn("Weed Pressure Scale:", html)
        self.assertIn("advRegionTitle", html)
        self.assertIn("advRegionConcernText", html)
        self.assertIn("advNextAction", html)
        self.assertIn("advHistoryBox", html)
        self.assertIn("Mode A: Image-Based Field Map", html)


if __name__ == "__main__":
    unittest.main()
