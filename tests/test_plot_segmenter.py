"""Automated Test Suite for Four-Part Agricultural Plot Segmentation & Analysis.

Verifies:
1. Correct four-way image segmentation.
2. Odd image dimensions.
3. Empty and invalid image input.
4. Correct quadrant identifiers.
5. Correct local-to-global coordinate transformation.
6. Bounding boxes at image edges.
7. No accidental double-counting across quadrant boundaries.
8. Correct aggregation of quadrant counts.
9. Correct handling of zero cotton detections.
10. Partial quadrant-analysis failures.
11. Risk assessment with missing information.
12. API validation and error responses.
13. Preservation of existing camera and recommendation routes.
14. Dashboard components and UI integration.
"""
import unittest
import sys
import json
from pathlib import Path

import numpy as np
import cv2

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.plot_segmenter import PlotSegmenter, QUADRANT_DEFINITIONS
from app import app


class TestPlotSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = PlotSegmenter()

    def test_01_four_way_image_segmentation(self):
        """Verify image is divided into 4 quadrants covering 100% with no pixel loss."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Give distinct pixel markers to four quadrants
        img[0:360, 0:640] = 10     # NW
        img[0:360, 640:1280] = 20  # NE
        img[360:720, 0:640] = 30   # SW
        img[360:720, 640:1280] = 40 # SE

        res = self.segmenter.segment_image(img)
        self.assertEqual(res["original_dimensions"]["width"], 1280)
        self.assertEqual(res["original_dimensions"]["height"], 720)
        self.assertEqual(res["midpoint"]["x"], 640)
        self.assertEqual(res["midpoint"]["y"], 360)

        quads = res["quadrants"]
        self.assertEqual(len(quads), 4)

        # Verify NW
        nw = quads["NW"]
        self.assertEqual(nw["width"], 640)
        self.assertEqual(nw["height"], 360)
        self.assertEqual(nw["offset_x"], 0)
        self.assertEqual(nw["offset_y"], 0)
        self.assertEqual(nw["crop"].shape, (360, 640, 3))
        self.assertTrue(np.all(nw["crop"] == 10))

        # Verify NE
        ne = quads["NE"]
        self.assertEqual(ne["width"], 640)
        self.assertEqual(ne["height"], 360)
        self.assertEqual(ne["offset_x"], 640)
        self.assertEqual(ne["offset_y"], 0)
        self.assertEqual(ne["crop"].shape, (360, 640, 3))
        self.assertTrue(np.all(ne["crop"] == 20))

        # Verify SW
        sw = quads["SW"]
        self.assertEqual(sw["width"], 640)
        self.assertEqual(sw["height"], 360)
        self.assertEqual(sw["offset_x"], 0)
        self.assertEqual(sw["offset_y"], 360)
        self.assertEqual(sw["crop"].shape, (360, 640, 3))
        self.assertTrue(np.all(sw["crop"] == 30))

        # Verify SE
        se = quads["SE"]
        self.assertEqual(se["width"], 640)
        self.assertEqual(se["height"], 360)
        self.assertEqual(se["offset_x"], 640)
        self.assertEqual(se["offset_y"], 360)
        self.assertEqual(se["crop"].shape, (360, 640, 3))
        self.assertTrue(np.all(se["crop"] == 40))

        # Total pixels must equal original image pixels
        total_quad_pixels = sum(q["width"] * q["height"] for q in quads.values())
        self.assertEqual(total_quad_pixels, 1280 * 720)

    def test_02_odd_image_dimensions(self):
        """Verify odd image dimensions are partitioned without pixel loss or overlap."""
        img = np.ones((703, 1001, 3), dtype=np.uint8)
        res = self.segmenter.segment_image(img)
        quads = res["quadrants"]

        self.assertEqual(res["original_dimensions"]["width"], 1001)
        self.assertEqual(res["original_dimensions"]["height"], 703)
        self.assertEqual(res["midpoint"]["x"], 500)
        self.assertEqual(res["midpoint"]["y"], 351)

        # NW: 500 x 351
        self.assertEqual(quads["NW"]["width"], 500)
        self.assertEqual(quads["NW"]["height"], 351)
        # NE: 501 x 351 (500 + 501 = 1001)
        self.assertEqual(quads["NE"]["width"], 501)
        self.assertEqual(quads["NE"]["height"], 351)
        # SW: 500 x 352 (351 + 352 = 703)
        self.assertEqual(quads["SW"]["width"], 500)
        self.assertEqual(quads["SW"]["height"], 352)
        # SE: 501 x 352
        self.assertEqual(quads["SE"]["width"], 501)
        self.assertEqual(quads["SE"]["height"], 352)

        # Total pixel area check
        total_pixels = sum(q["width"] * q["height"] for q in quads.values())
        self.assertEqual(total_pixels, 1001 * 703)

    def test_03_empty_and_invalid_image_inputs(self):
        """Verify invalid or empty inputs raise clear ValueErrors."""
        with self.assertRaises(ValueError):
            self.segmenter.validate_image(None)

        with self.assertRaises(ValueError):
            self.segmenter.validate_image("not an array")

        with self.assertRaises(ValueError):
            self.segmenter.validate_image(np.zeros((0, 0, 3)))

        with self.assertRaises(ValueError):
            self.segmenter.validate_image(np.zeros((2, 2, 3)))  # too small

        with self.assertRaises(ValueError):
            self.segmenter.validate_image(np.zeros((10, 10, 3, 3)))  # 4D unsupported

    def test_04_correct_quadrant_identifiers(self):
        """Verify stable identifiers and names."""
        keys = [q["key"] for q in QUADRANT_DEFINITIONS]
        ids = [q["id"] for q in QUADRANT_DEFINITIONS]
        names = [q["name"] for q in QUADRANT_DEFINITIONS]

        self.assertEqual(ids, ["NW", "NE", "SW", "SE"])
        self.assertEqual(keys, ["part_01_northwest", "part_02_northeast", "part_03_southwest", "part_04_southeast"])
        self.assertIn("Part 01 — Northwest", names)
        self.assertIn("Part 02 — Northeast", names)
        self.assertIn("Part 03 — Southwest", names)
        self.assertIn("Part 04 — Southeast", names)

    def test_05_local_to_global_coordinate_transformation(self):
        """Verify local quadrant coordinates are transformed to global plot coordinates."""
        local_boxes = [
            {"kind": "cotton", "conf": 0.9, "x1": 10, "y1": 20, "x2": 60, "y2": 70},
            {"kind": "grass", "conf": 0.8, "x1": 100, "y1": 150, "x2": 140, "y2": 190},
        ]
        # Map from SE quadrant of 1280x720 image (offset_x=640, offset_y=360)
        global_boxes = self.segmenter.map_local_to_global(local_boxes, offset_x=640, offset_y=360, image_width=1280, image_height=720)

        self.assertEqual(len(global_boxes), 2)
        b1 = global_boxes[0]
        self.assertEqual(b1["x1"], 650)
        self.assertEqual(b1["y1"], 380)
        self.assertEqual(b1["x2"], 700)
        self.assertEqual(b1["y2"], 430)
        self.assertEqual(b1["local_coordinates"]["x1"], 10)
        self.assertEqual(b1["local_coordinates"]["y1"], 20)
        self.assertTrue(b1["is_global"])

        # Test idempotence: mapping already-global boxes must not add offset again
        remap = self.segmenter.map_local_to_global(global_boxes, offset_x=640, offset_y=360, image_width=1280, image_height=720)
        self.assertEqual(remap[0]["x1"], 650)
        self.assertEqual(remap[0]["y1"], 380)

    def test_06_bounding_boxes_at_image_edges(self):
        """Verify boxes that touch or slightly exceed edge boundaries are clipped."""
        boxes = [
            {"kind": "cotton", "x1": -10, "y1": -5, "x2": 50, "y2": 50},
            {"kind": "grass", "x1": 600, "y1": 300, "x2": 700, "y2": 400},
        ]
        mapped = self.segmenter.map_local_to_global(boxes, offset_x=640, offset_y=360, image_width=1280, image_height=720)

        b1 = mapped[0]
        self.assertGreaterEqual(b1["x1"], 0.0)
        self.assertGreaterEqual(b1["y1"], 0.0)

        b2 = mapped[1]
        self.assertLessEqual(b2["x2"], 1280.0)
        self.assertLessEqual(b2["y2"], 720.0)

    def test_07_no_double_counting_across_boundaries(self):
        """Verify plant detected across boundary line is deduplicated."""
        mid_x = 640
        mid_y = 360
        # Plant spanning dividing line x=640:
        # Detected in NW with global coordinates (625..655, 100..140)
        # Detected in NE with global coordinates (630..660, 105..145)
        boxes = [
            {"kind": "grass", "conf": 0.85, "x1": 625, "y1": 100, "x2": 655, "y2": 140, "quadrant_id": "NW"},
            {"kind": "grass", "conf": 0.82, "x1": 630, "y1": 105, "x2": 660, "y2": 145, "quadrant_id": "NE"},
            {"kind": "cotton", "conf": 0.90, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "quadrant_id": "NW"},
        ]
        deduped = self.segmenter.deduplicate_boundary_detections(boxes, mid_x=mid_x, mid_y=mid_y)
        # The 2 duplicate grass boxes must be merged to 1
        grass_count = sum(1 for b in deduped if b["kind"] == "grass")
        self.assertEqual(grass_count, 1)
        self.assertEqual(len(deduped), 2)

    def test_08_aggregation_of_quadrant_counts(self):
        """Verify whole field aggregation accurately combines quadrant counts."""
        q_results = {
            "NW": {
                "id": "NW", "key": "part_01_northwest", "name": "Part 01 — Northwest",
                "width": 640, "height": 360, "offset_x": 0, "offset_y": 0, "status": "SUCCESS",
                "counts": {"visible_cotton_count": 2, "visible_weed_count": 1, "weed_to_crop_ratio": 0.5, "weed_to_crop_ratio_display": "0.50"},
                "risk_assessment": {"risk_category": "LOW", "action_priority": "ROUTINE_MONITORING", "color_tag": "green"},
                "spatial_summary": {"cluster_count": 0, "weeds_adjacent_to_crop": 0},
                "global_boxes": [
                    {"kind": "cotton", "conf": 0.9, "x1": 100, "y1": 100, "x2": 150, "y2": 150, "quadrant_id": "NW"},
                    {"kind": "cotton", "conf": 0.9, "x1": 200, "y1": 100, "x2": 250, "y2": 150, "quadrant_id": "NW"},
                    {"kind": "grass", "conf": 0.8, "x1": 400, "y1": 100, "x2": 450, "y2": 150, "quadrant_id": "NW"},
                ],
            },
            "NE": {
                "id": "NE", "key": "part_02_northeast", "name": "Part 02 — Northeast",
                "width": 640, "height": 360, "offset_x": 640, "offset_y": 0, "status": "SUCCESS",
                "counts": {"visible_cotton_count": 1, "visible_weed_count": 4, "weed_to_crop_ratio": 4.0, "weed_to_crop_ratio_display": "4.00"},
                "risk_assessment": {"risk_category": "HIGH", "action_priority": "HIGH_PRIORITY", "color_tag": "red"},
                "spatial_summary": {"cluster_count": 1, "weeds_adjacent_to_crop": 2},
                "global_boxes": [
                    {"kind": "cotton", "conf": 0.9, "x1": 700, "y1": 100, "x2": 750, "y2": 150, "quadrant_id": "NE"},
                    {"kind": "grass", "conf": 0.8, "x1": 710, "y1": 110, "x2": 740, "y2": 140, "quadrant_id": "NE"},
                    {"kind": "grass", "conf": 0.8, "x1": 750, "y1": 120, "x2": 780, "y2": 150, "quadrant_id": "NE"},
                    {"kind": "grass", "conf": 0.8, "x1": 800, "y1": 130, "x2": 830, "y2": 160, "quadrant_id": "NE"},
                    {"kind": "grass", "conf": 0.8, "x1": 840, "y1": 140, "x2": 870, "y2": 170, "quadrant_id": "NE"},
                ],
            },
            "SW": {
                "id": "SW", "key": "part_03_southwest", "name": "Part 03 — Southwest",
                "width": 640, "height": 360, "offset_x": 0, "offset_y": 360, "status": "SUCCESS",
                "counts": {"visible_cotton_count": 3, "visible_weed_count": 0, "weed_to_crop_ratio": 0.0, "weed_to_crop_ratio_display": "0.00"},
                "risk_assessment": {"risk_category": "LOW", "action_priority": "ROUTINE_MONITORING", "color_tag": "green"},
                "spatial_summary": {"cluster_count": 0, "weeds_adjacent_to_crop": 0},
                "global_boxes": [
                    {"kind": "cotton", "conf": 0.9, "x1": 100, "y1": 400, "x2": 150, "y2": 450, "quadrant_id": "SW"},
                    {"kind": "cotton", "conf": 0.9, "x1": 200, "y1": 400, "x2": 250, "y2": 450, "quadrant_id": "SW"},
                    {"kind": "cotton", "conf": 0.9, "x1": 300, "y1": 400, "x2": 350, "y2": 450, "quadrant_id": "SW"},
                ],
            },
            "SE": {
                "id": "SE", "key": "part_04_southeast", "name": "Part 04 — Southeast",
                "width": 640, "height": 360, "offset_x": 640, "offset_y": 360, "status": "SUCCESS",
                "counts": {"visible_cotton_count": 1, "visible_weed_count": 2, "weed_to_crop_ratio": 2.0, "weed_to_crop_ratio_display": "2.00"},
                "risk_assessment": {"risk_category": "MODERATE", "action_priority": "MEDIUM_PRIORITY", "color_tag": "amber"},
                "spatial_summary": {"cluster_count": 0, "weeds_adjacent_to_crop": 1},
                "global_boxes": [
                    {"kind": "cotton", "conf": 0.9, "x1": 700, "y1": 400, "x2": 750, "y2": 450, "quadrant_id": "SE"},
                    {"kind": "grass", "conf": 0.8, "x1": 800, "y1": 400, "x2": 850, "y2": 450, "quadrant_id": "SE"},
                    {"kind": "grass", "conf": 0.8, "x1": 900, "y1": 400, "x2": 950, "y2": 450, "quadrant_id": "SE"},
                ],
            },
        }

        agg = self.segmenter.aggregate_field(q_results, 1280, 720, 640, 360)
        s = agg["summary"]
        self.assertEqual(s["total_visible_cotton"], 7)
        self.assertEqual(s["total_visible_weeds"], 7)
        self.assertEqual(s["highest_weed_quadrant"], "NE")
        self.assertEqual(s["highest_weed_count"], 4)
        self.assertEqual(s["overall_risk_category"], "HIGH")

    def test_09_zero_cotton_handling(self):
        """Verify weed-to-crop ratio handles zero cotton detections gracefully."""
        qinfo = {
            "id": "NW", "key": "part_01_northwest", "name": "Part 01 — Northwest",
            "display_name": "Part 01 — Northwest", "width": 640, "height": 360, "offset_x": 0, "offset_y": 0,
            "bounds": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}
        }
        # 3 weeds, 0 cotton
        local_boxes = [
            {"kind": "grass", "conf": 0.8, "x1": 50, "y1": 50, "x2": 90, "y2": 90},
            {"kind": "grass", "conf": 0.8, "x1": 150, "y1": 150, "x2": 190, "y2": 190},
            {"kind": "grass", "conf": 0.8, "x1": 250, "y1": 250, "x2": 290, "y2": 290},
        ]
        res = self.segmenter.analyze_quadrant(qinfo, local_boxes)
        cnt = res["counts"]
        self.assertEqual(cnt["visible_cotton_count"], 0)
        self.assertEqual(cnt["visible_weed_count"], 3)
        self.assertIn("no cotton observed", cnt["weed_to_crop_ratio_display"])
        self.assertTrue(any("No cotton plants observed" in w for w in res["warnings"]))

    def test_10_partial_quadrant_analysis_failures(self):
        """Verify system handles a detector exception on one quadrant gracefully."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        def failing_detector(crop, qinfo):
            if qinfo["id"] == "NE":
                raise RuntimeError("Roboflow connection timeout on NE quadrant")
            return [{"kind": "cotton", "conf": 0.85, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        res = self.segmenter.process_plot(img, detector_fn=failing_detector)
        self.assertTrue(res["ok"])
        quads = res["quadrants"]

        # NE failed
        self.assertEqual(quads["NE"]["status"], "ERROR")
        self.assertTrue(any("timeout" in w for w in quads["NE"]["warnings"]))

        # NW, SW, SE succeeded
        self.assertEqual(quads["NW"]["status"], "SUCCESS")
        self.assertEqual(quads["SW"]["status"], "SUCCESS")
        self.assertEqual(quads["SE"]["status"], "SUCCESS")

    def test_11_risk_assessment_with_missing_information(self):
        """Verify risk assessment correctly signals INSUFFICIENT_DATA when optical evidence is poor."""
        qinfo = {
            "id": "SW", "key": "part_03_southwest", "name": "Part 03 — Southwest",
            "display_name": "Part 03 — Southwest", "width": 640, "height": 360, "offset_x": 0, "offset_y": 360,
            "bounds": {"x1": 0, "y1": 360, "x2": 640, "y2": 720}
        }
        # Low confidence (<35%) and unknown stage
        boxes = [{"kind": "grass", "conf": 0.22, "x1": 50, "y1": 50, "x2": 90, "y2": 90}]
        res = self.segmenter.analyze_quadrant(qinfo, boxes, field_context={"growth_stage": "UNKNOWN"})
        risk = res["risk_assessment"]
        self.assertEqual(risk["risk_category"], "INSUFFICIENT_DATA")
        self.assertEqual(risk["color_tag"], "gray")


class TestPlotApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_12_api_plot_analyze_and_validation(self):
        """Verify POST /api/plot/analyze, GET analysis, and CSV/JSON export."""
        # 1. Missing image returns 400
        res = self.client.post("/api/plot/analyze", json={})
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

        # 2. Analyze with synthetic test_boxes
        test_payload = {
            "test_boxes": [
                {"kind": "cotton", "conf": 0.9, "x1": 100, "y1": 100, "x2": 180, "y2": 180, "track_id": 1, "class_name": "cotton"},
                {"kind": "grass", "conf": 0.85, "x1": 700, "y1": 120, "x2": 760, "y2": 180, "track_id": 10, "class_name": "grass"},
                {"kind": "grass", "conf": 0.80, "x1": 750, "y1": 140, "x2": 810, "y2": 200, "track_id": 11, "class_name": "grass"},
            ],
            "field_context": {
                "field_id": "Plot-UnitTest-101",
                "growth_stage": "VEGETATIVE_SQUARING",
                "surveyed_area_sqm": 15.0,
            }
        }
        # Provide blank 1280x720 base64 image
        import base64
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, enc = cv2.imencode('.jpg', blank)
        test_payload["image_base64"] = base64.b64encode(enc).decode('utf-8')

        post_res = self.client.post("/api/plot/analyze", json=test_payload)
        self.assertEqual(post_res.status_code, 200)
        data = post_res.get_json()
        self.assertTrue(data["ok"])
        aid = data["analysis_id"]
        self.assertIn("quadrants", data)
        self.assertIn("NW", data["quadrants"])
        self.assertIn("NE", data["quadrants"])
        self.assertIn("SW", data["quadrants"])
        self.assertIn("SE", data["quadrants"])

        # 3. GET /api/plot/analysis/<analysis_id>
        get_res = self.client.get(f"/api/plot/analysis/{aid}")
        self.assertEqual(get_res.status_code, 200)
        self.assertTrue(get_res.get_json()["ok"])

        # 4. GET /api/plot/latest
        latest_res = self.client.get("/api/plot/latest")
        self.assertEqual(latest_res.status_code, 200)
        self.assertTrue(latest_res.get_json()["available"])

        # 5. Export CSV
        csv_res = self.client.get(f"/api/plot/export/{aid}?format=csv")
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("text/csv", csv_res.headers.get("Content-Type", ""))
        csv_text = csv_res.data.decode("utf-8")
        self.assertIn("NW", csv_text)
        self.assertIn("NE", csv_text)
        self.assertIn("SW", csv_text)
        self.assertIn("SE", csv_text)
        self.assertIn("WHOLE_FIELD", csv_text)

        # 6. Export JSON
        json_res = self.client.get(f"/api/plot/export/{aid}?format=json")
        self.assertEqual(json_res.status_code, 200)
        self.assertIn("application/json", json_res.headers.get("Content-Type", ""))
        export_data = json.loads(json_res.data.decode("utf-8"))
        self.assertEqual(export_data["analysis_id"], aid)

    def test_13_api_plot_demo_endpoint(self):
        """Verify POST /api/plot/demo generates demo plot with all quadrants."""
        res = self.client.post("/api/plot/demo")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        fa = data["field_assessment"]["summary"]
        self.assertGreater(fa["total_visible_cotton"], 0)
        self.assertGreater(fa["total_visible_weeds"], 0)
        self.assertEqual(fa["highest_weed_quadrant"], "NE")

    def test_14_preservation_of_existing_routes_and_dashboard(self):
        """Verify existing routes and dashboard components remain fully functional."""
        # Root dashboard HTML
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Existing components preserved
        self.assertIn("Live camera feed", html)
        self.assertIn("Detection controls", html)
        self.assertIn("Detection history", html)
        self.assertIn("AI Farmer Advisor & Crop Risk Assessment", html)

        # New Plot Analysis components present
        self.assertIn("Four-Part Precision Agricultural Plot Analysis", html)
        self.assertIn("Part 01 — Northwest", html)
        self.assertIn("Part 02 — Northeast", html)
        self.assertIn("Part 03 — Southwest", html)
        self.assertIn("Part 04 — Southeast", html)
        self.assertIn("WHOLE-FIELD ASSESSMENT & QUADRANT COMPARISON", html)
        self.assertIn("chkPlotGrid", html)
        self.assertIn("chkPlotCotton", html)
        self.assertIn("chkPlotWeeds", html)

        # Existing endpoints
        status_res = self.client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertTrue(status_res.get_json().get("ok"))

        rec_status = self.client.get("/api/recommendations/status")
        self.assertEqual(rec_status.status_code, 200)
        self.assertTrue(rec_status.get_json().get("ok"))


if __name__ == "__main__":
    unittest.main()
