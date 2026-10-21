"""End-to-end workflow verification with an agricultural plot image."""
import unittest
import sys
from pathlib import Path
import json
import numpy as np
import cv2

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.plot_segmenter import PlotSegmenter
from app import SNAPSHOTS, app


class TestRealAgriculturalPlotWorkflow(unittest.TestCase):
    def test_end_to_end_plot_workflow(self):
        # 1. Create a 1280x720 high-fidelity agricultural plot image
        h, w = 720, 1280
        field_img = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep soil tone BGR: (35, 52, 45)
        field_img[:] = (35, 52, 45)

        # Draw cotton furrows / crop beds
        for y in range(40, h, 80):
            cv2.line(field_img, (0, y), (w, y), (45, 65, 55), 20)

        # Add realistic cotton plants across all quadrants
        # Part 01 - NW: 3 cottons, 1 weed
        cv2.circle(field_img, (150, 120), 30, (40, 160, 60), -1) # cotton foliage
        cv2.circle(field_img, (150, 120), 8, (245, 245, 245), -1) # cotton boll
        cv2.circle(field_img, (350, 140), 28, (40, 160, 60), -1)
        cv2.circle(field_img, (350, 140), 7, (245, 245, 245), -1)
        cv2.circle(field_img, (250, 260), 32, (40, 160, 60), -1)
        cv2.circle(field_img, (250, 260), 9, (245, 245, 245), -1)
        cv2.circle(field_img, (520, 90), 22, (25, 200, 180), -1) # weed patch

        # Part 02 - NE: 2 cottons, heavy weed infestation (5 weeds)
        cv2.circle(field_img, (780, 130), 28, (40, 160, 60), -1)
        cv2.circle(field_img, (780, 130), 8, (245, 245, 245), -1)
        cv2.circle(field_img, (1020, 140), 30, (40, 160, 60), -1)
        cv2.circle(field_img, (1020, 140), 8, (245, 245, 245), -1)
        # Weeds in NE
        for wx, wy in [(800, 140), (840, 200), (900, 150), (940, 220), (980, 120)]:
            cv2.circle(field_img, (wx, wy), 20, (20, 195, 175), -1)

        # Part 03 - SW: 3 cottons, 2 weeds
        cv2.circle(field_img, (140, 480), 30, (40, 160, 60), -1)
        cv2.circle(field_img, (320, 490), 28, (40, 160, 60), -1)
        cv2.circle(field_img, (220, 620), 32, (40, 160, 60), -1)
        cv2.circle(field_img, (190, 500), 20, (20, 195, 175), -1)
        cv2.circle(field_img, (400, 600), 22, (20, 195, 175), -1)

        # Part 04 - SE: 4 clean cottons, 0 weeds
        cv2.circle(field_img, (760, 480), 30, (40, 160, 60), -1)
        cv2.circle(field_img, (920, 490), 28, (40, 160, 60), -1)
        cv2.circle(field_img, (1080, 485), 29, (40, 160, 60), -1)
        cv2.circle(field_img, (840, 620), 31, (40, 160, 60), -1)

        # Save as test plot image
        test_img_path = SNAPSHOTS / "sample_agricultural_plot.jpg"
        cv2.imwrite(str(test_img_path), field_img)
        self.assertTrue(test_img_path.exists())

        # 2. Run through PlotSegmenter with mock/verified detector
        segmenter = PlotSegmenter()

        known_boxes = [
            # NW
            {"kind": "cotton", "conf": 0.92, "x1": 120, "y1": 90, "x2": 180, "y2": 150, "track_id": 1, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.89, "x1": 320, "y1": 110, "x2": 380, "y2": 170, "track_id": 2, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.94, "x1": 220, "y1": 230, "x2": 280, "y2": 290, "track_id": 3, "class_name": "cotton"},
            {"kind": "grass", "conf": 0.81, "x1": 500, "y1": 70, "x2": 540, "y2": 110, "track_id": 10, "class_name": "grass"},
            # NE
            {"kind": "cotton", "conf": 0.88, "x1": 750, "y1": 100, "x2": 810, "y2": 160, "track_id": 4, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.90, "x1": 990, "y1": 110, "x2": 1050, "y2": 170, "track_id": 5, "class_name": "cotton"},
            {"kind": "grass", "conf": 0.84, "x1": 780, "y1": 120, "x2": 820, "y2": 160, "track_id": 11, "class_name": "grass"},
            {"kind": "grass", "conf": 0.79, "x1": 820, "y1": 180, "x2": 860, "y2": 220, "track_id": 12, "class_name": "grass"},
            {"kind": "grass", "conf": 0.83, "x1": 880, "y1": 130, "x2": 920, "y2": 170, "track_id": 13, "class_name": "grass"},
            {"kind": "grass", "conf": 0.77, "x1": 920, "y1": 200, "x2": 960, "y2": 240, "track_id": 14, "class_name": "grass"},
            {"kind": "grass", "conf": 0.82, "x1": 960, "y1": 100, "x2": 1000, "y2": 140, "track_id": 15, "class_name": "grass"},
            # SW
            {"kind": "cotton", "conf": 0.91, "x1": 110, "y1": 450, "x2": 170, "y2": 510, "track_id": 6, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.87, "x1": 290, "y1": 460, "x2": 350, "y2": 520, "track_id": 7, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.93, "x1": 190, "y1": 590, "x2": 250, "y2": 650, "track_id": 8, "class_name": "cotton"},
            {"kind": "grass", "conf": 0.80, "x1": 170, "y1": 480, "x2": 210, "y2": 520, "track_id": 16, "class_name": "grass"},
            {"kind": "grass", "conf": 0.76, "x1": 380, "y1": 580, "x2": 420, "y2": 620, "track_id": 17, "class_name": "grass"},
            # SE
            {"kind": "cotton", "conf": 0.95, "x1": 730, "y1": 450, "x2": 790, "y2": 510, "track_id": 9, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.92, "x1": 890, "y1": 460, "x2": 950, "y2": 520, "track_id": 10, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.89, "x1": 1050, "y1": 455, "x2": 1110, "y2": 515, "track_id": 11, "class_name": "cotton"},
            {"kind": "cotton", "conf": 0.94, "x1": 810, "y1": 590, "x2": 870, "y2": 650, "track_id": 12, "class_name": "cotton"},
        ]

        def detector_fn(crop, qinfo):
            ox, oy = qinfo["offset_x"], qinfo["offset_y"]
            qw, qh = qinfo["width"], qinfo["height"]
            local_b = []
            for kb in known_boxes:
                gx1, gy1 = kb["x1"], kb["y1"]
                gx2, gy2 = kb["x2"], kb["y2"]
                if max(ox, gx1) < min(ox + qw, gx2) and max(oy, gy1) < min(oy + qh, gy2):
                    local_b.append({
                        "kind": kb["kind"],
                        "conf": kb["conf"],
                        "x1": max(0, gx1 - ox),
                        "y1": max(0, gy1 - oy),
                        "x2": min(qw, gx2 - ox),
                        "y2": min(qh, gy2 - oy),
                        "track_id": kb["track_id"],
                        "class_name": kb["class_name"],
                    })
            return local_b

        ctx = {
            "field_id": "Plot-Sector-C4",
            "growth_stage": "VEGETATIVE_SQUARING",
            "surveyed_area_sqm": 20.0,
        }
        res = segmenter.process_plot(field_img, detector_fn=detector_fn, field_context=ctx)

        self.assertTrue(res["ok"])
        self.assertEqual(res["original_dimensions"]["width"], 1280)
        self.assertEqual(res["original_dimensions"]["height"], 720)

        # Verify quadrants
        q = res["quadrants"]
        self.assertEqual(q["NW"]["counts"]["visible_cotton_count"], 3)
        self.assertEqual(q["NW"]["counts"]["visible_weed_count"], 1)

        self.assertEqual(q["NE"]["counts"]["visible_cotton_count"], 2)
        self.assertEqual(q["NE"]["counts"]["visible_weed_count"], 5)
        self.assertEqual(q["NE"]["risk_assessment"]["risk_category"], "HIGH")

        self.assertEqual(q["SW"]["counts"]["visible_cotton_count"], 3)
        self.assertEqual(q["SW"]["counts"]["visible_weed_count"], 2)

        self.assertEqual(q["SE"]["counts"]["visible_cotton_count"], 4)
        self.assertEqual(q["SE"]["counts"]["visible_weed_count"], 0)
        self.assertEqual(q["SE"]["risk_assessment"]["risk_category"], "LOW")

        # Verify Field Assessment
        fa = res["field_assessment"]["summary"]
        self.assertEqual(fa["total_visible_cotton"], 12)
        self.assertEqual(fa["total_visible_weeds"], 8)
        self.assertEqual(fa["highest_weed_quadrant"], "NE")
        self.assertEqual(fa["highest_weed_count"], 5)
        self.assertEqual(fa["overall_risk_category"], "HIGH")

        # Verify LangGraph AI Farmer Advisor synthesis
        adv = res["field_assessment"]["advisor_recommendation"]
        self.assertIsNotNone(adv)
        self.assertIn("HIGH_PRIORITY", adv["action_priority"])
        self.assertTrue(len(adv["management_options"]) > 0)
        self.assertTrue(len(adv["sources"]) > 0)

        # Render and verify annotated plot
        annotated = segmenter.render_annotated_plot(field_img, res)
        self.assertEqual(annotated.shape, field_img.shape)
        annotated_path = SNAPSHOTS / f"sample_annotated_plot_{res['analysis_id']}.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        self.assertTrue(annotated_path.exists())


if __name__ == "__main__":
    unittest.main()
