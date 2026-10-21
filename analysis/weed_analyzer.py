"""Weed Analysis Engine for AgriSight AI.

Distinguishes between:
A. Current visible weed count
B. Unique tracked weed count (ByteTrack)
C. Estimated weed density (requires verified physical scale/area)
D. Estimated infestation severity

Ensures rigorous data quality reporting and never confuses image-plane metrics
with physical agronomical field measurements without genuine calibration.
"""
from typing import Dict, List, Optional, Any, Tuple


class WeedAnalyzer:
    """Analyzes weed and crop detections to assess counts, clustering tendencies,

    ratios, and data quality indicators.
    """

    # Infestation thresholds based on visible plant proportion & count in view
    INFESTATION_THRESHOLDS = {
        "LOW": {"max_visible_count": 2, "max_weed_ratio": 0.25},
        "MODERATE": {"max_visible_count": 6, "max_weed_ratio": 0.55},
        "HIGH": {"max_visible_count": 12, "max_weed_ratio": 0.80},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def analyze(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        field_context: Optional[Dict[str, Any]] = None,
        cumulative_tracked_ids: Optional[Dict[str, set]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive weed analysis on current frame/session detections.

        Args:
            boxes: List of bounding box dictionaries, each containing:
                   `kind` ('cotton' or 'grass'), `conf` (float), `track_id` (optional int),
                   `x1`, `y1`, `x2`, `y2`, `class_name` (str).
            image_width: Width of image in pixels.
            image_height: Height of image in pixels.
            field_context: Optional dictionary with agricultural metadata like:
                           `field_area_sqm`, `camera_height_m`, `ground_sampling_distance_cm`,
                           `growth_stage`, `field_id`.
            cumulative_tracked_ids: Optional dict mapping 'cotton' and 'grass' to sets of observed track IDs.

        Returns:
            Structured analysis dictionary.
        """
        field_ctx = field_context or {}
        missing_info: List[str] = []

        # 1. Segregate detections by verified kind
        weed_boxes = [b for b in boxes if b.get("kind") == "grass"]
        cotton_boxes = [b for b in boxes if b.get("kind") == "cotton"]

        visible_weed_count = len(weed_boxes)
        visible_cotton_count = len(cotton_boxes)
        total_visible_plants = visible_weed_count + visible_cotton_count

        # 2. Tracking handling (ByteTrack IDs vs frame-only fallback)
        tracked_weed_ids = set()
        has_tracker_data = False

        for b in weed_boxes:
            tid = b.get("track_id")
            if tid is not None:
                tracked_weed_ids.add(tid)
                has_tracker_data = True

        # If cumulative session tracking was passed in, merge it
        if cumulative_tracked_ids and "grass" in cumulative_tracked_ids:
            tracked_weed_ids = tracked_weed_ids.union(cumulative_tracked_ids["grass"])
            has_tracker_data = True

        tracked_weed_count = len(tracked_weed_ids) if has_tracker_data else None
        tracking_status = (
            "PERSISTENT_BYTETRACK" if has_tracker_data else "FRAME_BASED_ONLY"
        )

        if not has_tracker_data:
            missing_info.append("Temporal ByteTrack IDs (counts are frame-based only)")

        # 3. Detection proportions and ratios
        if total_visible_plants > 0:
            weed_to_total_ratio = round(visible_weed_count / total_visible_plants, 3)
            weed_to_crop_ratio = (
                round(visible_weed_count / max(1, visible_cotton_count), 2)
                if visible_cotton_count > 0
                else float(visible_weed_count)
            )
        else:
            weed_to_total_ratio = 0.0
            weed_to_crop_ratio = 0.0

        # 4. Physical Area & Density Estimation (With Explicit Guardrails)
        # Never invent physical density without explicit ground sampling distance or surveyed area
        physical_density_sqm = None
        density_status = "UNKNOWN_NO_CALIBRATION"
        surveyed_area_sqm = field_ctx.get("surveyed_area_sqm") or field_ctx.get(
            "field_area_sqm"
        )
        gsd_cm = field_ctx.get("ground_sampling_distance_cm")
        cam_height_m = field_ctx.get("camera_height_m")

        if surveyed_area_sqm and float(surveyed_area_sqm) > 0:
            effective_count = (
                tracked_weed_count
                if tracked_weed_count is not None
                else visible_weed_count
            )
            physical_density_sqm = round(effective_count / float(surveyed_area_sqm), 2)
            density_status = "CALCULATED_FROM_SURVEYED_AREA"
        elif gsd_cm and image_width > 0 and image_height > 0:
            # Approximate frame area from GSD: (width * gsd_cm / 100) * (height * gsd_cm / 100)
            frame_area_sqm = (image_width * float(gsd_cm) / 100.0) * (
                image_height * float(gsd_cm) / 100.0
            )
            if frame_area_sqm > 0:
                physical_density_sqm = round(visible_weed_count / frame_area_sqm, 2)
                density_status = "CALCULATED_FROM_GSD"
        else:
            missing_info.append(
                "Camera calibration / Ground Sampling Distance (GSD) or surveyed field area"
            )

        # 5. Dispersion & Clustering Estimation
        clustering_info = self._calculate_dispersion(
            weed_boxes, image_width, image_height
        )

        # 6. Infestation Severity Level Assessment
        infestation_level = self._estimate_infestation_level(
            visible_weed_count=visible_weed_count,
            weed_to_total_ratio=weed_to_total_ratio,
            is_concentrated=clustering_info["is_concentrated"],
        )

        # 7. Data Quality & Confidence Indicators
        confidences = [b.get("conf", 0.0) for b in weed_boxes]
        avg_confidence = (
            round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        )
        min_confidence = round(min(confidences), 3) if confidences else 0.0

        if avg_confidence > 0 and avg_confidence < 0.35:
            missing_info.append("Higher confidence optical verification")

        if not field_ctx.get("growth_stage"):
            missing_info.append("Cotton crop growth stage (e.g. seedling, vegetative, flowering)")

        return {
            "counts": {
                "visible_weed_count": visible_weed_count,
                "visible_cotton_count": visible_cotton_count,
                "total_visible_plants": total_visible_plants,
                "tracked_weed_count": tracked_weed_count,
                "tracking_status": tracking_status,
                "weed_to_total_ratio": weed_to_total_ratio,
                "weed_to_crop_ratio": weed_to_crop_ratio,
            },
            "density": {
                "density_status": density_status,
                "density_per_sqm": physical_density_sqm,
                "surveyed_area_sqm": surveyed_area_sqm,
                "density_caveat": (
                    "Physical weed density (plants/m²) requires calibrated GSD or measured survey plot. "
                    "Image-plane counts must not be interpreted as hectare-scale density without sensor calibration."
                ),
            },
            "dispersion": clustering_info,
            "infestation_level": infestation_level,
            "data_quality": {
                "avg_weed_confidence": avg_confidence,
                "min_weed_confidence": min_confidence,
                "resolution": f"{image_width}x{image_height}",
                "reliable_optical_evidence": avg_confidence >= 0.40 and visible_weed_count > 0,
            },
            "missing_information": missing_info,
        }

    def _calculate_dispersion(
        self, weed_boxes: List[Dict[str, Any]], width: int, height: int
    ) -> Dict[str, Any]:
        """Examine weed centroids to determine if weeds are scattered or concentrated."""
        if len(weed_boxes) <= 1 or width <= 0 or height <= 0:
            return {
                "distribution_type": "SINGLE_OR_NONE" if weed_boxes else "NO_WEEDS_DETECTED",
                "is_concentrated": False,
                "cluster_count": 1 if weed_boxes else 0,
                "spread_metric": 0.0,
            }

        # Calculate centroids normalized to [0, 1]
        centroids = []
        for b in weed_boxes:
            cx = ((b.get("x1", 0) + b.get("x2", 0)) / 2.0) / width
            cy = ((b.get("y1", 0) + b.get("y2", 0)) / 2.0) / height
            centroids.append((cx, cy))

        # Standard deviation of positions
        avg_x = sum(c[0] for c in centroids) / len(centroids)
        avg_y = sum(c[1] for c in centroids) / len(centroids)

        variance_x = sum((c[0] - avg_x) ** 2 for c in centroids) / len(centroids)
        variance_y = sum((c[1] - avg_y) ** 2 for c in centroids) / len(centroids)
        std_dev = (variance_x + variance_y) ** 0.5

        # Lower std_dev relative to total count indicates tight spatial grouping
        # If standard deviation is low (<0.18 on normalized coordinates) and count >= 3, it's concentrated
        is_concentrated = (len(weed_boxes) >= 3 and std_dev < 0.22) or (len(weed_boxes) >= 6 and std_dev < 0.28)

        if is_concentrated:
            distribution_type = "CONCENTRATED_PATCH"
        elif std_dev >= 0.30:
            distribution_type = "SCATTERED_ACROSS_FIELD"
        else:
            distribution_type = "MODERATELY_DISTRIBUTED"

        return {
            "distribution_type": distribution_type,
            "is_concentrated": is_concentrated,
            "spread_metric": round(float(std_dev), 3),
            "centroid_center": (round(avg_x * width, 1), round(avg_y * height, 1)),
        }

    def _estimate_infestation_level(
        self, visible_weed_count: int, weed_to_total_ratio: float, is_concentrated: bool
    ) -> str:
        """Estimate infestation level based on visible abundance and crop proportion."""
        if visible_weed_count == 0:
            return "NO_WEEDS"

        t = self.INFESTATION_THRESHOLDS
        if (
            visible_weed_count <= t["LOW"]["max_visible_count"]
            and weed_to_total_ratio <= t["LOW"]["max_weed_ratio"]
        ):
            return "LOW"
        elif (
            visible_weed_count <= t["MODERATE"]["max_visible_count"]
            and weed_to_total_ratio <= t["MODERATE"]["max_weed_ratio"]
            and not is_concentrated
        ):
            return "MODERATE"
        elif visible_weed_count <= t["HIGH"]["max_visible_count"]:
            return "HIGH"
        else:
            return "SEVERE"
