"""Spatial Analysis Engine for AgriSight AI.

Performs image-plane spatial analysis:
- Cluster identification on image coordinates
- Relative quadrant/grid distribution
- Inter-crop proximity modeling (distance between weed and cotton bboxes)
- Highlight areas of interest for field scouting

Explicitly distinguishes image-plane pixels from geographic GPS coordinates.
"""
from typing import Dict, List, Optional, Any, Tuple
import math


class SpatialAnalyzer:
    """Analyzes spatial distribution of bounding boxes on the 2D image plane."""

    GRID_ROWS = 3
    GRID_COLS = 3
    QUADRANT_LABELS = [
        ["Top-Left", "Top-Center", "Top-Right"],
        ["Mid-Left", "Center", "Mid-Right"],
        ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
    ]

    def __init__(self, proximity_threshold_px: float = 75.0):
        self.proximity_threshold_px = proximity_threshold_px

    def analyze(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        gps_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform spatial analysis on detected bounding boxes in image space.

        Args:
            boxes: List of bounding box dictionaries.
            image_width: Frame width in pixels.
            image_height: Frame height in pixels.
            gps_metadata: Optional genuine geographic coordinates (latitude, longitude, altitude).

        Returns:
            Spatial analysis results.
        """
        if image_width <= 0 or image_height <= 0:
            image_width = max(1, image_width)
            image_height = max(1, image_height)

        weed_boxes = [b for b in boxes if b.get("kind") == "grass"]
        cotton_boxes = [b for b in boxes if b.get("kind") == "cotton"]

        # 1. Image Grid / Region Partitioning
        grid_counts = self._compute_grid_distribution(
            weed_boxes, image_width, image_height
        )

        # 2. Identify Concentrated Clusters in Image Plane
        clusters = self._find_clusters(weed_boxes, distance_threshold=image_width * 0.15)

        # 3. Inter-Crop Proximity (Distance between weed centroids and cotton centroids)
        proximity_insights = self._analyze_inter_crop_proximity(
            weed_boxes, cotton_boxes
        )

        # 4. Image Regions Requiring Inspection (High-density zones)
        priority_inspection_regions = self._identify_inspection_zones(
            grid_counts, clusters, image_width, image_height
        )

        # 5. Geographic Hotspot Mapping Caveat
        geo_context = {
            "has_genuine_gps": bool(gps_metadata and "latitude" in gps_metadata and "longitude" in gps_metadata),
            "gps_coordinates": gps_metadata if gps_metadata else None,
            "geo_notice": (
                "Coordinates reported are image-plane pixel bounds. "
                "Geographic field coordinates require camera-synchronized GPS telemetry."
            ),
        }

        # 6. Overall distribution classification
        if not weed_boxes:
            spatial_distribution = "CLEAR_NO_WEEDS"
        elif len(clusters) > 0 and any(c["weed_count"] >= 3 for c in clusters):
            spatial_distribution = "CONCENTRATED_CLUSTERS"
        elif len(weed_boxes) <= 2:
            spatial_distribution = "ISOLATED_OCCURRENCES"
        else:
            spatial_distribution = "SCATTERED_DISTRIBUTION"

        return {
            "spatial_distribution": spatial_distribution,
            "grid_distribution": grid_counts,
            "clusters": clusters,
            "inter_crop_proximity": proximity_insights,
            "inspection_regions": priority_inspection_regions,
            "coordinate_system": "IMAGE_PIXEL_PLANE",
            "frame_dimensions": {"width": image_width, "height": image_height},
            "geographic_data": geo_context,
        }

    def _compute_grid_distribution(
        self, weed_boxes: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Divide the image into a 3x3 grid and count weed occurrences."""
        grid = []
        cell_w = width / self.GRID_COLS
        cell_h = height / self.GRID_ROWS

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                name = self.QUADRANT_LABELS[r][c]
                rx1, ry1 = c * cell_w, r * cell_h
                rx2, ry2 = rx1 + cell_w, ry1 + cell_h

                matching = []
                for b in weed_boxes:
                    cx = (b.get("x1", 0) + b.get("x2", 0)) / 2.0
                    cy = (b.get("y1", 0) + b.get("y2", 0)) / 2.0
                    if rx1 <= cx < rx2 and ry1 <= cy < ry2:
                        matching.append(b.get("track_id"))

                grid.append(
                    {
                        "quadrant": name,
                        "row": r,
                        "col": c,
                        "bounds_px": {
                            "x1": round(rx1, 1),
                            "y1": round(ry1, 1),
                            "x2": round(rx2, 1),
                            "y2": round(ry2, 1),
                        },
                        "weed_count": len(matching),
                        "weed_track_ids": [t for t in matching if t is not None],
                    }
                )
        return grid

    def _find_clusters(
        self, weed_boxes: List[Dict[str, Any]], distance_threshold: float
    ) -> List[Dict[str, Any]]:
        """Group weed boxes within distance_threshold of each other into clusters."""
        if not weed_boxes:
            return []

        clusters = []
        visited = set()

        def centroid(b):
            return ((b.get("x1", 0) + b.get("x2", 0)) / 2.0, (b.get("y1", 0) + b.get("y2", 0)) / 2.0)

        for i, b1 in enumerate(weed_boxes):
            if i in visited:
                continue
            current_cluster = [b1]
            visited.add(i)
            c1 = centroid(b1)

            for j, b2 in enumerate(weed_boxes):
                if j in visited:
                    continue
                c2 = centroid(b2)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist <= distance_threshold:
                    current_cluster.append(b2)
                    visited.add(j)

            if len(current_cluster) >= 2:
                # Compute bounding box and center of the cluster
                min_x = min(b.get("x1", 0) for b in current_cluster)
                min_y = min(b.get("y1", 0) for b in current_cluster)
                max_x = max(b.get("x2", 0) for b in current_cluster)
                max_y = max(b.get("y2", 0) for b in current_cluster)
                track_ids = [b.get("track_id") for b in current_cluster if b.get("track_id") is not None]

                clusters.append(
                    {
                        "cluster_id": len(clusters) + 1,
                        "weed_count": len(current_cluster),
                        "center_px": (round((min_x + max_x) / 2.0, 1), round((min_y + max_y) / 2.0, 1)),
                        "bounding_box_px": {
                            "x1": round(min_x, 1),
                            "y1": round(min_y, 1),
                            "x2": round(max_x, 1),
                            "y2": round(max_y, 1),
                        },
                        "member_track_ids": track_ids,
                    }
                )

        return clusters

    def _analyze_inter_crop_proximity(
        self, weed_boxes: List[Dict[str, Any]], cotton_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate distances between weed boxes and nearby cotton plants."""
        if not weed_boxes or not cotton_boxes:
            return {
                "can_evaluate_crop_proximity": bool(weed_boxes and cotton_boxes),
                "weeds_adjacent_to_crop": 0,
                "min_weed_crop_distance_px": None,
                "crop_weed_overlap_detected": False,
            }

        def centroid(b):
            return ((b.get("x1", 0) + b.get("x2", 0)) / 2.0, (b.get("y1", 0) + b.get("y2", 0)) / 2.0)

        adjacent_count = 0
        min_dist = float("inf")
        overlap_detected = False

        for wb in weed_boxes:
            wc = centroid(wb)
            for cb in cotton_boxes:
                cc = centroid(cb)
                dist = math.hypot(wc[0] - cc[0], wc[1] - cc[1])
                if dist < min_dist:
                    min_dist = dist

                # Check if boxes intersect / touch in the image
                inter_x = max(0, min(wb.get("x2", 0), cb.get("x2", 0)) - max(wb.get("x1", 0), cb.get("x1", 0)))
                inter_y = max(0, min(wb.get("y2", 0), cb.get("y2", 0)) - max(wb.get("y1", 0), cb.get("y1", 0)))
                if inter_x * inter_y > 0:
                    overlap_detected = True

                if dist <= self.proximity_threshold_px:
                    adjacent_count += 1
                    break

        return {
            "can_evaluate_crop_proximity": True,
            "weeds_adjacent_to_crop": adjacent_count,
            "min_weed_crop_distance_px": round(min_dist, 1) if min_dist != float("inf") else None,
            "crop_weed_overlap_detected": overlap_detected,
            "proximity_threshold_px": self.proximity_threshold_px,
        }

    def _identify_inspection_zones(
        self,
        grid: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Select top quadrants/clusters requiring physical ground scouting."""
        zones = []
        for cell in sorted(grid, key=lambda c: c["weed_count"], reverse=True):
            if cell["weed_count"] >= 2:
                zones.append(
                    {
                        "priority": "HIGH" if cell["weed_count"] >= 4 else "MEDIUM",
                        "description": f"Concentration in {cell['quadrant']} quadrant ({cell['weed_count']} weeds)",
                        "quadrant": cell["quadrant"],
                        "bounds_px": cell["bounds_px"],
                        "weed_count": cell["weed_count"],
                        "track_ids": cell["weed_track_ids"],
                    }
                )
        return zones
