"""Visualization helpers for droplet/spatter detections."""

import cv2

def draw_detected_blobs(image, features, label, color):
    """Draw detected blob outlines and labels on the given image."""
    for feature in features or []:
        centroid = feature.get("centroid")
        if centroid is None:
            continue

        x, y = int(centroid[0]), int(centroid[1])
        radius = int(feature.get("radius_px", max(1, round(feature.get("diameter_px", 0) / 2))))
        radius = max(1, radius)

        cv2.circle(image, (x, y), radius, color, 2)
        cv2.putText(
            image,
            label,
            (x + 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
