"""Droplet/spatter blob detection from frame-level search-space regions."""

import cv2
import numpy as np

def detect_droplets_spatters(image,
                             searchspace_box,
                             weldpool_location,
                             wiretip_location,
                             center_tolerance=30,
                             min_area=10,
                             max_white_fraction=0.01):
    """
    Detects blobs inside ROIs defined relative to the searchspace box.
    Keeps only blobs where few pixels are white inside the blob.
    Classifies them as droplet (center ROI) or spatter (left/right ROI).
    """

    if searchspace_box is None:
        return [], []

    img_h, img_w = image.shape[:2]

    # Convert searchspace box to coordinates
    pts = searchspace_box.reshape(-1, 2)
    box_x_min = int(pts[:, 0].min())
    box_x_max = int(pts[:, 0].max())
    box_y_min = int(pts[:, 1].min())
    box_y_max = int(pts[:, 1].max())
    box_center_x = (box_x_min + box_x_max) // 2

    # ================================
    # DEFINE ROIs
    # ================================
    
    # Limit center ROI bottom to weldpool y
    roi_center_y_max = weldpool_location[1] if weldpool_location is not None else box_y_max
    roi_center = (box_center_x - center_tolerance, box_center_x + center_tolerance, box_y_min, roi_center_y_max)
    roi_left = (0, box_x_min, box_y_min, box_y_max)
    roi_right = (box_x_max, img_w, box_y_min, box_y_max)

    # ================================
    # SIMPLE BLOB DETECTION
    # ================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    droplets_list = []
    spatters_list = []

    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        
        # Skip blobs outside vertical bounds
        if not (box_y_min <= cy <= box_y_max):
            continue

        # Mask and white pixel check
        total_area = np.pi * radius * radius

        # Calculate diameter in pixels
        diameter_px = 2 * int(np.sqrt(total_area / np.pi))

        # Determine type and assign
        if roi_center[0] <= cx <= roi_center[1] and roi_center[2] <= cy <= roi_center[3]:
            if total_area >= 500:
                droplets_list.append({
                    "centroid": (cx, cy),
                    "area": total_area,
                    "diameter_px": diameter_px,
                    "radius_px": radius,
                })
            else:
                spatters_list.append({
                    "centroid": (cx, cy),
                    "area": total_area,
                    "diameter_px": diameter_px,
                    "radius_px": radius,
                })
        elif cx <= roi_left[1] or cx >= roi_right[0]:
            spatters_list.append({
                "centroid": (cx, cy),
                "area": total_area,
                "diameter_px": diameter_px,
                "radius_px": radius,
            })
        else:
            continue

    return droplets_list, spatters_list
