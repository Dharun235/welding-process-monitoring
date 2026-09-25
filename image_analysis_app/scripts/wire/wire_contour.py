import cv2
import numpy as np

def identify_wire_contour(contours, min_length, min_width, max_width, max_angle, min_h_to_w, split_threshold):
    '''
    Identify wire contour(s). If the longest contour is shorter than the split
    threshold, return the two longest candidates (wire sides).
    Parameters:
    - contours: List of contours to evaluate.
    - min_length: Minimum contour length in pixels.
    - min_width: Minimum effective width of the contour in pixels.
    - max_width: Maximum effective width of the contour in pixels.
    - max_angle: Maximum allowed angle from vertical in degrees.
    - min_h_to_w: Minimum height to width ratio of the contour.
    - split_threshold: If the longest contour is shorter than this, return two contours to represent split wire sides.

    Returns:
    - List of one or two contours representing the wire, or None if no suitable contour is found.
    '''

    if contours is None:
        return None

    candidates = []
    for contour in contours:
        if cv2.arcLength(contour, closed=False) < min_length:
            continue
        (cx, cy), (rw, rh), rangle = cv2.minAreaRect(contour)
        w_eff = min(rw, rh)
        h_eff = max(rw, rh)
        if not (min_width <= w_eff <= max_width):
            continue
        if h_eff < w_eff * min_h_to_w:
            continue
        vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(np.asarray(vx).reshape(-1)[0])
        vy = float(np.asarray(vy).reshape(-1)[0])
        angle = np.degrees(np.arctan2(vy, vx))
        angle_from_vertical = min(abs(90 - angle), abs(-90 - angle))
        if angle_from_vertical > max_angle:
            continue

        length = cv2.arcLength(contour, closed=False)
        candidates.append((length, contour))

    if not candidates:
        print("Wire contours found: 0")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_len, best_contour = candidates[0]

    if best_len < split_threshold and len(candidates) >= 2:
        # Return two longest contours to represent split wire sides.
        print("Wire contours found: 2")
        return [candidates[0][1], candidates[1][1]]

    print("Wire contours found: 1")
    print(f"Best contour length: {best_len:.2f}")
    return [best_contour]
