"""Estimate the solid-to-molten wire transition in a welding frame."""

import cv2
import numpy as np

def solid_molten_transition(
    original_image,
    wiretip_location,
    wire_edges
):
    '''
    Estimate the transition point between solid and molten wire by defining the solid textured region with glare and applying
    a mask onto the glare region to find the transition point. A fallback method is used where the transition point is estimated
    as the midpoint between the bottom of the straight wire edges if the glare-based method fails.

    Parameters:
    - original_image: The original input image (numpy array).
    - wiretip_location: (x, y) wire tip location; only white pixels above this y are considered.
    - wire_edges: Tuple/list with two line segments representing the wire edges: ((x1,y1,x2,y2), (x1,y1,x2,y2))

    Returns:
    - (x, y) coordinates of the estimated transition point, or None if it cannot be determined.
    '''

    transition_point = None

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    img_n = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    k = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(img_n, cv2.MORPH_TOPHAT, kernel)

    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    #cv2.imshow("Tophat", tophat)
    #cv2.imshow("Glare Mask", mask)
    #cv2.waitKey(0)

    if wiretip_location is not None:
        wiretip_y = int(wiretip_location[1])
        wiretip_y = max(0, min(wiretip_y, mask.shape[0]))
        mask[wiretip_y:, :] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Exclude background (label 0) and pick the largest connected white region.
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == largest_label)
        if ys.size:
            max_y = ys.max()
            x_at_max_y = int(np.round(xs[ys == max_y].mean()))
            transition_point = (x_at_max_y, int(max_y))

    return transition_point
