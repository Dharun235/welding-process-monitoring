"""Backlit wire-frame preprocessing routines."""

import cv2


def preprocess_backlit(img, canny1, canny2, morph):
    '''
    Preprocessesing method for backlit images.
    
    Parameters:
    - img: Input color image (numpy array).
    - canny1: First threshold for the hysteresis procedure in Canny edge detection.
    - canny2: Second threshold for the hysteresis procedure in Canny edge detection.


    Returns:
    - preprocessed_img: Preprocessed grayscale image (numpy array).
    - morph_edges: Morphologically closed Canny edges of the preprocessed image (numpy array).
    '''

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    preprocessed_img = cv2.GaussianBlur(normalized, (5, 5), 1)
    
    edges = cv2.Canny(preprocessed_img, canny1, canny2, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if morph:
        morph_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return preprocessed_img, morph_edges

    return preprocessed_img, edges
