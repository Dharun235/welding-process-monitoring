"""Laserlit wire-frame preprocessing routines."""

import cv2

from scripts.preprocessing.remove_glare import remove_glare

def preprocess_laserlit(img, canny1, canny2, morph):
    '''
    Preprocessesing method for laserlit images.
    Parameters:
    - img: Input color image (numpy array).
    - canny1: First threshold for the hysteresis procedure in Canny edge detection.
    - canny2: Second threshold for the hysteresis procedure in Canny edge detection.
    - morph: Whether to apply morphological closing to the edges after Canny edge detection.

    Returns:
    - preprocessed_img: Preprocessed grayscale image (numpy array).
    - morph_edges: Morphologically closed Canny edges of the preprocessed image (numpy array).
    '''

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    glare_removed = remove_glare(normalized)
    #cv2.imshow("Before blur", glare_removed)
    preprocessed_img = cv2.GaussianBlur(glare_removed, (5, 5), 1)
    #preprocessed_img = cv2.GaussianBlur(preprocessed_img, (5, 5), 1)
    #preprocessed_img = cv2.GaussianBlur(preprocessed_img, (5, 5), 1)
    edges = cv2.Canny(preprocessed_img, canny1, canny2, apertureSize=3)

    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    #cv2.imshow("Original Image", img)
    #cv2.imshow("Preprocessed Image", preprocessed_img)
    #cv2.imshow("Canny Edges", edges)
    #cv2.waitKey(0)

    return preprocessed_img, edges
