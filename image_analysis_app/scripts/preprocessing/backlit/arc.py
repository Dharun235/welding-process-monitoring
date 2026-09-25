"""Backlit arc-preprocessing routines."""

import cv2

def preprocess_backlit_arc(image):
    """
    Preprocesses the input image for arc detection in backlit welding videos.
    This function applies a series of image processing steps to enhance the visibility of the arc.

    Parameters:
    - image: The input image (BGR format) to be preprocessed.

    Returns:
    - preprocessed_img: The preprocessed image ready for arc detection.
    """
    preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_img = cv2.GaussianBlur(preprocessed_img, (5, 5), 1)
    
    return preprocessed_img