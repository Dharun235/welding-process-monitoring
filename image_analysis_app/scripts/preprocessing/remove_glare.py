import cv2
import numpy as np

def remove_glare(img):
    '''
    Removes glare from the input image by identifying bright regions using a top-hat morphological operation and then inpainting those regions.
    Parameters:
    - img: Input grayscale image (numpy array).
    '''
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    img_n = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    k = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(img_n, cv2.MORPH_TOPHAT, kernel)

    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    #cv2.imshow("Glare Mask", mask)

    #out = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)
    out = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)

    return out