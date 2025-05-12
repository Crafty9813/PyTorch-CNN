import cv2
import numpy as np

img = cv2.resize(cv2.imread('assets/desk.jpeg', 0), (0, 0), fx = 2, fy = 2) #0 is for grayscale
template = cv2.imread('assets/headphone.png', 0)

h, w = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method) #convolution (template image is "kernel"/slide)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc #top left
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img2, top_left, bottom_right, 255, 5)

    cv2.imshow('frame', img2)

    cv2.waitKey(0) #hmm waitKey for a specific key dont work
    cv2.destroyAllWindows()