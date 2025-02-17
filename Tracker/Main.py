import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img (title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('img/HuMen_01.jpg')
show_img('img', img)
sift = cv2.SIFT_create()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_keypoint = sift.detect(img_gray, None)
show_img('img_keypoint', cv2.drawKeypoints(img, img_keypoint, None))