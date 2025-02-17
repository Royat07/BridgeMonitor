import cv2
import numpy as np
import os
import argparse

test_mode = False

def show_img (title, img):
    global test_mode
    if test_mode is True:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_res (title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-d", "--directory", help = "Images directory.", type = str)
args_parser.add_argument("-t", "--test", action = "store_true", help = "Test mode.")
args = args_parser.parse_args()
if args.directory is None:
    image_directory = input("Enter the images directory: ")
else:
    image_directory = args.directory
test_mode = args.test
if test_mode is True:
    print("Test mode is on.")

image_name = os.listdir(image_directory)
images = []
for index, item in enumerate(image_name):
    try:
        if item.endswith(".jpg") or item.endswith(".png") or item.endswith(".jpeg") or item.endswith(".bmp"):
            images.append(cv2.imread(os.path.join(image_directory, item)))
        else:
            print("{} is not a image.".format(item))
    except:
        print("Failed to read {}.".format(item))
print("{} images found.".format(len(images)))

for image in images:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (500, 500))
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    image_threshold = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    show_img("Original image", image)
