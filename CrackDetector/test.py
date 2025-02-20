import cv2
import numpy as np
import os
import argparse

test_mode = False

# Results Presentation Function 用于结果展示的函数
def show_res (title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Phased Presentation Function 用于阶段性展示的函数
def show_img(title, img):
    global test_mode
    if test_mode is True:
        show_res(title, img)

# Args Parser:
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

# Read Images
image_name = os.listdir(image_directory)
images = []
for index, item in enumerate(image_name):
    try:
        if item.endswith(".jpg") or item.endswith(".png") or item.endswith(".jpeg") or item.endswith(".bmp"):
            images.append(cv2.imread(os.path.join(image_directory, item), cv2.IMREAD_COLOR))
        else:
            print("{} is not a image.".format(item))
            del image_name[index]
    except:
        print("Failed to read {}.".format(item))
        del image_name[index]
print("{} images found.".format(len(images)))

# Process Images
for index, image in enumerate(images):
    show_img("Original image " + image_name[index], image)
    image_resized = cv2.resize(image, (512, 512))
    show_img("Resized image " + image_name[index], image_resized)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    show_img("Grayscale image " + image_name[index], image_gray)
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    show_img("Blurred image " + image_name[index], image_blurred)
    image_edged = cv2.Canny(image_blurred, 50, 150)
    show_img("Edged image " + image_name[index], image_edged)
    image_equalized = cv2.equalizeHist(image_blurred)
    show_img("Equalized image " + image_name[index], image_equalized)
    image_threshold = cv2.adaptiveThreshold(image_equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    show_img("Threshold image " + image_name[index], image_threshold)
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    image_opened =cv2.morphologyEx(image_threshold, cv2.MORPH_CLOSE, kernel_open)
    show_img("Opened image " + image_name[index], image_opened)
    image_contour, _ = cv2.findContours(image_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_contour_drew = cv2.drawContours(cv2.cvtColor(image_opened.copy(), cv2.COLOR_GRAY2BGR), image_contour, -1, (0, 0, 255), 1)
    show_img("Contour image " + image_name[index], image_contour_drew)
    filtered_contour = [contour for contour in image_contour if cv2.arcLength(contour, True) > 100]
    filtered_contour_drew = cv2.drawContours(image_resized.copy(), filtered_contour, -1, (0, 0, 255), 1)
    show_img("Filtered contour image " + image_name[index], filtered_contour_drew)