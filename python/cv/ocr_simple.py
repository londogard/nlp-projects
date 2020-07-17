import cv2
import pytesseract
import imutils

img = cv2.imread('images/example_02.jpg')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
print(pytesseract.image_to_string(img, config=custom_config))