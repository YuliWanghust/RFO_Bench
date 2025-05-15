import cv2 as cv
import os
from PIL import Image

def globalEqualHist(image, index):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    dst = cv.equalizeHist(gray)
    # cv.imshow("global equalizeHist", dst)
    cv.imwrite("./aaa/"+ str(index) + '.jpg', dst)
def localEqualHist(image, index):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7,7))
    dst = clahe.apply(gray)
    cv.imwrite("./aaa/"+ str(index) + '.jpg', dst)

# src = cv.imread('./preprocess_image/1.2.156.112536.2.560.7050106199148.1363827174830.3530.jpg')
# localEqualHist(src)

img_list = os.listdir('./preprocess_image')
for i, img in enumerate(img_list):
    # src = Image.open(os.path.join('./preprocess_image', img)).convert("RGB")
    src = cv.imread(os.path.join('./preprocess_image', img))
    globalEqualHist(src, i)
#