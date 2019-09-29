import cv2
import numpy as np


def rbg2gray(img, scale):
    # pic = np.zeros(np.shape(img),np.uint8)
    height = img.shape[0]
    width = img.shape[1]
    pic_gray = np.zeros((height, width), np.uint8)
    #print(np.shape(pic_gray))
    #print(np.shape(r))
    for row in range(height):
        for col in range(width):
            pic_gray[row, col] = (r[row, col] * 299 + g[row, col] * 587 + b[row, col] * 114) / 1000 * scale
    cv2.imshow("GRAY_"+str(scale), pic_gray)
    cv2.imwrite("D:\\python_project\\GRAY_"+str(scale)+".png", pic_gray)


img = cv2.imread('lena_bmp.bmp')
cv2.imshow('bmp',img)
#b, g, r = cv2.split(img)
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
#print(b.shape)

pic_b = np.zeros(np.shape(img), np.uint8)
pic_b[:, :, 0] = b
cv2.namedWindow('Blue')
cv2.imshow('Blue', pic_b)
cv2.imwrite("D:\\python_project\\Blue.png", pic_b)

pic_g = np.zeros(np.shape(img), np.uint8)
pic_g[:, :, 1] = g
cv2.namedWindow('Green')
cv2.imshow('Green', pic_g)
cv2.imwrite("D:\\python_project\\Green.png", pic_g)

pic_r = np.zeros(np.shape(img), np.uint8)
pic_r[:, :, 2] = r
cv2.namedWindow('Red')
cv2.imshow('Red', pic_r)
cv2.imwrite("D:\\python_project\\Red.png", pic_r)

rbg2gray(img, 0.5)
rbg2gray(img, 0.25)


cv2.waitKey(0)
cv2.destroyAllWindows()
