import cv2 as cv
import numpy as np

#单色 n为色阶数
def gray(n):
    img = np.ones((512, 1024), dtype=np.uint8)
    height = img.shape[0]
    width = img.shape[1]

    for row in range(height):
        for col in range(width):
            t = int(col/width*n+1)
            img[row, col] = (256/n)*t-1

    cv.imshow("img_gray"+str(n), img)
    cv.imwrite("gray_"+str(n)+".png", img)

#彩色 n为色阶数
def color(n):
    img_gray = np.ones((512, 1024), dtype=np.uint8)
    img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    height_color = img_color.shape[0]
    width_color = img_color.shape[1]
    channels = img_color.shape[2]

    for row in range(height_color):
        for col in range(width_color):
            t = int(col / width_color * n + 1)
            img_color[row, col, 0] = 255 - (256/n) * t + 1
            img_color[row, col, 1] = 255 - (256/n) * t + 1
            img_color[row, col, 2] = 64

    cv.imshow("img_color"+str(n), img_color)
    cv.imwrite("color_"+str(n)+".png", img_color)


gray(32)
color(32)
cv.waitKey(0)
cv.destroyAllWindows()

