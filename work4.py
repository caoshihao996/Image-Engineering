import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib


def liner_transform(img1, a, b):
    img = np.array(img1, np.uint8)
    img_liner = np.zeros(img1.shape, np.uint8)
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                piexl = img[row, col, channel]*a+b
                if piexl>255:
                    piexl = 255
                    img_liner[row, col, channel] = piexl
                elif piexl<0:
                    piexl = 0
                    img_liner[row, col, channel] = piexl
                else:
                    img_liner[row, col, channel] = piexl

    cv.imshow("liner_transform", img_liner)
    cv.imwrite("D:\\python_project\\liner_transform.png", img_liner)
    draw_histogram(img_liner)


def water_demo(img):
    img = np.divide(img, 256)
    row, col, channel = img.shape

    img_out = img*1.0
    c_x = (col - 1)/2.0
    c_y = (row - 1)/2.0
    xmk = np.array([list(range(col))]*row)
    ymk = np.array([list(range(row))]*col)
    ymk = np.transpose(ymk)
    xxd = xmk - c_x
    yyd = c_y - ymk
    x = 20*np.sin(2*np.pi*yyd/70)+xxd
    y = 20*np.cos(2*np.pi*xxd/30)+yyd
    xn = x+c_x
    yn = c_y - y
    zzx = np.floor(xn).astype(int)
    zzy = np.floor(yn).astype(int)

    for ii in range(row):
        for jj in range(col):
            new_xx = zzx[ii, jj]
            new_yy = zzy[ii, jj]

            if xn[ii, jj] < 0 or xn[ii, jj] > col - 1:
                continue
            if yn[ii, jj] < 0 or yn[ii, jj] > row - 1:
                continue
            img_out[ii, jj, :] = img[new_yy, new_xx, :]
    cv.imwrite("D:\\python_project\\water_demo.png", img_out*256)
    cv.imshow("water_demo", img_out)


def Sculpture_demo(src):
    img = np.array(src, np.uint8)
    sculpture_img = np.zeros(src.shape, np.uint8)
    rows = img.shape[0]
    cols = img.shape[1]
    channels = img.shape[2]
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                if row == 0 or col == 0:
                    sculpture_img[row, col, channel] = img[row, col, channel]
                else:
                    if img[row, col, channel]<img[row-1, col-1, channel]:
                        sculpture_img[row, col, channel] = 128
                    else:
                        sculpture_img[row, col, channel] = img[row, col, channel]-img[row-1, col-1, channel] + 128
    cv.imshow("Sculpture_demo", sculpture_img)
    cv.imwrite("D:\\python_project\\Sculpture_demo.png", sculpture_img)


#统计图像像素点值的分布，速度太慢
def calHist_demo(src, n):
    img = np.array(src, np.uint8)
    arr = np.zeros((n,),np.int)
    rows = img.shape[0]
    cols = img.shape[1]
    for row in range(rows):
        for col in range(cols):
            for i in range(n):
                if img[row, col] in range(int(i*256/n), int((i+1)*256/n)):
                    arr[i] += 1
    return arr


def equalizationHist_opencv_demo(src):
    img = np.array(src, np.uint8)
    b = cv.equalizeHist(img[:, :, 0])
    g = cv.equalizeHist(img[:, :, 1])
    r = cv.equalizeHist(img[:, :, 2])
    dst = cv.merge([b, g, r])
    cv.imshow("opencv_equlizationHist_demo", dst)
    #cv.imwrite("D:\\python_project\\opencv_equlizationHist_demo.png", dst)


def equalizationHist_demo(src, n):
    t = 256/n
    img = np.array(src, np.uint8)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    bins = np.zeros((n+1,), np.int)
    flag_b = np.zeros((n,), np.int)
    flag_g = np.zeros((n,), np.int)
    flag_r = np.zeros((n,), np.int)
    k = np.zeros((n,), np.double)
    for i in range(n):
        k[i] += i/(n-1)
    for i in range(n+1):
        bins[i] = 256/n*i
    b_hist, bins = np.histogram(b.ravel(), bins)
    g_hist, bins = np.histogram(g.ravel(), bins)
    r_hist, bins = np.histogram(r.ravel(), bins)
    b_cdf = b_hist.cumsum()/(img.shape[0] * img.shape[1])
    g_cdf = g_hist.cumsum()/(img.shape[0] * img.shape[1])
    r_cdf = r_hist.cumsum()/(img.shape[0] * img.shape[1])
    for i in range(n):
        for j in range(n):
            if abs(b_cdf[i] - k[j]) < 1 / ((n - 1) * 2):
                flag_b[i] = j
            if abs(g_cdf[i] - k[j]) < 1 / ((n - 1) * 2):
                flag_g[i] = j
            if abs(r_cdf[i] - k[j]) < 1 / ((n - 1) * 2):
                flag_r[i] = j
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            b_i = int(img[row, col, 0] / t)
            img[row, col, 0] = int(flag_b[b_i] * t) + t / 2
            g_i = int(img[row, col, 1] / t)
            img[row, col, 1] = int(flag_g[g_i] * t) + t / 2
            r_i = int(img[row, col, 2] / t)
            img[row, col, 2] = int(flag_r[r_i] * t) + t / 2
    cv.imshow("equalizationHist_demo", img)
    cv.imwrite("D:\\python_project\\equalizationHist_demo_"+str(n)+".png", img)
    draw_histogram(img)


def histmap(src, dst):
    n = len(dst)
    srcMin = np.zeros((n,n), np.double)
    for y in range(n):
        for x in range(n):
            srcMin[x][y] = abs(src[y] - dst[x])

    lastStartY = 0
    lastEndY = 0
    startY = 0
    endY = 0
    histMap = np.zeros((n,), np.int)
    for x in range(n):
        minValue = srcMin[x][0]
        for y in range(n):
            if minValue > srcMin[x][y]:
                endY = y
                minValue = srcMin[x][y]
        if startY != lastStartY or endY != lastEndY:
            for i in range(startY, endY + 1):
                histMap[i] = x
            lastStartY = startY
            lastEndY = endY
            startY = lastEndY + 1
    return histMap


#对灰度图像进行直方图规定化
def GML_Hist_demo(src, a):
    t = 256/len(a)
    img = np.array(src, np.uint8)
    #b = img[:, :, 0]
    #g = img[:, :, 1]
    #r = img[:, :, 2]
    bins = np.zeros((len(a)+1,), np.int)

    for i in range(len(a) + 1):
        bins[i] = 256 / len(a) * i

    k = np.zeros((len(a),), np.double)
    for i in range(len(a)):
        k[i] += a[i]

    #b_hist, bins = np.histogram(b.ravel(), bins)
    #g_hist, bins = np.histogram(g.ravel(), bins)
    #r_hist, bins = np.histogram(r.ravel(), bins)
    hist, bins = np.histogram(src.ravel(), bins)
    #b_cdf = b_hist.cumsum() / (img.shape[0] * img.shape[1])
    #g_cdf = g_hist.cumsum() / (img.shape[0] * img.shape[1])
    #r_cdf = r_hist.cumsum() / (img.shape[0] * img.shape[1])
    cdf = hist.cumsum() / (img.shape[0] * img.shape[1])
    #histMap_b = histmap(b_cdf, k)
    #histMap_g = histmap(g_cdf, k)
    #histMap_r = histmap(r_cdf, k)
    histMap = histmap(cdf, k)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            i = int(img[row, col] / t)
            img[row, col] = int(histMap[i] * t) + t / 2
            '''
            b_i = int(img[row, col, 0] / t)
            img[row, col, 0] = int(histMap_b[b_i] * t) + t / 2
            g_i = int(img[row, col, 1] / t)
            img[row, col, 1] = int(histMap_g[g_i] * t) + t / 2
            r_i = int(img[row, col, 2] / t)
            img[row, col, 2] = int(histMap_r[r_i] * t) + t / 2
            '''
    cv.imshow("GML_demo", img)
    cv.imwrite("D:\\python_project\\GML_demo.png", img)
    cv.imshow("gray", src)
    #cv.imwrite("D:\\python_project\\gray.png", src)


def draw_histogram(img):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 255])
    plt.savefig("D:\\python_project\\histogram.png")
    plt.show()


def main():
    src = cv.imread("D:\\python_project\\aero1.jpg")
    cv.imshow("src", src)
    #liner_transform(src, 0.5, 64)
    #water_demo(src)
    #Sculpture_demo(src)
    #equalizationHist_demo(src, 8)
    #equalizationHist_opencv_demo(src)

    a = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    GML_Hist_demo(gray, a)

    #draw_histogram(src)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()