import cv2 as cv
import numpy as np


def video2image():
    cap = cv.VideoCapture('D:\\python_project\\Megamind.avi')
    i=0
    while(True):
        rect, frame = cap.read()
        if rect == False:
            break
        i += 1
        #cv.imwrite('D:\\python_project\\_frame_' + str(i) + '.jpg',frame)
        cv.imshow("video", frame)
        print(frame.shape)
        c = cv.waitKey(100)
        if c == 27:
            break

    cap.release()


def subtract_demo():
    img2 = cv.imread('D:\\python_project\\WindowsLogo.jpg')
    img1 = cv.imread('D:\\python_project\\LinuxLogo.jpg')
    if img1.shape != img2.shape:
        print('two images shape are different')
    rows = img1.shape[0]
    cols = img1.shape[1]
    img_sub = np.zeros((rows, cols, 3), np.uint8)
    for row in range(rows):
        for col in range(cols):
            if img1[row, col, 0] < img2[row, col, 0] :
                img_sub[row, col, 0] = 0
            else:
                img_sub[row, col, 0] = int(img1[row, col, 0])-int(img2[row, col, 0])

            if img1[row, col, 1] < img2[row, col, 1]:
                img_sub[row, col, 1] = 0
            else:
                img_sub[row, col, 1] = int(img1[row, col, 1])-int(img2[row, col, 1])

            if img1[row, col, 2] < img2[row, col, 2]:
                img_sub[row, col, 2] = 0
            else:
                img_sub[row, col, 2] = int(img1[row, col, 2]) - int(img2[row, col, 2])
    cv.imshow("subtract_demo", img_sub)
    cv.imwrite("D:\\python_project\\subtract_demo.png", img_sub)
    dst = cv.subtract(img1, img2)
    cv.imshow("opencv_subtract_demo", dst)


#最邻近差值法
def Nearest(img, bigger_height, bigger_width, channels):
    near_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    for i in range(0,bigger_height):
        for j in range(0, bigger_width):
            row = ( i / bigger_height) * img.shape[0]
            col = ( j / bigger_width) * img.shape[1]
            near_row = round(row)
            near_col = round(col)
            if near_row == img.shape[0] or near_col == img.shape[1]:
                near_row -= 1
                near_col -= 1

            near_img[i, j] = img[near_row, near_col]
    return near_img


def Cut(img, height, width):
    dst = np.zeros((height, width), np.uint8)
    x_begin = int(img.shape[0]*0.25)
    x_end = x_begin+height
    y_begin = int(img.shape[1]*0.25)
    y_end = y_begin + width
    dst = img[x_begin:x_end, y_begin:y_end]
    return dst



def bitwise_and_demo(img1, img2):
    if img1.shape != img2.shape:
        print('the mask shape is different from video frame')
    rows = img1.shape[0]
    cols = img1.shape[1]
    dst = np.zeros((rows, cols, 3), np.uint8)
    for row in range(rows):
        for col in range(cols):
            #红外效果只显示G通道，R,B通道值为0
            #dst[row, col, 0] = (int(img1[row, col, 0]) & int(img2[row, col, 0]))
            dst[row, col, 1] = (int(img1[row, col, 1]) & int(img2[row, col, 1]))
            #dst[row, col, 2] = (int(img1[row, col, 2]) & int(img2[row, col, 2]))
    return dst


def video_process():
    videoCapture = cv.VideoCapture('D:\\python_project\\Megamind.avi')
    fps = videoCapture.get(cv.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print(size)
    #绘制一个圆形的望远镜形状
    Circle = np.zeros((528, 720, 3), np.uint8)
    cv.circle(Circle, (360, 264), 200, (255, 255, 255), -1)
    #绘制望远镜中心的十字架
    cv.line(Circle, (350, 264), (370, 264), (0, 0, 0), 3)
    cv.line(Circle, (360, 254), (360, 274), (0, 0, 0), 3)
    cv.imshow('Mask', Circle)
    cap = cv.VideoCapture('D:\\python_project\\Megamind.avi')
    out = cv.VideoWriter('D:\\python_project\\video_process.avi',cv.VideoWriter_fourcc('I','4','2','0'), fps, size)
    while(True):
        rect, frame = cap.read()
        if rect == False:
            break
        near_img = Nearest(frame, int(frame.shape[0]*1.5), int(frame.shape[1]*1.5), 3)
        cut_img = Cut(near_img, frame.shape[0], frame.shape[1])
        dst = bitwise_and_demo(Circle, cut_img)
        #dst = cv.bitwise_and(Circle,frame)
        cv.imshow("video_process", dst)
        out.write(dst)
        c = cv.waitKey(20)
        if c == 27:
            break

    cap.release()


def main():
    #video2image()
    subtract_demo()
    video_process()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()