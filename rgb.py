# coding=utf-8
import cv2
from matplotlib import pyplot as plt
import numpy as np


def nothing(x):
    pass


if __name__ == '__main__':
    # rgb = cv2.imread("rgb.png")
    # depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)

    rgb = cv2.imread("img.jpg")
    # 逐像素计算深度，模拟深度图
    depth = np.zeros([450, 800], np.int32)
    for i in range(depth.shape[1]):
        for j in range(depth.shape[0]):
            # 给小猪部分设置深度
            if (i - 300) * (i - 300) + (j - 228) * (j - 228) <= 115 * 115:
                depth[j, i] = (300 - i) * (300 - i) + (228 - j) * (228 - j)
            # 给小鸡部分设置深度
            elif (i - 563) * (i - 563) + (j - 216) * (j - 216) <= 124 * 124:
                depth[j, i] = (563 - i) * (563 - i) + (216 - j) * (216 - j)
            # 其它区域设置深度
            else:
                depth[j, i] = 0.5 * (i - 400) * (i - 400) + (j - 225) * (j - 225) + 100

    plt.imshow(depth, cmap='gray')
    plt.show()

    max_dis = int(np.max(depth))
    min_dis = int(np.min(depth))
    th_dis = int(np.mean(depth))
    print 'max dis:', max_dis
    print 'min dis:', min_dis
    print 'threshold dis:', th_dis

    rgb_res = np.zeros(rgb.shape, rgb.dtype)
    rgb_blur = cv2.GaussianBlur(rgb, (31, 31), 0)

    # 逐像素判断深度是否大于阈值
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if depth[i, j] >= th_dis:
                rgb_res[i, j] = rgb_blur[i, j]
            else:
                rgb_res[i, j] = rgb[i, j]

    cv2.namedWindow("BlurWithDepth")
    cv2.createTrackbar('Threshold', 'BlurWithDepth', th_dis, max_dis, nothing)

    while 1:
        cv2.imshow("BlurWithDepth", rgb_res)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        th_dis = cv2.getTrackbarPos('Threshold', 'BlurWithDepth')

        # 逐像素判断深度是否大于阈值
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if depth[i, j] >= th_dis:
                    rgb_res[i, j] = rgb_blur[i, j]
                else:
                    rgb_res[i, j] = rgb[i, j]

    cv2.destroyAllWindows()
