import cv2
import numpy as np
import queue

# canny edge detection
def canny(mat):
    # 转灰度图
    # cv2.GaussianBlur(mat, (5, 5), 1, 1)
    matG = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    n, m = matG.shape
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = np.zeros(shape = [n, m])
    Gy = np.zeros(shape = [n, m])
    G = np.zeros(shape = [n, m])
    edge = np.zeros(shape = [n, m])
    ## 计算两个方向上的梯度
    ## 使用sobel算子
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            matGx = matG[i - 1:i + 2, j - 1:j + 2] * Sx
            matGy = matG[i - 1:i + 2, j - 1:j + 2] * Sy
            Gx[i][j] = np.sum(matGx.reshape(1, -1))
            Gy[i][j] = np.sum(matGy.reshape(1, -1))
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = abs(Gx) + abs(Gy)

    ## 非极大值抑制
    ## 双阈值抑制
    ret, th = cv2.threshold(matG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    highThreshold = ret
    lowThreshold = 0.4 * highThreshold
    dummyG = np.zeros(shape = [n, m])
    counter = 0
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            ## 线性插值
            gx = Gx[i][j]
            gy = Gy[i][j]
            g = G[i][j]
            N = G[i - 1][j]
            S = G[i + 1][j]
            W = G[i][j - 1]
            E = G[i][j + 1]
            NE = G[i - 1][j + 1]
            NW = G[i - 1][j - 1]
            SE = G[i + 1][j + 1]
            SW = G[i + 1][j - 1]
            theta = np.arctan2(gy, gx) * 180 / np.pi;
            if(theta < 0):
                theta += 180
            g1 = 0
            g2 = 0
            if(theta >= 90 and theta < 135):
                tan = abs(1 / np.tan(theta / 180 * np.pi))
                g1 = (1 - tan) * N + tan * NW
                g2 = (1 - tan) * S + tan * SE

            elif(theta >= 135 and theta < 180):
                tan = abs(np.tan(theta / 180 * np.pi))
                g1 = (1 - tan) * W + tan * NW
                g2 = (1 - tan) * E + tan * SE

            elif(theta >= 45 and theta < 90):
                tan = abs(1 / np.tan(theta / 180 * np.pi))
                g1 = (1 - tan) * N + tan * NE
                g2 = (1 - tan) * S + tan * SW

            else:
                tan = abs(np.tan(theta / 180 * np.pi))
                g1 = (1 - tan) * E + tan * NE
                g2 = (1 - tan) * W + tan * SW

            if(g >= g1 and g >= g2):
                if(g >= highThreshold):
                    edge[i][j] = 255
                    dummyG[i][j] = highThreshold
                    counter += 1
                elif(g >= lowThreshold):
                    dummyG[i][j] = lowThreshold
    print(counter)
    ## 广搜寻找边缘
    counter = 0
    queueI = queue.Queue()
    queueJ = queue.Queue()
    mem = np.zeros(shape = [n, m])
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            if(dummyG[i][j] == highThreshold and mem[i][j] == 0):
                queueI.put(i)
                queueJ.put(j)
                mem[i][j] = 1
                print(i, j)
            while(not queueI.empty()):
                curI = queueI.get()
                curJ = queueJ.get()
                for k in range(curI - 1, curI + 2):
                    for l in range(curJ - 1, curJ + 2):
                        if(mem[k][l] == 1):
                            continue
                        if(k < 0 or k > n - 1 or l < 0 or l > m - 1):
                            continue
                        if(dummyG[k][l] >= lowThreshold):
                            edge[k][l] = 255
                            queueI.put(k)
                            queueJ.put(l)
                            mem[k][l] = 1
                            counter += 1
    print(counter)
    print(highThreshold, lowThreshold)
    return edge
