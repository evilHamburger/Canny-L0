import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from Canny import canny
from L0_Smooth import L0Smooth

# 对图像进行形态变换
def morph(img, edge, thickness, morphLen):
    scale = img.shape[0] / 1000
    r = np.round(thickness * scale)
    r = 1
    ## 边缘取反
    edge = (edge <= 0).astype("uint8")
    ## 图像侵蚀
    se = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (r, r))
    erode = cv2.erode(src = edge, kernel = se)
    ## 图像膨胀
    se2 = np.uint8(np.ones(1).reshape((-1, 1)))
    dilate = cv2.dilate(src = erode, kernel = se2)
    for i in range(img.shape[2]):
        img[:, :, i] = img[:, :, i] * dilate
    return img

def baweraopen(img,size):
    '''
    @image:单通道二值图，数据类型uint8
    @size:欲去除区域大小(黑底上的白区域)
    '''
    output = img.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    for i in range(1, nlabels-1):
        regions_size =  stats[i, 4]
        if regions_size < size:
            x0=stats[i, 0]
            y0=stats[i, 1]
            x1=stats[i, 0] + stats[i, 2]
            y1=stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        output[row, col] = 0
    return output

# 图像离散化
# 减少颜色种类
def quantize(img, colorNum):
    kmeans = MiniBatchKMeans(n_clusters = colorNum)
    N, M, D = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.reshape(N * M, D)
    labels = kmeans.fit_predict(img)
    quant = kmeans.cluster_centers_.astype('uint8')[labels]
    quant = quant.reshape((N, M, D))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

# 提高饱和度
def saturation(img):
    print("提高图片饱和度")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def main():
    img = cv2.imread("img/jordan.jpg")
    newImg = L0Smooth(img)
    newImg = cv2.normalize(newImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("image", newImg)
    cv2.waitKey(0)
    edge = canny(newImg)
    edge = baweraopen(edge.astype('uint8'), 140)
    cv2.imshow("edge", edge)
    cv2.waitKey(0)
    newImg = morph(newImg, edge, 0, 0)
    cv2.imshow("image", newImg)
    cv2.waitKey(0)
    newImg = quantize(newImg, 36)
    cv2.imshow("image", newImg)
    cv2.waitKey(0)
    # newImg = saturation(newImg)
    # cv2.imshow("image", newImg)
    cv2.waitKey(0)
    cv2.imwrite("img/output_jordan.jpg", newImg)


if __name__ == '__main__':
    main()