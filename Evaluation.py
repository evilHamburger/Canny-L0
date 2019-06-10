from scipy.io import loadmat
import PIL.Image
import numpy as np
import os

def evaluate(inDir, thresholds = 99):
    numList = [file[0: len(file)- 4] for file in os.listdir(inDir)]
    recList = []
    preList = []
    accList = []
    f1List = []
    for num in numList:
        gt = loadmat('BSR/BSDS500/data/groundTruth/test/{0}.mat'.format(num))['groundTruth']
        gtb = [gt[0,i]['Boundaries'][0,0] for i in range(gt.shape[1])]
        preEdge = np.array(PIL.Image.open(inDir + '/{0}.jpg'.format(num)))
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for g in gtb:
            TP += np.sum((g > 0) & (preEdge > 127))
            FN += np.sum((g > 0) & (preEdge <= 127))
            FP += np.sum((g <= 0) & (preEdge > 127))
            TN += np.sum((g <= 0) & (preEdge <= 127))
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        F1 = 2 * precision * recall / (precision + recall)
        print('recall :' + str(recall))
        print('precision :' + str(precision))
        print('accuracy :' + str(accuracy))
        print('F1 score :' + str(F1))
        recList.append(recall)
        preList.append(precision)
        accList.append(accuracy)
        f1List.append(F1)
    print('completed')
    print('recall :' + str(np.mean(recList)))
    print('precision :' + str(np.mean(preList)))
    print('accuracy :' + str(np.mean(accList)))
    print('F1 score :' + str(np.mean(f1List)))
evaluate(inDir = 'HEDOutput')

