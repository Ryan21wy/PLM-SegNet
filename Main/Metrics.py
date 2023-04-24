"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        classAcc[np.isnan(classAcc)] = 0
        return np.round(classAcc, 4)

    def classRecall(self):
        # recall = (TP) / TP + FN
        classRecall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        classRecall[np.isnan(classRecall)] = 0
        return np.round(classRecall, 4)

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) 
        return np.round(meanAcc, 4)

    def meanRecall(self):
        classRecall = self.classRecall()
        meanRecall = np.nanmean(classRecall)
        return np.round(meanRecall, 4)

    def class_F1_score(self):
        classRecall = self.classRecall()
        classRecall[classRecall == 0] = 0.00001
        classAcc = self.classPixelAccuracy()
        class_F1_score = classRecall * classAcc * 2 / (classRecall + classAcc)
        return np.round(class_F1_score, 4)

    def F1_score(self):
        classRecall = self.classRecall()
        meanRecall = np.nanmean(classRecall)
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        F1_score = meanRecall * meanAcc * 2 / (meanRecall + meanAcc)
        return np.round(F1_score, 4)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union 
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)  
        return [np.round(mIoU, 4), np.round(IoU, 4)]

    def genConfusionMatrix(self, imgPredict, imgLabel):  
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusion_Matrix = count.reshape(self.numClass, self.numClass)
        return confusion_Matrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def confusion_Matrix(self):
        confusionMatrix = self.confusionMatrix / np.sum(self.confusionMatrix)
        return confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))