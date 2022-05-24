import numpy as np
import matplotlib.pyplot as plt
import time

def load_data(file):
    class_val = []
    features_val = []
    f = open(file, "r")
    lines = f.readlines()
    for line in lines:
        # print line
        dataLine = []
        dataStr = line.split()
        class_val.append(float(dataStr[0]))
        for i in range(1, len(dataStr)):
            dataLine.append(float(dataStr[i]))
        features_val.append(dataLine)
    f.close()
    return np.array(class_val, dtype=np.float), np.array(features_val, dtype=np.float)

def nnAlg(trainX, trainY, featuresIdxs):
    predY = []
    for k in range(0, len(trainX)):
        sampleX = np.array([trainX[k]])
        trainXTemp = trainX[:, featuresIdxs]
        sampleXTemp = sampleX[:, featuresIdxs]
        eucDises = np.sqrt(np.sum(np.square(trainXTemp - sampleXTemp), axis=1))
        # dict = {}
        # for i in range(0, len(eucDises)):
        #     dict[eucDises[i]] = i
        sortedEucDises= np.sort(eucDises, kind='quicksort', order=None)
        for i in range(0, len(eucDises)):
            if(sortedEucDises[1] == eucDises[i]):
                break
        idx = i #dict[sortedEucDises[1]]
        predY.append(trainY[idx])
    # print(predY)
    return predY

def errorRate(Y,predy):
    return (predy != Y).mean()

def featureInList(feature, featureList):
    for i in range(0, len(featureList)):
        if(feature == featureList[i]):
            return True
    return False

def selection(trainX, trainY, dirction):
    bestLvlFeatureList = []
    bestUpperLvlFeatureList = []
    bestLowerLvlFeatureList = []
    bestFeatureList = []
    (m,n) = trainX.shape
    if(dirction == 0):
        for j in range(0, n):
            lvlErrRate = 1
            for i in range(0, n):
                if(featureInList(i, bestUpperLvlFeatureList)):
                    continue;
                tempFeatureList = bestUpperLvlFeatureList[:]
                tempFeatureList.append(i)
                predY = nnAlg(trainX, trainY, tempFeatureList)
                errRate = errorRate(trainY, predY)
                if(errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            bestUpperLvlFeatureList = bestLvlFeatureList
            bestFeatureList.append([lvlErrRate, bestLvlFeatureList])
    else:
        for i in range(0, n):
            bestLowerLvlFeatureList.append(i)
        lvlErrRate = errorRate(trainY, nnAlg(trainX, trainY, bestLowerLvlFeatureList)) #Calculate the error rate with all features
        bestFeatureList.append([lvlErrRate, bestLowerLvlFeatureList])
        for j in range(0, n - 1):
            lvlErrRate = 1
            for i in range(0, (n - j)):
                tempFeatureList = bestLowerLvlFeatureList[:]
                del tempFeatureList[i]
                predY = nnAlg(trainX, trainY, tempFeatureList)
                errRate = errorRate(trainY, predY)
                if (errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            bestLowerLvlFeatureList = bestLvlFeatureList
            bestFeatureList.append([lvlErrRate, bestLvlFeatureList])
    return bestFeatureList

if __name__ == '__main__':
    # XTest = np.array([[2.0, -1.0], [2.0, -1.0]])
    # eucDis(XTest, np.array([XTest[0]]), [0, 1])
    (small_test_class_data, small_test_features_data) = load_data("CS205_SP_2022_SMALLtestdata__78.txt")
    selection(small_test_features_data, small_test_class_data, 0)
    # featuresIdxs = [0, 2]
    # predY = nnAlg(small_test_features_data, small_test_class_data, featuresIdxs)
    # errRate = errorrate(small_test_class_data, predY)
    print("errRate:" + str(errRate))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
