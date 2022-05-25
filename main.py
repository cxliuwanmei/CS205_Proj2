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
            if(sortedEucDises[1] == eucDises[i]): #Find the index of the nearest neighbor of the sample point
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
    if(dirction == 1):
        for j in range(0, n):
            lvlErrRate = 1
            for i in range(0, n):
                if(featureInList(i, bestUpperLvlFeatureList)):
                    continue;
                tempFeatureList = bestUpperLvlFeatureList[:]
                tempFeatureList.append(i)
                errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
                print("    Using feature(s) " + str(tempFeatureList) + " accuracy is " + str((1 - errRate) * 100) + "%")
                if(errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            print("Best feature set is " + str(bestLvlFeatureList) + " accuracy is " + str((1 - lvlErrRate) * 100) + "%")
            bestUpperLvlFeatureList = bestLvlFeatureList
            bestFeatureList.append([lvlErrRate, np.sort(bestLvlFeatureList, kind='quicksort', order=None)])
            # print("Forward level " + str(j) + " lvlErrRate: " + str(lvlErrRate))
            # print("Forward level " + str(j) + " bestLvlFeatureList: " + str(bestLvlFeatureList))
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
                errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
                print("    Using feature(s) " + str(tempFeatureList) + " accuracy is " + str((1 - errRate) * 100) + "%")
                if (errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            print("Best feature set is " + str(bestLvlFeatureList) + " accuracy is " + str((1 - lvlErrRate) * 100) + "%")
            bestLowerLvlFeatureList = bestLvlFeatureList
            bestFeatureList.append([lvlErrRate, bestLvlFeatureList])
    errRate = 1
    idx = -1
    for i in range(0, len(bestFeatureList)):
        tempFeatureList = bestFeatureList[i]
        if(errRate > tempFeatureList[0]):
            errRate = tempFeatureList[0]
            idx = i
    print("Finish searchinng. The best featrue set is " + str(bestFeatureList[idx][1]) + " accuracy is " + str((1 - bestFeatureList[idx][0]) * 100) + "%")
    return bestFeatureList

if __name__ == '__main__':
    print("Please select a test file:")
    print("1. CS205_SP_2022_Largetestdata__60.txt")
    print("2. CS205_SP_2022_SMALLtestdata__78.txt")
    test_file = input("Input 1 or 2:")
    if(test_file == 1):
        file = "CS205_SP_2022_Largetestdata__60.txt"
    elif(test_file == 2):
        file = "CS205_SP_2022_SMALLtestdata__78.txt"
    else:
        print("Please input a correct value.")
        exit(-1)
    print("Please select a test algorithm:")
    print("1->Forward Selection")
    print("2->Backward Elimination")
    direction = input("Input 1 or 2:")
    if(direction != 1 and direction != 2):
        print("Please input a correct value.")
        exit(-1)
    (test_class_data, test_features_data) = load_data(file)
    bestFeatureList = selection(test_features_data, test_class_data, direction)

    # (large_test_class_data, large_test_features_data) = load_data("CS205_SP_2022_Largetestdata__60.txt")
    # bestFeatureList = selection(large_test_features_data, large_test_class_data, 0)
    print("bestFeatureList:" + str(bestFeatureList))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
