import numpy as np
import matplotlib.pyplot as plt
import time


def load_data(file):
    class_val = []
    features_val = []
    f = open(file, "r")
    lines = f.readlines()
    for line in lines:
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
    trainXTemp = trainX[:, featuresIdxs]
    for k in range(0, len(trainX)):
        sampleX = np.array([trainX[k]])
        sampleXTemp = sampleX[:, featuresIdxs]
        eucDises = np.sqrt(np.sum(np.square(trainXTemp - sampleXTemp), axis=1))
        sortedEucDises = np.argsort(eucDises)
        predY.append(trainY[sortedEucDises[1]])
    return predY


def errorRate(Y, predy):
    return (predy != Y).mean()


def featureInList(feature, featureList):
    for i in range(0, len(featureList)):
        if (feature == featureList[i]):
            return True
    return False


def selection(trainX, trainY, dirction):
    bestLvlFeatureList = []
    bestUpperLvlFeatureList = []
    bestLowerLvlFeatureList = []
    bestFeatureList = []
    (m, n) = trainX.shape
    if (dirction == 1):
        for j in range(0, n):
            lvlErrRate = 1
            for i in range(0, n):
                if (featureInList(i, bestUpperLvlFeatureList)):
                    continue;
                tempFeatureList = bestUpperLvlFeatureList[:]
                tempFeatureList.append(i)
                errRate = errorRate(trainY, nnAlg(trainX, trainY, tempFeatureList))
                print("    Using feature(s) " + str(
                    np.sort(tempFeatureList, kind='quicksort', order=None)) + " accuracy is " + str(
                    (1 - errRate) * 100) + "%")
                if (errRate < lvlErrRate):
                    lvlErrRate = errRate
                    bestLvlFeatureList = tempFeatureList
            print("Best feature set is " + str(
                np.sort(bestLvlFeatureList, kind='quicksort', order=None)) + " accuracy is " + str(
                (1 - lvlErrRate) * 100) + "%")
            bestUpperLvlFeatureList = bestLvlFeatureList
            bestFeatureList.insert(0, [lvlErrRate, np.sort(bestLvlFeatureList, kind='quicksort', order=None)])
    else:
        for i in range(0, n):
            bestLowerLvlFeatureList.append(i)
        lvlErrRate = errorRate(trainY, nnAlg(trainX, trainY,
                                             bestLowerLvlFeatureList))  # Calculate the error rate with all features
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
            print("Best feature set is " + str(bestLvlFeatureList) + " accuracy is " + str(
                (1 - lvlErrRate) * 100) + "%")
            bestLowerLvlFeatureList = bestLvlFeatureList
            bestFeatureList.insert(0, [lvlErrRate, bestLvlFeatureList])
    errRate = 1
    idx = -1
    for i in range(0, len(bestFeatureList)):
        tempFeatureList = bestFeatureList[i]
        if (errRate > tempFeatureList[0]):
            errRate = tempFeatureList[0]
            idx = i
    print("Finish searchinng. The best featrue set is " + str(bestFeatureList[idx][1]) + " accuracy is " + str(
        (1 - bestFeatureList[idx][0]) * 100) + "%")
    return np.array(bestFeatureList, dtype=object)


def plotChart(bestFeatureList, large, direction):
    x_data = []
    y_data = []
    title = ""

    if (direction == 1):
        if (large):
            title = "Figure 1: Large dataset with "
        else:
            title = "Figure 3: Small dataset with "
        title += "Forward Selection"
    else:
        if (large):
            title = "Figure 2: Large dataset with "
        else:
            title = "Figure 4: Small dataset with "
        title += "Backward Elimination"

    for i in range(0, len(bestFeatureList)):
        x_data.append(i)
        y_data.append(100 * (1 - bestFeatureList[i][0]))

    fs = 20
    plt.figure(figsize=(15, 10))
    if (large):
        fs = 10
    bars = plt.barh(x_data, y_data, fc='y')

    for i in range(0, len(bars)):
        bar = bars[i]
        height = bar.get_height()
        text = str('%2.2f' % (y_data[i])) + '%'
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, text, fontsize=fs)
        fl = []
        for j in range(0, len(bestFeatureList[i][1])):
            fl.append(bestFeatureList[i][1][j])
        plt.text(1, bar.get_y() + bar.get_height() / 2, str(fl), fontsize=fs)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20)
    plt.xlabel("Accuracy", fontsize=20)
    plt.ylabel("Feature Set", fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Please select a test file:")
    print("1. CS205_SP_2022_Largetestdata__60.txt")
    print("2. CS205_SP_2022_SMALLtestdata__78.txt")
    test_file = int(input("Input 1 or 2:"))
    file = "CS205_SP_2022_SMALLtestdata__78.txt"
    large = False
    print("test_file:" + str(test_file))
    if (test_file == 1):
        file = "CS205_SP_2022_Largetestdata__60.txt"
        large = True
        print("test_file 2:" + str(test_file))
    elif (test_file != 2):
        print("Please input a correct value.")
        exit(-1)
    print("Please select a test algorithm:")
    print("1->Forward Selection")
    print("2->Backward Elimination")
    direction = int(input("Input 1 or 2:"))
    if (direction != 1 and direction != 2):
        print("Please input a correct value.")
        exit(-1)
    (test_class_data, test_features_data) = load_data(file)
    start = time.time()
    bestFeatureList = selection(test_features_data, test_class_data, direction)
    end = time.time()
    print("time:" + str(end - start))
    plotChart(bestFeatureList, large, direction)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/