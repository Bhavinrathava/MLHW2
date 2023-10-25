import math
from k_nearest_neighbor import KNearestNeighbor

# split training data into labels and features
def processData(dataset):
    labels = []
    features = []

    for i in range(len(dataset)):
        features.append([int(x) for x in dataset[i][1]])
        labels.append(int(dataset[i][0]))

    return features, labels

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    distance = 0
    
    for i,j in zip(a,b):
        distance += (i-j) ** 2

    distance = math.sqrt(distance)
    return distance

        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    
    return(dist)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    bestK = 0
    bestAcc = 0
    bestLabels = []

    # Set this value to True if you want to try out a range of K Values
    testing = False
    
    if (testing == False):
        kRange = [1]
    else:
        kRange = range(1,51,2)
    for k in kRange:


        knn = KNearestNeighbor(k, metric, euclidean, cosim)
        xtrain,ytrain = processData(train)
        xtest,ytest = processData(query)
        
        knn.fit(xtrain,ytrain)
        predictedLabels = knn.predict(xtest)
        
        crct = 0

        for l in range(len(predictedLabels)):
            if(predictedLabels[l] == ytest[l]):
                crct +=1
        acc = 100 * (crct/len(predictedLabels))
        print("The Accuracy for K = {} is {}%".format(k, acc))

        if(acc>bestAcc):
            bestAcc = acc
            bestK = k
            bestLabels = predictedLabels
    print("we get the best accuracy when we have k={}, with accuracy of {}%".format(bestK,bestAcc))
    return bestLabels
# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    trainData = read_data("train.csv")
    testData = read_data("valid.csv")

    knn(trainData,testData,"euclidean")
    
if __name__ == "__main__":
    main()
    