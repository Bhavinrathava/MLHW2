import math
from k_nearest_neighbor import KNearestNeighbor
from kmeans import KMeans
import numpy as np
from saample_kmeans import KMeans

def map_centroids_to_labels(cluster_assignments, actual_labels):
    # Count the labels for each cluster
    mapping = {}
    for cluster_id, label in zip(cluster_assignments, actual_labels):
        if cluster_id not in mapping:
            mapping[cluster_id] = {}
        if label not in mapping[cluster_id]:
            mapping[cluster_id][label] = 0
        mapping[cluster_id][label] += 1

    # Determine the most frequent label for each cluster
    centroid_to_label = {}
    for cluster_id, label_counts in mapping.items():
        centroid_to_label[cluster_id] = max(label_counts, key=label_counts.get)
        
    return centroid_to_label

# split training data into labels and features
def processData(dataset):
    labels = []
    features = []

    for i in range(len(dataset)):
        features.append([int(x) for x in dataset[i][1]])
        labels.append(int(dataset[i][0]))

    return features, labels

# returns Euclidean distance between vectors a and b
def euclidean(a,b):
    distance = 0
    
    for i,j in zip(a,b):
        distance += (i-j) ** 2

    distance = math.sqrt(distance)
    return distance

        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    dist = 1 - (dot_product / (norm_a * norm_b))
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
        kRange = [3]
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

    xtrain, ytrain = processData(train)
    xtest, ytest = processData(query)

    bestK = 0
    bestAccuracy = 0
    bestLabels = []
    for k in range(5,20):

        kmeans = KMeans(n_clusters = k, metric = metric, euclidean = euclidean, cosim = cosim )
        kmeans.fit(xtrain)

        _, train_cluster_assignments = kmeans.predict(xtrain)
        centroid_to_label = map_centroids_to_labels(train_cluster_assignments, ytrain)

        # Predict cluster assignments for query set
        _, test_cluster_assignments = kmeans.predict(xtest)
        predicted_labels = [centroid_to_label[cluster_id] for cluster_id in test_cluster_assignments]
    
        # Compute accuracy
        correct_predictions = sum([1 for predicted, true in zip(predicted_labels, ytest) if predicted == true])
        accuracy = correct_predictions / len(ytest) * 100

        print("Accuracy for k = {} is {}%".format(k, accuracy))

        if(accuracy > bestAccuracy):
            bestK = k
            bestLabels = predicted_labels
            bestAccuracy = accuracy

    #print(f"Accuracy: {accuracy:.2f}%")
    print(ytest)

    return bestLabels, bestAccuracy, bestK

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

    predicted_labels, accuracy, k = kmeans(trainData, testData, metric = "euclidean")
    
    print("Predicted Labels for test data:", predicted_labels)
    print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans = {}".format(accuracy))
    
    knn(trainData,testData,"euclidean")
    
if __name__ == "__main__":
    main()
    