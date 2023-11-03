import math
from k_nearest_neighbor import KNearestNeighbor
from kmeans import KMeans
import numpy as np
from sklearn.decomposition import PCA
from kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to perform PCA for dimensionality reduction
def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Function to calculate accuracy
def calculate_accuracy(predicted_labels, true_labels):
    correct_predictions = sum([1 for predicted, true in zip(predicted_labels, true_labels) if predicted == true])
    return (correct_predictions / len(true_labels)) * 100

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

def processData(dataset):
    labels = []
    features = []

    for i in range(len(dataset)):
        label = int(dataset[i][0])
        grayscale_values = [int(x) for x in dataset[i][1]]
        
        # Map grayscale values to binary using a threshold (adjust threshold as needed)
        threshold = 128  # Example threshold: Change this to suit your needs
        binary_values = [1 if x > threshold else 0 for x in grayscale_values]
        
        features.append(binary_values)
        labels.append(label)

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
def knn(train,query,metric, testing = False):
    bestK = 0
    bestAcc = 0
    bestLabels = []

    # Set this value to True if you want to try out a range of K Values
    if (testing == False):
        kRange = [7]
    else:
        kRange = range(1,51,2)
    for k in kRange:


        knn = KNearestNeighbor(k, metric, euclidean, cosim)
        xtrain,ytrain = processData(train)
        xtest,ytest = processData(query)
        xtrain = normalize_data(xtrain)
        xtest = normalize_data(xtest)

        # xtrain= perform_pca(xtrain, n_components= 200)  # Apply PCA to training data
        # xtest = perform_pca(xtest, n_components= 200)    # Apply PCA to test data
        
        knn.fit(xtrain,ytrain)
        predictedLabels = knn.predict(xtest)
        
        if(not testing):
            plot_confusion_matrix(ytest, predictedLabels, [i for i in range(10)])

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

def kmeans(train,query,metric,soft = False, beta = 1000, testing = False, testingSoft = False):

    xtrain, ytrain = processData(train)
    xtest, ytest = processData(query)

    xtrain = normalize_data(xtrain)
    xtest = normalize_data(xtest)

    # xtrain= perform_pca(xtrain, n_components= 200)  # Apply PCA to training data
    # xtest = perform_pca(xtest, n_components= 200)    # Apply PCA to test data

    bestK = 0
    bestAccuracy = 0
    bestLabels = []
    krange = [23]

    if(testing):
        krange = [19]
    
    for k in krange:
        b  = [beta]
        if(testingSoft):
            b = range(2, 50, 2)

        for bt in b:
                

            kmeans = KMeans(n_clusters = k, metric = metric, euclidean = euclidean, cosim = cosim, soft=soft, beta = bt)
            kmeans.fit(xtrain)

            _, train_cluster_assignments = kmeans.predict(xtrain)
            centroid_to_label = map_centroids_to_labels(train_cluster_assignments, ytrain)

            # Predict cluster assignments for query set
            _, test_cluster_assignments = kmeans.predict(xtest)
            predicted_labels = [centroid_to_label[cluster_id] for cluster_id in test_cluster_assignments]

            if(not testing):
                plot_confusion_matrix(ytest, predicted_labels, [i for i in range(10)])

            # Compute accuracy
            correct_predictions = sum([1 for predicted, true in zip(predicted_labels, ytest) if predicted == true])
            accuracy = correct_predictions / len(ytest) * 100

            print("Accuracy for k = {} , Beta = {}, is {}%".format(k,bt, accuracy))

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

def visualize_pca_components(xtrain, num_components_range, num_samples=5):
    # Choose a few sample digits
    sample_digits = xtrain[:num_samples]

    # Create a subplot for each component value
    plt.figure(figsize=(15, 5))
    for i, num_components in enumerate(num_components_range):
        plt.subplot(1, len(num_components_range), i + 1)
        plt.title(f'{num_components} Components')

        # Fit PCA with the chosen number of components
        pca = PCA(n_components=num_components)
        pca.fit(xtrain)

        # Transform and inverse transform sample digits
        reduced_digits = pca.transform(sample_digits)
        reconstructed_digits = pca.inverse_transform(reduced_digits)
        
        # Display the reconstructed digits
        for j in range(len(sample_digits)):
            plt.imshow(reconstructed_digits[j].reshape(28, 28), cmap='gray')
            plt.axis('off')
        if i == 0:
            plt.ylabel('Sample Digits')

    plt.show()

from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized


def main():
    trainData = read_data("train.csv")
    testData = read_data("valid.csv")

    # # Prepare data for K-Means and KNN
    # xtrain, ytrain = processData(trainData)
    # xtest, ytest = processData(testData)

    # # Specify a range of component values to visualize
    # num_components_range = [20, 50, 75, 100, 150, 200, 400, 600]
    # # Visualize PCA components
    # visualize_pca_components(xtrain, num_components_range)
    
    # K-Means with Euclidean distance
    print("K-Means Results - Euclidean :")
    predicted_labels, accuracy, k = kmeans(trainData, testData, metric = "euclidean", testing=True)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans = {}".format(accuracy))

    # K-Means with Euclidean distance for Soft K Means
    print("K-Means Results - Euclidean :")
    predicted_labels, accuracy, k = kmeans(trainData, testData, metric = "euclidean", soft=True, beta=5, testing=True)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means= {}".format(accuracy))

    # K-Means with Cosine similarity
    print("K-Means Results - Cosim :")
    predicted_labels, accuracy, k = kmeans(trainData, testData, metric = "cosim", testing=True)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans = {}".format(accuracy))

    # K-Means with Euclidean distance for Soft K Means
    print("K-Means Results - Cosim :")
    predicted_labels, accuracy, k = kmeans(trainData, testData, metric = "cosim", soft=True, beta=5, testing=True)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means with Cosim= {}".format(accuracy))

    # KNN with Euclidean distance
    print("\nK-Nearest Neighbors Results - Euclidean:")
    knn(trainData,testData,"euclidean", testing=True)

    # KNN with Cosim distance
    print("\nK-Nearest Neighbors Results - Euclidean:")
    knn(trainData,testData,"cosim", testing=True)
    


    actualTest = read_data("test.csv")
    # KNN with Euclidean distance
    print("\nK-Nearest Neighbors Results - Euclidean:")
    knn(trainData,actualTest,"euclidean", testing=False)

    # KNN with Cosine similarity
    print("K-Nearest Neighbors Results - Cosim:")
    knn(trainData,actualTest,"cosim", testing=False)

        # K-Means with Euclidean distance
    print("K-Means Results - Euclidean :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "euclidean")
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans = {}".format(accuracy))

    # K-Means with Euclidean distance for Soft K Means
    print("K-Means Results - Euclidean :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "euclidean", soft=True, beta=5)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means= {}".format(accuracy))

    # K-Means with Cosine similarity
    print("K-Means Results - Cosim :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "cosim")
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans = {}".format(accuracy))

    # K-Means with Euclidean distance for Soft K Means
    print("K-Means Results - Cosim :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "cosim", soft=True, beta=5)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means with Cosim= {}".format(accuracy))
    

    # Testing BETA Values
    # K-Means with Euclidean distance for Soft K Means
    print("K-Means Results - Euclidean :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "euclidean", soft=True, beta=5, testingSoft=True)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means= {}".format(accuracy))
    
    actualTest = read_data("test.csv")
    # K-Means with Euclidean distance for 
    print("K-Means Results - Cosim :")
    predicted_labels, accuracy, k = kmeans(trainData, actualTest, metric = "cosim", soft=False, beta=4)
    # print("Predicted Labels for test data:", predicted_labels)
    # print("We found that we have best accuracy when k = {}, giving us accuracy of {}".format(k, accuracy))
    print("accuracy for KMeans With Soft K Means= {}".format(accuracy))

if __name__ == "__main__":
    main()
        
  
