import numpy as np 
import pandas as pd
from collections import Counter
class KNearestNeighbor():    
    def __init__(self, n_neighbors, euclideanFunction, cosimFunction,distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distanceMeasure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.target = None
        self.euclideanFunction = euclideanFunction
        self.cosimFunction = cosimFunction


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.features = features
        self.target = targets
        
    def findLabel(self,targets, strategy):
        if(strategy =="mean"):
            return sum(targets)/len(targets)
        
        elif(strategy =="mode"):
            data = Counter(targets)
            return data.most_common(1)[0][0]
        else:
            sorted(targets)
            return targets[len(targets)//2]

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
                # Find the eucledian Distance for the point from each of the training value
        distances = {}
        labels = []
        for i in range(len(features)):
            a = features[i]

            for j in range(len(self.features)):
                b = self.features[j]
                if(self.distanceMeasure == "euclidean"):
                    distance = self.euclideanFunction(a,b)
                else:
                    distance = self.cosimFunction(a,b)
                distances[j] = distance
        
    

            # Find the K Nearest Neighnors
            neighbors = []

            for element in sorted(distances, key = distances.get):
                if(len(neighbors) ==self.n_neighbors):
                    break
                neighbors.append(element)
            
            #neighbors2 = distances.argsort()[:self.n_neighbors]

            # Assign the value based on the method of aggregation

            targetNeighbors = []
            for n in neighbors:
                    targetNeighbors.append(self.target[n])

            #targetNeighbors = pd.DataFrame(targetNeighbors)

            # [9, 0, 2, 6, 8, 9, 5, 1, 3, 5]

            labels.append(self.findLabel(targetNeighbors,self.aggregator))
        return labels
