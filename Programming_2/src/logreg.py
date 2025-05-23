import argparse
import os
import numpy as np
from typing import List
from sting.data import Feature, FeatureType, parse_c45
import util

class Regression:
    def __init__(self, schema: List[Feature], learningRate: float = 0.001, regLambda: float = 0.0, maxIteration: int = 10000, tol: float = 1e-6):
        self.schema = schema
        self.learningRate = learningRate
        self.regLambda = regLambda
        self.maxIteration = maxIteration
        self.tol = tol
        self.weights = None
        self.uniqueValues = {}
        self.means, self.stds = None, None 

    def missingValues(self, X: np.ndarray):
        # we are filling the continuos values with mean for missing values and mode for categorical values
        for i, feature in enumerate(self.schema):
            if np.any(np.isnan(X[:, i])):
                if feature.ftype == FeatureType.CONTINUOUS:
                    mean_val = np.nanmean(X[:, i])
                    X[:, i] = np.where(np.isnan(X[:, i]), mean_val, X[:, i])
                elif feature.ftype == FeatureType.NOMINAL:
                    mode_val = np.argmax(np.bincount(X[~np.isnan(X[:, i]), i].astype(int)))
                    X[:, i] = np.where(np.isnan(X[:, i]), mode_val, X[:, i])
        return X

    def featureStandardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        # to have zero mean and unit variance here we are standardizing the features
        if fit:
            self.means = np.nanmean(X, axis=0)
            self.stds = np.nanstd(X, axis=0)
        return (X - self.means) / (self.stds + 1e-6)  # Avoid division by zero

    def oneHotEncoding(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        # this methos is for Encode the nominal attributes as vectors to make sure the consistency of traing and testing data
        encodedFeatures = []
        for i, feature in enumerate(self.schema):
            if feature.ftype == FeatureType.NOMINAL:
                if fit:
                    self.uniqueValues[i] = np.unique(X[:, i])
                values = self.uniqueValues[i]
                one_hot = (X[:, i].reshape(-1, 1) == values).astype(float)
                encodedFeatures.append(one_hot)
            else:
                encodedFeatures.append(X[:, i].reshape(-1, 1))
        return np.hstack(encodedFeatures)

    def sigmoid(self, k: np.ndarray) -> np.ndarray:
        # we are using the sigmoid acativating function
        k = np.clip(k, -500, 500)  
        return 1 / (1 + np.exp(-k))

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Untill the convergence here we are training the logistic regression model by using the gradient descent
        X = self.missingValues(X)
        X = self.featureStandardize(X, fit=True)
        X = self.oneHotEncoding(X, fit=True)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in range(self.maxIteration):
            linearModel = np.dot(X, self.weights)
            predictions = self.sigmoid(linearModel)

            errors = predictions - y
            gradient = np.dot(X.T, errors) / n_samples
            regularization = self.regLambda * self.weights
            weightUpdate = self.learningRate * (gradient + regularization)

            self.weights =self.weights - weightUpdate
            if np.linalg.norm(weightUpdate, ord=1) < self.tol:
                print("Converged after", i+1,  "iterations.")
                break

    def predictProbability(self, X: np.ndarray) -> np.ndarray:
        # this method is to predict the probabilities for each and every class
        X = self.featureStandardize(X)
        X = self.oneHotEncoding(X)
        linearModel = np.dot(X, self.weights)
        predictPorb = self.sigmoid(linearModel)
        return predictPorb

    def predict(self, X: np.ndarray) -> np.ndarray:
        # To predic the binary labels by using the class probabilities
        probaility = self.predictProbability(X)
        value = (probaility >= 0.5).astype(int)
        return value

def logisticRegression(data_path: str, use_cross_validation: bool, regLambda: float):
    # Logistic Regression method to run on the data, cross-validation is optional
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    accuracies, precisions, recalls = [], [], []
    yTrueLable, allConfidences = [], []

    for X_train, y_train, X_test, y_test in datasets:
        model = Regression(schema=schema, regLambda=regLambda)
        model.fit(X_train, y_train)
        
        # for evaluation purpose we created predict and predict probability method
        y_pred = model.predict(X_test)
        confidences = model.predictProbability(X_test)

        # gathering all true labels and conficences
        yTrueLable.extend(y_test)
        allConfidences.extend(confidences)

        # calculate the metrics and append to respective list
        accuracies.append( util.accuracy(y_test, y_pred))
        precisions.append(util.precision(y_test, y_pred))
        recalls.append(util.recall(y_test, y_pred))

    
    #here we are colleting the all true labels and confidences aofr future (auc) calculatoin
    yTrueLable = np.array(yTrueLable)
    allConfidences = np.array(allConfidences)
    auc = util.auc(yTrueLable, allConfidences)

    # print the accuracy,precisoin and Recall,ROC in required format
    print(f"Accuracy: {np.mean(accuracies):.3f} {np.std(accuracies):.3f}")
    print(f"Precision: {np.mean(precisions):.3f} {np.std(precisions):.3f}")
    print(f"Recall: {np.mean(recalls):.3f} {np.std(recalls):.3f}")
    print(f"Area under ROC: {auc:.3f}")
    # Return the mean accuracy, precision and recalls and area uner curve(auc)
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), auc



if __name__ == '__main__':
    # To run the logistic regression with command-line arguments
    parser = argparse.ArgumentParser(description='Run a logistic regression algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('regLambda', metavar='LAMBDA', type=float, help='Regularization constant for L2 regularization.')
    parser.add_argument('--no-cv', dest='cv', action='store_true',
                        help='Enables cross-validation and trains on the dataset with it.')
    parser.set_defaults(cv=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    regLambda = args.regLambda
    use_cross_validation = args.cv

    # to call the logistic regression method
    logisticRegression(data_path, use_cross_validation, regLambda)
