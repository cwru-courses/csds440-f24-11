import argparse
import os
import numpy as np
from typing import List
from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45
import util

class NaiveBayes(Classifier):
    def __init__(self, data_schema: List[Feature], numb_of_bins: int, m: float):
        super().__init__()
        self.numb_of_bins = numb_of_bins
        self.m = m
        self.data_schema = data_schema
        self.feature_bins = {}
        self.cls_probabilities = {}
        self.feature_probabilities = {}
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        #  divides the feature range   into self.bins intervals 
        X_discrete = self.discretize_continuous_features(X)
 
        # Initialize empty Dictionary to store class name and its count
        class_dict = {}

        # Counting the occurrences of each class 
        for i in y:
            if i in class_dict:
                class_dict[i] += 1
            else:
                class_dict[i] = 1

        # store classes and their counts from the dictionary
        unique_classes = list(class_dict.keys())
        class_counts = list(class_dict.values())

        # finding the number of unique classes and total samples
        num_of_classes = len(unique_classes)
        total_samples = len(y)

         
        # Calculate class probabilities with m-estimate or Laplace smoothing
        self.calculate_cls_probs(num_of_classes,unique_classes,class_counts,total_samples)

        # calculate feature probs
        self.calculate_feature_probs(unique_classes,X_discrete,y)

    def calculate_cls_probs(self,num_of_classes,unique_classes,class_counts,total_samples):
            for i in range(num_of_classes):
                cls = unique_classes[i]
                count = class_counts[i]
                
                # Calculating probability
                if self.m >= 0:
                    prob = (count + self.m / num_of_classes) / (total_samples + self.m)
                else:
                    prob = (count + 1) / (total_samples + num_of_classes)  # use Laplace smoothing if m < 0

                # Assign to the dictionary
                self.cls_probabilities[cls] = prob
    def calculate_feature_probs(self,unique_classes,X_discrete,y):
                
        # Initialize feature probabilities as an empty dictionary for each class
        self.feature_probabilities = {}
        for cls in unique_classes:
            cls_idxs = np.where(y == cls)[0]  # Idxs for samples in the perticular cls
            num_of_class_samples = len(cls_idxs)
            self.feature_probabilities[cls] = {}  # Initializing each class dictionary

            nub_of_colums =X_discrete.shape[1]
            for feature_idx in range(nub_of_colums):
                # Get the feature values for the samples of the current class
                feature_values = X_discrete[cls_idxs, feature_idx]
                # defining bin_counts
                bin_counts = np.zeros(self.numb_of_bins, dtype=int)

                # Count the occurrences in each bin
                for value in feature_values:
                    if 0 <= value  and value < self.numb_of_bins:
                        bin_counts[value] = bin_counts[value] + 1
                                
                # Initializing the dictionary for current feature
                self.feature_probabilities[cls][feature_idx] = {}
                
                for bin_idx in range(self.numb_of_bins):
                    # Calculating the probabilities with m-estimate and for Laplace smoothing
                    if self.m < 0:
                        probability = (bin_counts[bin_idx] + 1) / (num_of_class_samples + self.numb_of_bins)
                    else:
                        probability = (bin_counts[bin_idx] + self.m / self.numb_of_bins) / (num_of_class_samples + self.m)
                        

                    # Assigning the probabilities for the  bin idxs
                    self.feature_probabilities[cls][feature_idx][bin_idx] = probability

           
    def discretize_continuous_features(self, X: np.ndarray) -> np.ndarray:
        #Discretizing continuous features into  numb_of_bins
        X_discrete = np.zeros_like(X, dtype=int)
        for idx, feature in enumerate(self.data_schema):
            if feature.ftype == FeatureType.CONTINUOUS:
                min_val = np.min(X[:, idx])
                max_val = np.max(X[:, idx])
                bin_width = (max_val - min_val) / self.numb_of_bins
                self.feature_bins[idx] = (min_val, bin_width)
                z=(X[:, idx] - min_val) / bin_width
                z_floor= np.floor(z).astype(int)
                X_discrete[:, idx] = np.clip(
                    z_floor, 0, self.numb_of_bins - 1
                )
            else:
                X_discrete[:, idx] = X[:,idx]
        return X_discrete

    def predict(self, X: np.ndarray):
        #function to predict labels for input data and return ndarray of predictions.
        predictions =[]
        X_discrete = self.discretize_continuous_features(X)
        for x in X_discrete:
            predictions.append(self.predict_single_input(x))
        
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray):
        #function for probabilities prediction for each class
        X_discrete = self.discretize_continuous_features(X)
        probs =[]
        X_discrete = self.discretize_continuous_features(X)
        for x in X_discrete:
            probs.append(self.predict_prob_single_input(x))
        
        return np.array(probs)

    def predict_single_input(self, x: np.ndarray) -> int:
        #function to Predict the output for each example
        log_probs = self.predict_prob_single_input(x)
        max_logprob_cls= max(log_probs, key=log_probs.get)
        return max_logprob_cls

    def predict_prob_single_input(self, x: np.ndarray) -> dict:
        #function to Calculate log probabilities for each class for a single example.
        log_probs = {}
        for cls in self.cls_probabilities:
            log_prob = np.log(self.cls_probabilities[cls])
            for feature_idx, feature_value in enumerate(x):
                col=self.feature_probabilities[cls][feature_idx]
                prob = col.get(feature_value, 1e-10)
                if prob > 0:
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(1e-10)
            log_probs[cls] = log_prob
        return log_probs


def nbayes(data_path: str, use_cross_validation: bool, numb_of_bins: int, m: float):
    """Run Naive Bayes on the dataset, optionally using cross-validation."""
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]
    root_dir = os.sep.join(path[:-1])
    data_schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    all_y_true, all_confidences, accuracies, precisions, recalls = [], [], [], [], []
     

    for X_train, y_train, X_test, y_test in datasets:
        model = NaiveBayes(data_schema=data_schema, numb_of_bins=numb_of_bins, m=m)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)



        # Calculate metrics for the current fold
        acc = util.accuracy(y_test, y_pred)
        accuracies.append(acc)
        prec = util.precision(y_test, y_pred)
        precisions.append(prec)
        rec = util.recall(y_test, y_pred)
        recalls.append(rec)

        # Get log prob and convert it into confidences for ROC/AUC
        proba_dicts = model.predict_proba(X_test)
        confidences = [np.exp(proba_dicts[i][1]) for i in range(len(y_pred))]

        # Collecting all true labels and its confidences for calculation of  pooled AUC 
        all_y_true.extend(y_test)
        all_confidences.extend(confidences)  
       

    #Calculate AUC
    auc = util.auc(np.array(all_y_true), np.array(all_confidences))

    # Print metrics as per the required format
    print(f"Accuracy: {np.mean(accuracies):.3f} {np.std(accuracies):.3f}")
    print(f"Precision: {np.mean(precisions):.3f} {np.std(precisions):.3f}")
    print(f"Recall: {np.mean(recalls):.3f} {np.std(recalls):.3f}")
    print(f"Area under ROC: {auc:.3f}")

    # Return the metrics as a tuple
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), auc

if __name__ == '__main__':
    """Main function for running Naive Bayes with command-line arguments."""
    parser = argparse.ArgumentParser(description='Run a Naïve Bayes algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('numb_of_bins', metavar='BINS', type=int,
                        help='Number of numb_of_bins for discretizing continuous features (must be at least 2).')
    parser.add_argument('m', metavar='M_ESTIMATE', type=float,
                        help='m-estimate value for smoothing probabilities.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # Input validation for numb_of_bins and m
    if args.numb_of_bins < 2:
        raise argparse.ArgumentTypeError('Number of numb_of_bins must be at least 2.')

    data_path = os.path.expanduser(args.path)
    numb_of_bins = args.numb_of_bins
    m = args.m
    use_cross_validation = args.cv

    nbayes(data_path, use_cross_validation, numb_of_bins, m)

