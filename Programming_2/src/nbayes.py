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



def nbayes(data_path: str, no_cv: bool, numb_of_bins: int, m: float):
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
