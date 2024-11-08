import argparse
import os
import numpy as np
from typing import List
from sting.data import Feature, FeatureType, parse_c45
import util



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
