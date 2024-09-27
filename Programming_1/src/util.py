import random
import warnings
from typing import Tuple, Iterable

import numpy as np

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()  # How does this work? What does (y == 1) return?
    n_zeros = y.size - n_ones
    return n_zeros, n_ones

# Entropy definition method
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

# Information Gain Method
def information_gain( y, splits, current_entropy):
        total_samples = len(y)
        weighted_entropy = 0
        for indices in splits.values():
            if len(indices) == 0:
                continue
            subset_entropy = entropy(y[indices])
            weighted_entropy += (len(indices) / total_samples) * subset_entropy
        return current_entropy - weighted_entropy

def split_info_nominal(X_feature):
    """Calculate split information for nominal features."""
    total_samples = len(X_feature)
    _, counts = np.unique(X_feature, return_counts=True)
    probabilities = counts / total_samples
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)

def split_info_continuous(X_feature, threshold):
    """Calculate split information for continuous features based on threshold."""
    total_samples = len(X_feature)
    left = X_feature <= threshold
    right = X_feature > threshold
    counts = np.array([np.sum(left), np.sum(right)])
    probabilities = counts / total_samples
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)

def gain_ratio(y, splits, current_entropy, X_feature=None, threshold=None):
    """Calculate the gain ratio for a split, taking into account split information."""
    gain = information_gain(y, splits, current_entropy)
    
    # Determine split info based on the feature type (nominal or continuous)
    if threshold is not None:
        split_info = split_info_continuous(X_feature, threshold)
    else:
        split_info = split_info_nominal(X_feature)

    # Avoid division by zero
    return gain / split_info if split_info != 0 else 0


def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    datasets = []

    if stratified:
        classes, y_indices = np.unique(y, return_inverse=True)
        class_indices = [np.where(y_indices == i)[0] for i in range(len(classes))]

        fold_indices = [[] for _ in range(folds)]
        for cls_indices in class_indices:
            np.random.shuffle(cls_indices)
            splits = np.array_split(cls_indices, folds)
            for i in range(folds):
                fold_indices[i].extend(splits[i])

        for i in range(folds):
            test_idx = np.array(fold_indices[i])
            train_idx = np.array([idx for idx in indices if idx not in test_idx])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            datasets.append((X_train, y_train, X_test, y_test))
    else:
        np.random.shuffle(indices)
        splits = np.array_split(indices, folds)
        for i in range(folds):
            test_idx = splits[i]
            train_idx = np.hstack([splits[j] for j in range(folds) if j != i])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            datasets.append((X_train, y_train, X_test, y_test))

    return datasets


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    raise NotImplementedError()


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    raise NotImplementedError()