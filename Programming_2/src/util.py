import random
from typing import Tuple, Iterable
import numpy as np

def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    n_ones = (y == 1).sum()
    n_zeros = y.size - n_ones
    return n_zeros, n_ones

# To Calculate the entropy
def entropy(y):
    uq_lbls, count = np.unique(y, return_counts=True)
    #Prob = favourable counts / total counts
    prob = count / len(y)
    #entropy = Prob * log (Prob)
    ent=prob*np.log2(prob + 1e-9) # Small number Added to avoid Log(0)  
    return -np.sum(ent)  

def information_gain(y, splits, cnt_ent):
    ent_sum_weighted = 0
    for i in splits.values():
        splt_ent = entropy(y[i])
        #weighted sum entropy = wighted sum entropy + (length of split index /length of y ) * entropy of y[split index]
        ent_sum_weighted += len(i) / len(y) * splt_ent
    return cnt_ent - ent_sum_weighted

def split_info_nominal(X_feature):
    total = len(X_feature)
    _, feature_count = np.unique(X_feature, return_counts=True)
    #Probability after split
    prob = feature_count / total
    #return entropy after split
    return -np.sum(prob * np.log2(prob + 1e-9))  # Added 1e-9 to avoid log(0)

def split_info_continuous(X_feature, threshold):
    total = len(X_feature)
    #smaller than thresold
    small = X_feature <= threshold
    #larger than thresold
    large = X_feature > threshold
    #count total number
    count = np.array([np.sum(small), np.sum(large)])
    #Prob = favourable /total
    probabilities = count / total
    #return Entropy
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)

def gain_ratio(y, splits, current_entropy, X_feature=None, threshold=None):
    #Calculate Gain
    gain = information_gain(y, splits, current_entropy)
    if (threshold !=None):
        splt = split_info_continuous(X_feature, threshold)
    else:
        splt = split_info_nominal(X_feature)
    #Calculate Gain Ratio= Gain / Split information , Ig gain = 0 then return Gain Ratio = 0
    if(splt!=0):
        temp=gain/splt
    else:
        temp=0
    return temp

def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    np.random.seed(12345)
    random.seed(12345)

    #Calculate the total number of samples  
    total_samples = X.shape[0]
    #Assign index for shuffling
    index = np.arange(total_samples)
    datasets = []
    # we use stratified split to maintain the similar distribution across classess as of original data
    if stratified:
        classes, y_index = np.unique(y, return_inverse=True)

        #Count each class instances
        class_counts = np.bincount(y_index)

        #Generate Each class index 
        class_index = [np.where(y_index == i)[0] for i in range(len(classes))]

        #Distribute the samples from each class across all folds
        index_fold = [[] for _ in range(folds)]
        for cls_index in class_index:
            np.random.shuffle(cls_index)
            splt = np.array_split(cls_index, folds)
            for i in range(folds):
                index_fold[i].extend(splt[i])
        #Genrating sets for each fold (Training and Testing)
        for i in range(folds):
            test_idx = np.array(index_fold[i])
            train_idx = np.array([idx for idx in index if idx not in test_idx])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            datasets.append((X_train, y_train, X_test, y_test))
    else:
        #Remaining unstratified split requires random shuffling then split into
        np.random.shuffle(index)
        splt = np.array_split(index, folds)
        for i in range(folds):
            test_idx = splt[i]
            train_idx = np.hstack([splt[j] for j in range(folds) if j != i])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            datasets.append((X_train, y_train, X_test, y_test))

    return datasets

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    #Calculate Accuracy = count of correctly identified / total
    correct_count = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            correct_count += 1

    accuracy = correct_count / len(y)
    return accuracy

def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    #count True Positives and False Positives
    true_positive = np.sum((y == 1) & (y_hat == 1))
    false_positive = np.sum((y == 0) & (y_hat == 1))
    # Calculate Precision
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)
    return precision

def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    # Counts True Positives and False Negatives
    true_positive = np.sum((y == 1) & (y_hat == 1))
    false_negative = np.sum((y == 1) & (y_hat == 0))
    #Calculate Recall
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive+false_negative)
    return recall

def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    #Store thresolds 
    thrsld = np.sort(np.unique(p_y_hat))[::-1]
    tpr_fpr_pairs = []
    
    for thr in thrsld:
        y_hat = (p_y_hat >= thr).astype(int)
        #Calculate confusion matrix
        true_positive = np.sum((y == 1) & (y_hat == 1))
        false_positive = np.sum((y == 0) & (y_hat == 1))
        false_negative = np.sum((y == 1) & (y_hat == 0))
        true_negative = np.sum((y == 0) & (y_hat == 0))
        #calculate the True Positive Rate and False Positive Rate
        if (true_positive + false_negative) > 0:
            tpr = true_positive / (true_positive + false_negative)
        else:
            tpr = 0
        if (false_positive + true_negative) > 0: 
            fpr = false_positive / (false_positive + true_negative)
        else:
            fpr = 0

        tpr_fpr_pairs.append((fpr, tpr))
    
    return tpr_fpr_pairs

def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    #Find the ROC Pairs
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    auc_value = 0.0
    for i in range(1, len(roc_pairs)):
        x1, y1 = roc_pairs[i - 1]
        x2, y2 = roc_pairs[i]
        diff = x2 - x1
        sum = y1 + y2
        # Calculate the Area under curve
        auc_value += diff * sum / 2.0  
    return auc_value
