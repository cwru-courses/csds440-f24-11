import argparse
import os
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

import util

class Node:
    def __init__(self, feature_index=None, threshold=None, children=None, value=None, is_leaf=False):
        """
        Initializes a node.

        Args:
            feature_index: Index of the feature to split on.
            threshold: Threshold value for continuous features.
            children: Dictionary of child nodes for nominal features.
            value: Class label for leaf nodes.
            is_leaf: True if node is a leaf.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.children = children  # For nominal features, dict of {value: child_node}
        self.left = None  # For continuous features, left child (<= threshold)
        self.right = None  # For continuous features, right child (> threshold)
        self.value = value
        self.is_leaf = is_leaf

# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.
class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature], max_depth=0, criterion='information_gain', min_gain_threshold=0):
        """
        Initialize the decision tree.

        Args:
            schema: List of Features.
            max_depth: Maximum depth of the tree (0 means no limit).
            criterion: Split criterion ('information_gain' or 'gain_ratio').
            min_gain_threshold: Minimum gain required to continue splitting.
        """
        self._schema = schema
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_gain_threshold = min_gain_threshold
        self.root = None


    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        # In Java, it is best practice to LBYL (Look Before You Leap), i.e. check to see if code will throw an exception
        # BEFORE running it. In Python, the dominant paradigm is EAFP (Easier to Ask Forgiveness than Permission), where
        # try/except blocks (like try/catch blocks) are commonly used to catch expected exceptions and deal with them.
        self.root = self._build_tree(X, y, depth=1)

    def _build_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node()
        node.value = predicted_class

        if (self.max_depth != 0 and depth > self.max_depth) or len(y) < 2:
            node.is_leaf = True
            return node

        feature_index, threshold, splits, best_gain = self._determine_best_split(X, y)
        if feature_index is None:
        #or best_gain < self.min_gain_threshold:
            node.is_leaf = True
            return node

        node.feature_index = feature_index
        feature = self._schema[feature_index]

        if threshold is not None:
            node.threshold = threshold
            left_indices = splits['left']
            right_indices = splits['right']
            node.left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
            node.right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        else:
            node.children = {}
            for val, indices in splits.items():
                child = self._build_tree(X[indices], y[indices], depth + 1)
                node.children[val] = child
        return node

    def _determine_best_split(self, X, y):
        num_features = X.shape[1]
        best_gain = -np.inf
        best_feature_index = None
        best_threshold = None
        best_splits = None
        current_entropy = util.entropy(y)

        for feature_index in range(num_features):
            feature = self._schema[feature_index]
            X_feature = X[:, feature_index]

            if feature.ftype == FeatureType.NOMINAL:
                # Nominal feature
                values = np.unique(X_feature)
                splits = {}
                for val in values:
                    splits[val] = np.where(X_feature == val)[0]

                criterion_value, gain = self._determine_split_criterion(
                    y, splits, current_entropy, X_feature=X_feature
                )

                if criterion_value > best_gain:
                    best_gain = criterion_value
                    best_feature_index = feature_index
                    best_threshold = None
                    best_splits = splits
            else:
                # Continuous feature
                X_feature_sorted, y_sorted = self._sort_feature(X_feature, y)
                thresholds = self._find_thresholds(X_feature_sorted, y_sorted)

                if len(thresholds) == 0:
                    continue

                for threshold in thresholds:
                    left_indices = np.where(X_feature <= threshold)[0]
                    right_indices = np.where(X_feature > threshold)[0]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    splits = {'left': left_indices, 'right': right_indices}

                    criterion_value, gain = self._determine_split_criterion(
                        y, splits, current_entropy, X_feature=X_feature, threshold=threshold
                    )

                    if criterion_value > best_gain:
                        best_gain = criterion_value
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_splits = splits

        return best_feature_index, best_threshold, best_splits, best_gain

    def _sort_feature(self, X_feature, y):
        sorted_indices = np.argsort(X_feature)
        X_feature_sorted = X_feature[sorted_indices]
        y_sorted = y[sorted_indices]
        return X_feature_sorted, y_sorted

    def _find_thresholds(self, X_feature_sorted, y_sorted):
        # Find label changes in sorted labels
        label_changes = np.where(y_sorted[:-1] != y_sorted[1:])[0]
        # Calculate thresholds as midpoints between feature values where labels change
        thresholds = (X_feature_sorted[label_changes] + X_feature_sorted[label_changes + 1]) / 2.0
        return np.unique(thresholds)    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions
    
    def _predict_sample(self, x):
        node = self.root
        while not node.is_leaf:
            feature_index = node.feature_index
            feature_value = x[feature_index]
            feature = self._schema[feature_index]
            if feature.ftype == FeatureType.NOMINAL:
                if feature_value in node.children:
                    node = node.children[feature_value]
                else:
                    # Handle unseen feature value, use majority class
                    return node.value
            else:
                if feature_value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.value

    # In Python, instead of getters and setters we have properties: docs.python.org/3/library/functions.html#property
    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

# In the DecisionTree class (dtree.py)

    def _determine_split_criterion(self, y, splits, current_entropy, X_feature=None, threshold=None):
        """
        Determine the split criterion value (information gain or gain ratio) for a given split.

        Args:
            y: Labels of the current node.
            splits: Dictionary of splits with indices.
            current_entropy: Entropy of the current node.
            X_feature: Feature values (required for gain ratio calculation).
            threshold: Threshold value for continuous features (required for gain ratio calculation).

        Returns:
            Tuple containing criterion value and information gain.
        """
        # Calculate information gain
        gain = util.information_gain(y, splits, current_entropy)

        # Determine criterion value based on the selected criterion
        if self.criterion == 'gain_ratio':
            # Calculate the gain ratio
            criterion_value = util.gain_ratio(y, splits, current_entropy, X_feature, threshold)
        else:
            # Information gain
            criterion_value = gain

        return criterion_value, gain


    def get_size(self):
        return self._get_size(self.root)

    def _get_size(self, node):
        if node.is_leaf:
            return 1
        if node.threshold is not None:
            # Continuous feature
            return 1 + self._get_size(node.left) + self._get_size(node.right)
        else:
            # Nominal feature
            size = 1
            for child in node.children.values():
                size += self._get_size(child)
            return size

    def get_max_depth(self):
        return self._get_max_depth(self.root)

    def _get_max_depth(self, node):
        if node.is_leaf:
            return 1
        depths = []
        if node.threshold is not None:
            depths.append(self._get_max_depth(node.left))
            depths.append(self._get_max_depth(node.right))
        else:
            for child in node.children.values():
                depths.append(self._get_max_depth(child))
        return 1 + max(depths)

    def get_first_test(self):
        if self.root.is_leaf:
            return 'None'
        feature_index = self.root.feature_index
        feature = self._schema[feature_index]
        if feature.ftype == FeatureType.NOMINAL:
            return feature.name
        else:
            return f"{feature.name} <= {self.root.threshold:.4f}"

    def get_second_tests(self):
        if self.root.is_leaf:
            return ['None']
        second_tests = []
        if self.root.threshold is not None:
            # Continuous feature
            if not self.root.left.is_leaf:
                feature_index = self.root.left.feature_index
                feature = self._schema[feature_index]
                if feature.ftype == FeatureType.NOMINAL:
                    second_tests.append(feature.name)
                else:
                    second_tests.append(f"{feature.name} <= {self.root.left.threshold:.4f}")
            if not self.root.right.is_leaf:
                feature_index = self.root.right.feature_index
                feature = self._schema[feature_index]
                if feature.ftype == FeatureType.NOMINAL:
                    second_tests.append(feature.name)
                else:
                    second_tests.append(f"{feature.name} <= {self.root.right.threshold:.4f}")
        else:
            # Nominal feature
            for child in self.root.children.values():
                if not child.is_leaf:
                    feature_index = child.feature_index
                    feature = self._schema[feature_index]
                    if feature.ftype == FeatureType.NOMINAL:
                        second_tests.append(feature.name)
                    else:
                        second_tests.append(f"{feature.name} <= {child.threshold:.4f}")
        return second_tests if second_tests else ['None']



def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    size = dtree.get_size()
    max_depth = dtree.get_max_depth()
    first_test = dtree.get_first_test()
    second_tests = dtree.get_second_tests()
    second_tests_str = ', '.join(second_tests)
    print(f'Accuracy: {acc:.2f}')
    print('Size:', size)
    print('Maximum Depth:', max_depth)
    print('First Test:', first_test)
    print('Second Tests:', second_tests_str)


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    # Step 1: Remove "image id" feature if it exists in the schema
    feature_names = [feature.name for feature in schema]
    if "image_id" in feature_names:
        image_id_index = feature_names.index("image_id")  # Get index of "image id" feature
        X = np.delete(X, image_id_index, axis=1)  # Remove the "image id" column from the dataset
        del schema[image_id_index]  # Remove the "image id" feature from the schema

    # Convert data types for efficiency (optional, can uncomment if needed)
    # X = X.astype(np.float32)
    # y = y.astype(np.int8)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = [(X, y, X, y)]

    criterion = 'information_gain' if information_gain else 'gain_ratio'

    accuracies = []

    for X_train, y_train, X_test, y_test in datasets:
        decision_tree = DecisionTree(schema, max_depth=tree_depth_limit, criterion=criterion)
        decision_tree.fit(X_train, y_train)
        evaluate_and_print_metrics(decision_tree, X_test, y_test)
        y_hat = decision_tree.predict(X_test)
        acc = util.accuracy(y_test, y_hat)
        accuracies.append(acc)

    if use_cross_validation:
        avg_acc = np.mean(accuracies)
        print(f'Average Accuracy over {len(datasets)} folds: {avg_acc:.2f}')

    # Return the last decision tree for analysis (optional, depending on use case)


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here.
    """
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)
