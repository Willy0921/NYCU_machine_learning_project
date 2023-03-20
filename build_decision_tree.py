import numpy as np
import anytree
import pickle
import pandas as pd
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-md", "--max_depth", help="Max depth", type=int, default=20)
parser.add_argument("-p", "--partition_threshold", help="Partition threshold", type=int, default=70)
parser.add_argument("-d", "--dataset", help="Dataset source", type=str, default="dataset1")
args = parser.parse_args()

TEST_NUM = 100


class DecisionTreeNode(anytree.NodeMixin):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.partition_feature_idx = None
        self.partition_feature_val = None


class PartitionDataset():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


def threshold_binary_search(sorted_X, sorted_y, lower, mid, upper):

    IG_mid = information_gain(sorted_X, sorted_y, sorted_X[mid])

    left_mean = math.floor((lower + mid - 1)/2)
    IG_left = information_gain(sorted_X, sorted_y, sorted_X[left_mean])

    right_mean = math.floor((mid + upper)/2)
    IG_right = information_gain(sorted_X, sorted_y, sorted_X[right_mean])

    if IG_left > IG_right and IG_left > IG_mid:
        upper = mid - 1
        mid = left_mean
        return threshold_binary_search(sorted_X, sorted_y, lower, mid, upper)
    elif IG_right > IG_left and IG_right > IG_mid:
        lower = mid + 1
        mid = right_mean
        return threshold_binary_search(sorted_X, sorted_y, lower, mid, upper)
    else:
        return sorted_X[mid], IG_mid


def find_partition_feature(dataset):

    feature_threshold = []
    feature_IG = []
    feature_GR = []
    X_data = dataset.X_train
    y_data = dataset.y_train
    for feature_idx in range(X_data.shape[1]):

        tmp_X = X_data[:, feature_idx]
        tmp_y = y_data

        sorted_indices = tmp_X.argsort()
        sorted_X = tmp_X[sorted_indices]
        sorted_X = sorted_X.reshape((len(sorted_X), -1))
        sorted_y = tmp_y[sorted_indices]

        lower = 0
        upper = sorted_X.shape[0] - 1
        mid = math.floor((lower + upper)/2)

        idx, val = threshold_binary_search(sorted_X, sorted_y, lower, mid, upper)

        feature_threshold.append(idx)
        feature_IG.append(val)

        ratio = entropy(X_data[:, feature_idx], X_data.shape[0], True, feature_threshold[feature_idx])
        if ratio == 0:
            ratio = 0.00001

        feature_GR.append(feature_IG[feature_idx]/ratio)

    feature_threshold = np.array(feature_threshold)
    feature_IG = np.array(feature_IG)
    feature_GR = np.array(feature_GR)

    selected_feature_idx = np.argmax(feature_GR)
    if feature_GR[selected_feature_idx] == 0:
        return None, None
    # selected_feature_idx = np.argmax(feature_IG)

    # print(f"feature_threshold: {feature_threshold}")
    # print(f"ffeature_IG: {feature_IG}")
    # print(f"feature_GR : {feature_GR }")

    print(f"feature_idx: {selected_feature_idx}  val: {feature_threshold[selected_feature_idx]}")
    return selected_feature_idx, feature_threshold[selected_feature_idx]


def create_new_node(node, slice_indices, feature_idx, val, depth):

    sub_X = node.dataset.X_train[slice_indices]
    sub_y = node.dataset.y_train[slice_indices[0]]

    if sub_X.shape[0] > 0:

        subset = PartitionDataset(sub_X, sub_y)
        child = DecisionTreeNode(str(sub_X.shape[0]) + " " + str(feature_idx) + " " + str(val), subset)
        child.parent = node
        if sub_X.shape[0] != node.dataset.X_train.shape[0]:
            grow_tree_branch(child, depth + 1)


def grow_tree_branch(node, depth):
    if len(np.unique(node.dataset.y_train)) == 1:
        return
    if depth > args.max_depth:
        return
    if node.dataset.X_train.shape[0] < args.partition_threshold:
        return

    feature_idx, val = find_partition_feature(node.dataset)
    if feature_idx == None and val == None:
        return
    node.partition_feature_idx = feature_idx
    node.partition_feature_val = val

    slice_indices = np.where(node.dataset.X_train[:, node.partition_feature_idx] < node.partition_feature_val)
    create_new_node(node, slice_indices, feature_idx, val, depth)

    slice_indices = np.where(node.dataset.X_train[:, node.partition_feature_idx] >= node.partition_feature_val)
    create_new_node(node, slice_indices, feature_idx, val, depth)


def load_dataset():

    X_train = pd.read_excel(f"./{args.dataset}/X_train_processed.xlsx").to_numpy()
    y_train = pd.read_excel(f"./{args.dataset}/y_train.xlsx").to_numpy()

    return X_train, y_train


if __name__ == "__main__":

    X_train, y_train = load_dataset()

    X_train = X_train[TEST_NUM:, :]
    y_train = y_train[TEST_NUM:, :]

    dataset = PartitionDataset(X_train, y_train)

    root = DecisionTreeNode("root", dataset)

    grow_tree_branch(root, 1)

    # for pre, fill, node in anytree.RenderTree(root):
    #     print("%s%s" % (pre, node.name))

    # print(f"tree height: {root.height}")
    if args.dataset == "dataset1":
        with open("./model/decision_tree_model1.pkl", "wb") as f:
            pickle.dump(root, f)
    elif args.dataset == "dataset2":
        with open("./model/decision_tree_model2.pkl", "wb") as f:
            pickle.dump(root, f)

    print("Building decision tree model done!")
