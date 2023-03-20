import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from build_decision_tree import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset source", type=str, default="dataset1")
args = parser.parse_args()

TEST_NUM = 100


def traverse_tree(instance, node):
    if node.partition_feature_idx == None:
        label_set = np.unique(node.dataset.y_train)
        y_occurance = np.array([np.sum(node.dataset.y_train == label) for label in label_set])
        return label_set[np.argmax(y_occurance)]
    else:
        if instance[node.partition_feature_idx] < node.partition_feature_val:
            node = node.children[0]
            return traverse_tree(instance, node)
        else:
            node = node.children[1]
            return traverse_tree(instance, node)


def predict_target_value(instance, tree_node):
    return traverse_tree(instance, tree_node)


def report(predictions, y_test):
    print('Accuracy: %s' % accuracy_score(y_test, predictions))
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, predictions))
    print()
    print('Classification Report:')
    print(classification_report(y_test, predictions, zero_division=1))


def load_dataset():

    X_train = pd.read_excel(f"./{args.dataset}/X_train_processed.xlsx").to_numpy()
    y_train = pd.read_excel(f"./{args.dataset}/y_train.xlsx").to_numpy()
    return X_train, y_train


if __name__ == "__main__":
    X_train, y_train = load_dataset()

    # X_train = data_processing(X_train)

    X_test = X_train[:TEST_NUM, :]
    y_test = y_train[:TEST_NUM, :]
    if args.dataset == "dataset1":
        with open("./model/decision_tree_model1.pkl", "rb") as f:
            root = pickle.load(f)
    elif args.dataset == "dataset2":
        with open("./model/decision_tree_model2.pkl", "rb") as f:
            root = pickle.load(f)

    y_predict = np.array([predict_target_value(X_test[instance, :], root) for instance in range(X_test.shape[0])])
    y_predict = y_predict.reshape((y_predict.shape[0], -1))

    # for pre, fill, node in anytree.RenderTree(root):
    #     print("%s%s" % (pre, node.name))

    # print(y_predict.T)
    # print(y_test.T)
    # print(np.sum(np.equal(y_predict, y_test)))
    # print(np.sum(np.equal(y_predict, y_test))/y_test.shape[0])
    report(y_predict, y_test)
