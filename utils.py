import numpy as np
import math


def entropy(data, num, continuous, threshold):
    h = 0
    if continuous:
        p1 = np.sum(data < threshold)/num
        p2 = np.sum(data >= threshold)/num
        if p1 != 0:
            h -= p1 * math.log2(p1)
        if p2 != 0:
            h -= p2 * math.log2(p2)
    else:
        label_set = np.unique(data)
        for label in label_set:
            if label in data:
                p = np.sum(data == label)/num
                h -= p * math.log2(p)
    # print(h)
    return h


def information_gain(X_data, y_data, feature_threshold):

    sub_y1 = y_data[np.where(X_data < feature_threshold)]
    sub_y2 = y_data[np.where(X_data >= feature_threshold)]

    instance_num = y_data.shape[0]
    y1_num = sub_y1.shape[0]
    y2_num = sub_y2.shape[0]

    rem = 0
    rem += y1_num/instance_num * entropy(sub_y1, y1_num, False, None)
    rem += y2_num/instance_num * entropy(sub_y2, y2_num, False, None)

    IG = entropy(y_data, instance_num, False, None) - rem
    return IG
