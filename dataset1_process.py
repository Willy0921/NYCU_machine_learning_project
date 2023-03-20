import numpy as np
import pandas as pd
import math
import openpyxl
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--mode", help="Data mode", type=str, default=None)
args = parser.parse_args()


def handle_outliers(dataset, mean, std):
    upper = mean + 3*std
    lower = mean - 3*std
    upper_outlier_indices = np.where(dataset > upper)
    lower_outlier_indices = np.where(dataset < lower)
    dataset[upper_outlier_indices] = upper[upper_outlier_indices[1]]
    dataset[lower_outlier_indices] = lower[lower_outlier_indices[1]]
    return dataset


def handle_missing_data(dataset, median):
    nan_indices = np.where(np.isnan(dataset))
    dataset[nan_indices] = median[nan_indices[1]]
    return dataset


def drop_data_lost_instances(dataset):
    drop_threshold = math.floor(dataset.shape[1] * 3 / 5)
    delete_row = np.where(np.sum(np.isnan(dataset), axis=1) >= drop_threshold)[0]
    return np.delete(dataset, delete_row, axis=0)


def calculate_mean_std_median(dataset):
    mean = np.nanmean(dataset, axis=0)
    std = np.nanstd(dataset, axis=0)
    median = np.nanmedian(dataset, axis=0)
    return mean, std, median


def data_processing(X_data):
    mean, std, median = calculate_mean_std_median(X_data)

    X_data = drop_data_lost_instances(X_data)

    X_data = handle_missing_data(X_data, median)

    X_data = handle_outliers(X_data, mean, std)

    return X_data


def load_dataset():
    print("Load raw dataset...")
    X_data = pd.read_excel(f"./dataset1/X_{args.mode}.xlsx").to_numpy()

    return X_data


def save_processed_dataset(X_data):
    X_data = X_data.tolist()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "dataset1"
    title = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]
    for i in range(len(title)):
        ws.cell(row=1, column=i + 1).value = title[i]
    for instance in X_data:
        ws.append(instance)
    wb.save(filename=f"./dataset1/X_{args.mode}_processed.xlsx")
    print("Saving processed dataset done!")


if __name__ == "__main__":
    X_raw = load_dataset()
    X_data = data_processing(X_raw)
    save_processed_dataset(X_data)
