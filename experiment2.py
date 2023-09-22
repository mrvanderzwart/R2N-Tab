import json
import random

import numpy as np
import torch
from datasets.dataset import *
from R2Ntab import R2Ntab


def transform_dataset_dummies(
    name, method="ordinal", negations=False, labels="ordinal", n_dummies=20
):
    """
    Copy of datasets/dataset.py/transform_dataset but with ability to add dummy features
    """

    METHOD = ["origin", "ordinal", "onehot", "onehot-compare"]
    LABELS = ["ordinal", "binary", "onehot"]
    if method not in METHOD:
        raise ValueError(f"method={method} is not a valid option. The options are {METHOD}")
    if labels not in LABELS:
        raise ValueError(f"labels={labels} is not a valid option. The options are {LABELS}")

    table_X, table_Y, categorical_cols, numerical_cols = predefined_dataset(name, binary_y=labels == "binary")

    for i in range(n_dummies):
        table_X[f"dummy{i}"] = random.choices([1, 2, 3], k=len(table_X))

    if categorical_cols is None:
        categorical_cols = list(table_X.columns[(table_X.dtypes == np.dtype("O")).to_numpy().nonzero()[0]])
    if numerical_cols is None:
        numerical_cols = [
            col
            for col in table_X.columns
            if col not in categorical_cols
            and np.unique(table_X[col].to_numpy()).shape[0] > 5
        ]
        categorical_cols = [col for col in table_X.columns if col not in numerical_cols]

    if len(categorical_cols) != 0:
        imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        table_X[categorical_cols] = imp_cat.fit_transform(table_X[categorical_cols])
    if len(numerical_cols) != 0:
        imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
        table_X[numerical_cols] = imp_num.fit_transform(table_X[numerical_cols])

    if np.nan in table_X or np.nan in table_Y:
        raise ValueError("Dataset should not have nan value!")

    X = table_X.copy()

    col_categories = []
    if method in ["origin", "ordinal"] and len(categorical_cols) != 0:
        ord_enc = OrdinalEncoder()
        X[categorical_cols] = ord_enc.fit_transform(X[categorical_cols])
        col_categories = {
            col: list(categories)
            for col, categories in zip(categorical_cols, ord_enc.categories_)
        }

    col_intervals = []
    if method in ["ordinal", "onehot"] and len(numerical_cols) != 0:
        kbin_dis = KBinsDiscretizer(encode="ordinal", strategy="kmeans")
        X[numerical_cols] = kbin_dis.fit_transform(X[numerical_cols])
        col_intervals = {
            col: [
                f"({intervals[i]:.2f}, {intervals[i+1]:.2f})"
                for i in range(len(intervals) - 1)
            ]
            for col, intervals in zip(numerical_cols, kbin_dis.bin_edges_)
        }

        if method in ["onehot"]:
            for col in numerical_cols:
                X[col] = np.array(col_intervals[col]).astype("object")[
                    X[col].astype(int)
                ]

    if method in ["onehot", "onehot-compare"]:
        fb = FeatureBinarizer(colCateg=categorical_cols, negations=negations)
        X = fb.fit_transform(X)

    if method in ["origin"]:
        X_headers = [column for column in X.columns]
    if method in ["ordinal"]:
        X_headers = {
            col: col_categories[col] if col in col_categories else col_intervals[col]
            for col in table_X.columns
        }
    else:
        X_headers = ["".join(map(str, column)) for column in X.columns]

    if method not in ["origin"]:
        X = X.astype(int)

    le = LabelEncoder()
    Y = le.fit_transform(table_Y).astype(int)
    Y_headers = le.classes_
    if labels == "onehot":
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)

    return X, Y, X_headers, Y_headers


def run():
    runs = 10
    rates = [1e-2, 1e-4, 1e-6]
    for name in ["adult", "heloc", "house", "magic"]:
        accuracies = {}
        dummies = {}
        for rate in rates:
            accuracies[rate] = []
            dummies[rate] = []

        print("dataset:", name)
        for run in range(runs):
            print("  run:", run + 1)
            X, Y, X_headers, Y_headers = transform_dataset_dummies(
                name, method="onehot-compare", negations=False, labels="binary"
            )
            datasets = kfold_dataset(X, Y, shuffle=1)
            X_train, X_test, Y_train, Y_test = datasets[0]
            train_set = torch.utils.data.TensorDataset(
                torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train)
            )
            test_set = torch.utils.data.TensorDataset(
                torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test)
            )

            for rate in rates:
                net = R2Ntab(train_set[:][0].size(1), 50, 1)
                dummies = net.fit(
                    train_set,
                    test_set=test_set,
                    device="cuda",
                    epochs=1000,
                    batch_size=400,
                    cancel_lam=rate,
                )
                accuracy = net.predict(X_test, Y_test)
                accuracies[rate].append(accuracy)
                dummies[rate].append(dummies)

        for rate in rates:
            dummies[rate] = np.array(dummies[rate])
            dummies[rate] = dummies[rate].mean(axis=0)

        with open(f"exp2-dummies-{name}.json", "w") as file:
            json.dump(dummies, file)

        with open(f"exp2-accuracies-{name}.json", "w") as file:
            json.dump(accuracies, file)
            
            
def plot():
    pass