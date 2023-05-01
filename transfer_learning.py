import math

import dataset
import evaluator
import transtab
import torch
import numpy as np
import random
import os
from itertools import product
import pickle


def categorise(split, cat_cols, num_cols, bin_cols):
    d_col = {'cat': None, 'num': None, 'bin': None}
    for col in split:
        if col in cat_cols:
            if d_col['cat'] is None:
                d_col['cat'] = []
            d_col['cat'].append(col)
        elif col in num_cols:
            if d_col['num'] is None:
                d_col['num'] = []
            d_col['num'].append(col)
        elif col in bin_cols:
            if d_col['bin'] is None:
                d_col['bin'] = []
            d_col['bin'].append(col)
        else:
            raise Exception
    return d_col


def to_empty(ls):
    if ls is None:
        return []
    return ls


seeds = [222, 41, 273, 522, 408, 796, 606, 706, 945, 555]
if not seeds:
    for i in range(10):
        seeds.append(random.randint(0, 1000))

t = None
d = {}
path = "transfer_learning.pickle"
# Load tuple last training - dictionary
if os.path.exists(path):
    with open(path, 'rb') as f:
        t, d = pickle.load(f)
datasets = ['credit-g', 'dresses-sales', 'adult', 'cylinder-bands', 'credit-approval', 'blastchar', '1995-income']
lrs = [1e-4, 5e-5, 2e-5]
batch_sizes = [64, 16, 128]
epochs = 100
patience = 10
trainings = list(product(datasets, seeds, lrs, batch_sizes))
if t is not None:
    trainings = trainings[trainings.index(t)+1:]
previous_set = None

for (set, seed, lr, batch_size) in trainings:
    # Load dataset by specifying dataset name
    if previous_set is not set:
        allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = dataset.load_data(set)
    cols = cat_cols + num_cols + bin_cols
    np.random.seed(123)
    np.random.shuffle(cols)
    set1_cols = cols[:math.floor(2/3*len(cols))]
    set2_cols = cols[math.ceil(len(cols)/3):]
    set1_cols_dict, set2_cols_dict = [categorise(x, cat_cols, num_cols, bin_cols) for x in [set1_cols, set2_cols]]
    auroc_scores = []
    for cols1_dict, cols1, cols2_dict, cols2 in [(set1_cols_dict, set1_cols, set2_cols_dict, set2_cols),
                                                 (set2_cols_dict, set2_cols, set1_cols_dict, set1_cols)]:
        torch.manual_seed(seed)
        model = transtab.build_classifier(categorical_columns=cols1_dict['cat'],
                                          numerical_columns=cols1_dict['num'],
                                          binary_columns=cols1_dict['bin'])
        # Pre-train model on training set 1
        transtab.train(model, (trainset[0][cols1], trainset[1]), (valset[0][cols1], valset[1]), lr=lr,
                       batch_size=batch_size, num_epoch=epochs, patience=patience)
        model.update(cols2_dict)
        # Fine-tune on training set 2
        transtab.train(model, (trainset[0][cols2], trainset[1]), (valset[0][cols2], valset[1]), lr=lr,
                       batch_size=batch_size, num_epoch=epochs, patience=patience)
        # Compute predictions on test set 2
        y_pred = evaluator.predict(model, testset[0][cols2])
        # Compute AUROC score
        auroc_score = evaluator.evaluate(y_pred, testset[1])
        if len(auroc_score) == 1:
            auroc_scores.append(auroc_score[0])
        else:
            raise Exception
    auroc_scores.reverse()
    d[(set, seed, lr, batch_size)] = auroc_scores
    with open(path, 'wb') as f:
        pickle.dump(((set, seed, lr, batch_size), d), f)
    previous_set = set
