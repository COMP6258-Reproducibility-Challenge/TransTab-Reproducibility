import dataset
import evaluator
import transtab
import torch
import random
import os
from itertools import product
import pickle

seeds = [222, 41, 273, 522, 408, 796, 606, 706, 945, 555]
if not seeds:
    for i in range(10):
        seeds.append(random.randint(0, 1000))

t = None
d = {}
path = "supervised_learning.pickle"
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
    torch.manual_seed(seed)
    # Load dataset by specifying dataset name
    if previous_set is not set:
        allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = dataset.load_data(set)
    # Build classifier
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)
    # Train model on training dataset
    transtab.train(model, trainset, valset, lr=lr, batch_size=batch_size, num_epoch=epochs, patience=patience)
    # Compute predictions on test dataset
    y_pred = evaluator.predict(model, testset[0])
    # Compute AUROC score
    auroc_score = evaluator.evaluate(y_pred, testset[1])
    if len(auroc_score) == 1:
        d[(set, seed, lr, batch_size)] = auroc_score[0]
        with open(path, 'wb') as f:
            pickle.dump(((set, seed, lr, batch_size), d), f)
        previous_set = set
    else:
        raise Exception