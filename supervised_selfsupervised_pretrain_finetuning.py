import dataset
import evaluator
import transtab
import random
import os
from itertools import product
import pickle

def dictionary_update(dic1, dic2):
    del dic1["projection_head.dense.weight"]
    for k in dic2.keys():
        if k not in dic1.keys():
            dic1[k] = dic2[k]
    return dic1

seeds = [222, 41, 273, 522, 408, 796, 606, 706, 945, 555]
if not seeds:
    for i in range(10):
        seeds.append(random.randint(0, 1000))

t = None
d = {}
path = "supervised_selfsupervised_pretrain_finetuning.pickle"
# Load tuple last training - dictionary
if os.path.exists(path):
    with open(path, 'rb') as f:
        t, d = pickle.load(f)
datasets = ['credit-g', 'credit-approval', 'dresses-sales', 'cylinder-bands']
lrs = [1e-4, 5e-5, 2e-5]
batch_sizes = [64, 16, 128]
pre_training_epochs = 25
epochs = 100
patience = 10
trainings = list(product(datasets, seeds, lrs, batch_sizes))
if t is not None:
    trainings = trainings[trainings.index(t)+1:]
previous_set = None

for (set, seed, lr, batch_size) in trainings:
    auroc_scores = []
    # Load other datasets
    if previous_set is not set:
        allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = dataset.load_data([dataset for dataset in datasets if dataset != set])
        allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = dataset.load_data(set)
    # Build contrastive learner not supervised
    model, collate_fn = transtab.build_contrastive_learner(cat_cols, num_cols, bin_cols)
    # Train model on training dataset
    transtab.train(model, trainset, lr=lr, batch_size=batch_size, num_epoch=pre_training_epochs, patience=patience, collate_fn=collate_fn)
    dic1 = model.state_dict()
    model = transtab.build_classifier(cat_cols2, num_cols2, bin_cols2)
    dic2 = model.state_dict()
    model.load_state_dict(dictionary_update(dic1, dic2))
    transtab.train(model, trainset2, valset2, lr=lr, batch_size=batch_size, num_epoch=epochs, patience=patience)
    # Compute predictions on test dataset
    y_pred = evaluator.predict(model, testset2[0])
    # Compute AUROC score
    auroc_score = evaluator.evaluate(y_pred, testset2[1])
    if len(auroc_score) == 1:
        auroc_scores.append(auroc_score[0])
    else:
        raise Exception
    # Build contrastive learner supervised
    model, collate_fn = transtab.build_contrastive_learner(cat_cols, num_cols, bin_cols, supervised=True)
    # Train model on training dataset
    transtab.train(model, trainset, lr=lr, batch_size=batch_size, num_epoch=pre_training_epochs,
                   patience=patience, collate_fn=collate_fn)
    dic1 = model.state_dict()
    model = transtab.build_classifier(cat_cols2, num_cols2, bin_cols2)
    dic2 = model.state_dict()
    model.load_state_dict(dictionary_update(dic1, dic2))
    transtab.train(model, trainset2, valset2, lr=lr, batch_size=batch_size, num_epoch=epochs, patience=patience)
    # Compute predictions on test dataset
    y_pred = evaluator.predict(model, testset2[0])
    # Compute AUROC score
    auroc_score = evaluator.evaluate(y_pred, testset2[1])
    if len(auroc_score) == 1:
        auroc_scores.append(auroc_score[0])
    else:
        raise Exception
    d[(set, seed, lr, batch_size)] = auroc_scores
    with open(path, 'wb') as f:
        pickle.dump(((set, seed, lr, batch_size), d), f)
    previous_set = set