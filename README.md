# A Reproducibility Study Of Transtab: Learning Transferable Tabular Transformer Across Tables
Created by <a href="https://github.com/albertotamajo" target="_blank">Alberto Tamajo</a>, <a href="https://github.com/JakubDylag" target="_blank">Jakub Dylag</a>, <a href="" target="_blank">Alessandro Nerla</a> and <a href="" target="_blank">Laurin Lanz</a>.

<p align="center">
  <img src="https://github.com/COMP6258-Reproducibility-Challenge/TransTab-Reproducibility/blob/main/transtab.png", width="60%", height="60%"/>
  <p align="center">Transtab architecture. Figure from original paper</p>
</p>

## Introduction
In this work, we verify the reproducibility of <a href="https://arxiv.org/abs/2205.09328" target="_blank">TransTab: Learning Transferable Tabular Transformers Across Tables</a> as part of COMP6258 module.

The ubiquity of tabular data in machine learning led Wang & Sun (2022) to introduce a versatile tabular learning framework, Transferable Tabular Transformer (TransTab), capable of modelling variable-column tables. Furthermore, they proposed a novel technique that enables supervised or self-supervised pretraining on multiple tables, as well as finetuning on the target dataset. Given the potential impact of their work, we aim to verify their claims by trying to reproduce their results. Specifically, we try to corroborate the ’methods’ and ’results’ reproducibility of their paper.

## How to run the reproducibility experiments
### Clone this project
The first step is to clone this project:
```bash
git clone https://github.com/COMP6258-Reproducibility-Challenge/TransTab-Reproducibility.git
cd Transtab-Reproducibility/
```
### CONDA environment
```
conda env create -f environment.yml
conda activate TranstabReproducibility
```

## Code
We verified Transtab's reproducibility by leveraging <a href="https://github.com/RyanWangZf/transtab">Transtab's code package</a> v. `0.0.2`. On the 05/04/23 v. `0.0.5` was released. In the following, we list our code and the one retrieved from the original repository.
- Our code:
  - `Rankings.ipynb`
  - `Transtab.ipynb`
  - `incremental_learning.py`
  - `supervised_learning.py`
  - `supervised_selfsupervised_pretrain_finetuning.py`
  - `transfer_learning.py`
  - `zeroshot_learning.py`
- Original code:
  - `constants.py`
  - `dataset.py`
  - `evaluator.py`
  - `load.py`
  - `modeling_transtab.py`
  - `trainer.py`
  - `trainer_utils.py`
  - `transtab.py`

## Experiment results
Our experiment results are saved in this repository as pickle files:
- **Supervised learning:** `supervised_learning.pickle`
- **Feature Incremental Learning:** `incremental_learning.pickle`
- **Transfer Learning:** `transfer_learning.pickle`
- **Zero-Shot Learning:** `zeroshot_learning.pickle`
- **Supervised and Self-supervised Pretraining:** `across_table_pretraining_finetuning.pickle`
