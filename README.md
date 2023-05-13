# A Reproducibility Study Of Transtab: Learning Transferable Tabular Transformer Across Tables
Created by <a href="https://github.com/albertotamajo" target="_blank">Alberto Tamajo</a>, <a href="https://github.com/JakubDylag" target="_blank">Jakub Dylag</a>, <a href="" target="_blank">Alessandro Nerla</a> and <a href="" target="_blank">Laurin Lanz</a>.

<p align="center">
  <img src="https://github.com/COMP6258-Reproducibility-Challenge/TransTab-Reproducibility/blob/main/transtab.png", width="60%", height="60%"/>
  <p align="center">Transtab architecture. Figure from original paper</p>
</p>

### Introduction
In this work, we verify the reproducibility of <a href="https://arxiv.org/abs/2205.09328" target="_blank">TransTab: Learning Transferable Tabular Transformers Across Tables</a> as part of COMP6258 module.

The ubiquity of tabular data in machine learning led Wang & Sun (2022) to introduce a versatile tabular learning framework, Transferable Tabular Transformer (TransTab), capable of modelling variable-column tables. Furthermore, they proposed a novel technique that enables supervised or self-supervised pretraining on multiple tables, as well as finetuning on the target dataset. Given the potential impact of their work, we aim to verify their claims by trying to reproduce their results. Specifically, we try to corroborate the ’methods’ and ’results’ reproducibility of their paper.

### Code
We verified Transtab's reproducibility by leveraging <a href="https://github.com/RyanWangZf/transtab">Transtab's code package</a> v. `0.0.2`

### Experiment results
Our experiment results are saved in this repository as pickle files:
- **Supervised learning:** `supervised_learning.pickle`
- **Feature Incremental Learning:** `incremental_learning.pickle`
- **Transfer Learning:** `transfer_learning.pickle`
- **Zero-Shot Learning:** `zeroshot_learning.pickle`
- **Supervised and Self-supervised Pretraining:** `across_table_pretraining_finetuning.pickle`
