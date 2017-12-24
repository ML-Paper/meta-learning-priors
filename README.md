# Implementation of the Meta-PAC-Bayes algorithm in PyTorch



## Prerequisites

- Python 3.5+ 
- [PyTorch 0.3+ with CUDA](http://pytorch.org)
- NumPy and Matplotlib


## Reproducing experiments in the paper:

* Stochsastic_Meta_Learning/main_Meta_Bayes.py             - Learns a prior from the obsereved (meta-training) tasks and use it to learn new (meta-test) tasks.
* Toy_Examples/Toy_Main.py -  Toy example of 2D  estimation.

* Single_Task/main_TwoTaskTransfer_PermuteLabels - runs alternative tranfer methods between two tasks in the permuted-labels experiment.

* Stochsastic_Meta_Learning/Analyze_Prior.py - Analysis of the weight uncertainty ine each layer of the learned prior (run after creating a prior with main_Meta_Bayes.py)

* MAML/run_MAML_PermuteLabels.py - runs MAML algorithm in the permuted-labels experiment.

## Other experiments:

* Single_Task/main_single_standard.py         - Learn standard neural network in a single task.
* Single_Task/main_single_Bayes.py            - Learn stochastic neural network in a single task.

MAML code is based on: https://github.com/katerakelly/pytorch-maml
