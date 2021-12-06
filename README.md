# Concept Learning for CPS (Code for Paper)

This repository contains the code for the paper [Learning Physical Concepts in CPS: A Case Study with a Three-Tank System](https://arxiv.org/abs/2111.14151). All experiments and the figures contained in the paper should be reproducible.

This repo is structured as follows:

```
.
├── .gitignore                  
├── README.md                   --> this field :)
├── data                        --> dataset for all solutions
├── figures                     --> all paper figurers
├── setup.py                    --> for package installation
├── solution-1                  --> subfolder for solution
│   ├── constants.py            --> hyperparameters, constants, ...
│   ├── datagen                 --> code for data generation
│   ├── scripts                 --> scripts to run the experiment
│   │   ├── create_figures.py
│   │   ├── generate_data.py
│   │   └── run_training.py
│   └── seq2seq-vae             --> model code
│       └── ...
├─ solution-2
│   └── ...
...

```

## Installation
To install the required packages, run the following from the project root of this repo:
```sh
conda create -n concept-learning-cps python=3.9
conda activate concep-learning-cps
pip install -e .
```

## Run the experiments
Each of the solutions has its own subdirectory in this repo and consists of three main steps:
1. Generate the data sets
3. Train the model
4. Generate result visualizations

The following commands will run those steps for solution 1. The other experiment can be reproduced accordingly.
From the project root and with the new environment activated:
```
ipython solution-1/scripts/generate_data.py
ipython solution-1/scripts/run_training.py
ipython solution-1/scripts/create_figures.py
```



