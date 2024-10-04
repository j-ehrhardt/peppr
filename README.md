[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/release/python-380/)
[![Mamba](https://img.shields.io/badge/Mamba-1.5.8-green)](https://mamba.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# Peppr - Learning Process Steps as Dynamical Systems

This is the complementary repository to the paper "Learning Process Steps as Dynamical Systems for a Sub-Symbolic Approach of Process Planning in Cyber-Physical Production Systems" accepted at the HYDRA Workshop at the European Conference on Artificial Intelligence (ECAI) 2023. 

## Content

The repo contains an implementation of the Process Step Representation network architecture (peppr), and all supplementary material and datasets to reproduce the results of the paper.

Approaches in AI planning for Cyber-Physical Production Systems (CPPS) are mainly symbolic and depend on comprehensive formalizations of system domains and planning problems.
Handcrafting such formalizations requires detailed knowledge of the formalization language, of the CPPS, and is overall considered difficult, tedious, and error-prone.
Within this paper, we suggest a sub-symbolic approach for solving planning problems in CPPS. 
Our approach relies on neural networks that learn the dynamical behavior of individual process steps from global time-series observations of the CPPS and are embedded in a superordinate network architecture. 
In this context, we present the peppr architecture, a novel neural network architecture, which can learn the behavior of individual or multiple dynamical systems from global time-series observations.
We evaluate peppr on real datasets from physical and biochemical CPPS, as well as artificial datasets from electrical and mathematical domains. 
Our model outperforms baseline models like multilayer perceptrons and variational autoencoders and can be considered as a first step towards a sub-symbolic approach for planning in CPPS. 

You can access the paper [here](https://www.researchgate.net/publication/XXXX)


## Requirements

Python and venv requirements cf. `peppr.yml`.
To install the venv with Anaconda: (1) change the path in `peppr.yml` to the directory you want to install the venv. (2) Open the file location in your terminal and type `mamba env create -f peppr.yml`.


## Data
We evaluated our model on four different datatset. The datasets can be accessed under the following links. 

| dataset                   | description                                                                | link                                                                            |
|---------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| polynom                   | Sinus function modulated by a polynomial functions of different degrees    | https://github.com/j-ehrhardt/basic-datagenerator                               |
| ode                       | Data from a PT-1, PT-2, and PT-3 unit                                      | https://github.com/j-ehrhardt/ode-ml-datasets                                   |
| BeRfiPl                   | A benchmark about the physical behavior of Cyber-Physical Process Systems. | https://github.com/j-ehrhardt/benchmark-for-diagnosis-reconf-planning           |
| IndPenSim                 | A dataset of a Penecillin fermentation process                             | http://www.industrialpenicillinsimulation.com/                                  |

## Models

All necessary code to run and evaluate the models is in the `./model` directory. 

To train the models, save the datasets into the equivalent directories in the `./data/dataset-name` directory. 
Modify the `experiments.json` file in the `./model` directory, and run the `main.py`. 

You can create a gridsearch by defining the parameter ranges of the `experimental_setup.py` file, creating a new `experiments.json` file that includes all experiments of the gridsearch.

### Modules

The `data_module.py` file contains all necessary dataimport and dataloaders. 

The `eval_module.py` file contains the evaluation metrics. 

The `train_module.py` file contains the training procedures for all network architectures. 

The `module.py` file contains the network architecture components, the networks are defined in the `nets.py` file. 

The `ultils.py` and `vis_utils.py` files contain utility and visualization functions. 



## Citation

When using the code from this paper, please cite: 
```
@ARTICLE{Ehrhardt2024-ke,
  title     = "Learning process steps as dynamical systems for a sub-symbolic
               approach of process planning in {Cyber-Physical} Production
               Systems",
  author    = "Ehrhardt, Jonas and Heesch, Ren{\'e} and Niggemann, Oliver",
  journal   = "Commun. Comput. Inf. Sci.",
  publisher = "Springer Nature Switzerland",
  pages     = "332--345",
  year      =  2024,
}

```


## LICENSE

Licensed under MIT license - cf. LICENSE

