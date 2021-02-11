# EQDiscovery
# Overview
Harnessing data to discover the underlying governing laws or equations that describe the behavior of complex physical systems can significantly advance our modeling, simulation and understanding of such systems in various science and engineering disciplines. This work introduces a novel physics-informed deep learning framework to discover governing partial differential equations (PDEs) from scarce and noisy data for nonlinear spatiotemporal systems. In particular, this symbolic reasoning approach seamlessly integrates the strengths of deep neural networks for rich representation learning, physics embedding, automatic differentiation and sparse regression to (1) approximate the solution of system variables, (2) compute essential derivatives, as well as (3) identify the key derivative terms and parameters that form the structure and explicit expression of the PDEs. The efficacy and robustness of this method are demonstrated, both numerically and experimentally, on discovering a variety of PDE systems with different levels of data scarcity and noise accounting for different initial/boundary conditions. The resulting computational framework shows the potential for closed-form model discovery in practical applications where large and accurate datasets are intractable to capture.

This reposititory provides supplementary codes and data for the following work 

- Chen, Zhao, Yang Liu, and Hao Sun. "Physics-informed learning of governing equations from scarce data." arXiv preprint arXiv:2005.03448 (2020).

# System Requirements
## Hardware requirements
This work was done on on 
- a workstation with 28 Intel Core i9-7940X CPUs and 2 NVIDIA GTX 1080Ti GPU cards; and
- an Nvidia DGX with four Tesla V100 GPU of 32 GB memory.

## Software requirements
### OS Requirements
The package has been tested on the following systems:
- Windows 10 Pro
- Linux: Ubuntu 18.04.3 LTS
### Software dependencies & Installation guide
This work was implemented on Tensorflow 1.15 in Python empowered by GPU. MATLAB was used for visual enhancement. We recommend installing Python via Anaconda first, then installing related Python packages through Anaconda and installing GPU operation packages. The list of softwares this work has been tested on is
- Anaconda
- Spyder
- Python 3.7
- Tensorflow 1.15
- NVIDIA® GPU drivers
- CUDA® Toolkit
- cuDNN SDK 8.0.4
- numpy
- matplotlib
- scipy
- matplotlib 
- mpl_toolkits
- pyDOE 
- tqdm

# How to run our cases?
To try out our simulations, please visit "Examples" directory and we recommend running corresponding Python file in Spyder IDE. All datas are included in the directory.

Expected outcome will be some error metrics showing the quality of discovery, some simple figures illustrating predictive system responses, discovered governing equations and training loss histories for diagnostics purpose. The predicted data is also saved for MATLAB plotting purpose. 

The expected running time on our workstation ranges from 20 minutes to several hours, depending on the complexity of the system, the network and the amount of data.

To use the code for your purpose, you can modify sections for loading data, defining neural network, defining candidate library, etc. The codes have been carefully structured and commented.


