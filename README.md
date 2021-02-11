# EQDiscovery
# Overview
Harnessing data to discover the underlying governing laws or equations that describe the behavior of complex physical systems can significantly advance our modeling, simulation and understanding of such systems in various science and engineering disciplines. This work introduces a novel physics-informed deep learning framework to discover governing partial differential equations (PDEs) from scarce and noisy data for nonlinear spatiotemporal systems. In particular, this symbolic reasoning approach seamlessly integrates the strengths of deep neural networks for rich representation learning, physics embedding, automatic differentiation and sparse regression to (1) approximate the solution of system variables, (2) compute essential derivatives, as well as (3) identify the key derivative terms and parameters that form the structure and explicit expression of the PDEs. The efficacy and robustness of this method are demonstrated, both numerically and experimentally, on discovering a variety of PDE systems with different levels of data scarcity and noise accounting for different initial/boundary conditions. The resulting computational framework shows the potential for closed-form model discovery in practical applications where large and accurate datasets are intractable to capture.

This reposititory provides supplementary codes and data for the following work 

- Chen, Zhao, Yang Liu, and Hao Sun. "Physics-informed learning of governing equations from scarce data." arXiv preprint arXiv:2005.03448 (2020).

# System Requirements
## Hardware requirements
This work was done with
- a workstation with 28 Intel Core i9-7940X CPUs and 2 NVIDIA GTX 1080Ti GPU cards; and
- an Nvidia DGX with four Tesla V100 GPU of 32 GB memory.

## Software requirements
### OS Requirements
The package has been tested on the following systems:
- Windows 10 Pro
- Linux: Ubuntu 18.04.3 LTS
### Software dependencies 
This work was implemented on Tensorflow-gpu 1.15 in Python empowered by GPU. MATLAB was used for final visual presentation in the manuscript. 
### Installation guide
We recommend installing Python via Anaconda first along with GPU support programs and then installing related Python packages through conda. The list of software this work has used is
- NVIDIA® GPU drivers
- CUDA® Toolkit
- cuDNN
- Anaconda
- Spyder
- Python 3.7
- Tensorflow-gpu 1.15
- numpy
- matplotlib
- scipy
- matplotlib 
- mpl_toolkits
- pyDOE 
- tqdm

# How to run our cases?
To try out our simulations, you can visit "Examples" directory and run corresponding Python file in Spyder IDE. Most datas are included in the directory. For data larger than 25 MB, we provide data generation code instead.

Expected outcome will show the quality of system response prediction and equation discovery. Specifically, some error metrics and simple figures illustrating predictive system responses, discovered governing equations and loss convergence for diagnostics purpose. Note that we used MATLAB codes to design formal figures in the manuscript.  

The expected running time on our workstation varies from 20 minutes to several hours, depending on the complexity of the system, the network architecture and the amount of data.



