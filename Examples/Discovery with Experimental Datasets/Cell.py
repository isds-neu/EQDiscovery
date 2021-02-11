# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the equation discovery of cell migration
# =============================================================================

import tensorflow as tf # tensorflow version 1.15.0
import numpy as np
# import matplotlib.pyplot as plt
import scipy.io
# from scipy.interpolate import griddata
# from scipy.spatial import distance
# from matplotlib import cm
import time
# from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
#    import sobol_seq
# import os
from Utils_Cell import *

with tf.device('/device:GPU:1'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data = scipy.io.loadmat('data_e.mat')

    # use the average at all time steps
    Exact = np.real(data['C'])
    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    
    xx, tt = np.meshgrid(x, t)
    
    X_star = np.hstack((xx.flatten()[:,None], tt.flatten()[:,None]))
    U_star = Exact.flatten()[:,None]            

    # flux = 0 bc
    X_l = np.hstack((xx[:, 0].flatten()[:,None],
                          tt[:, 0].flatten()[:,None]))
    X_r = np.hstack((xx[:, -1].flatten()[:,None], 
                          tt[:, -1].flatten()[:,None]))
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    # Measurement data
    X_U_meas = X_star
    U_meas = U_star
    
    # Collocation points
    N_f = 10000    
    X_f_train = lb + (ub-lb)*lhs(X_U_meas.shape[1], N_f)        
    X_f_train = np.vstack((X_f_train, X_U_meas))
    
# =============================================================================
#   define and train model
# =============================================================================
    layers = [2] + 3*[30] + [1]

    pre_ADO_iterations = 2
    Adam_epochs_Pre = 2000 
    BFGS_epochs_Pre = 2000 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 5
    Adam_epochs_ADO = 2000
    
    # pre_ADO_iterations = 2
    # Adam_epochs_Pre = 20 
    # BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 2
    # Adam_epochs_ADO = 20
    
    start_time = time.time()

    model = PiDL(layers, lb, ub, pre_ADO_iterations, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO)
    model.train(X_U_meas, U_meas, X_f_train, X_l, X_r)
        
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    # optional post-training: refinement based on non-zero library coefficients
    pt_ADO_iterations = 200
    Adam_epochs_Pt = 500
    
    # pt_ADO_iterations = 2
    # Adam_epochs_Pt = 20

    model.post_train(pt_ADO_iterations, Adam_epochs_Pt)

# =============================================================================
#   inference the full-field system response (if training is sufficient)
# =============================================================================
    # infer the full-field  system response
    U_Full_Pred = model.inference(X_star)  
    error_U = np.linalg.norm(np.reshape(U_star - U_Full_Pred, (-1,1)),2)/np.linalg.norm(np.reshape(U_star, (-1,1)),2)   
    print('Full Field Error u: %e \n' % (error_U))    
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'U_Full_Pred':U_Full_Pred}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_u_disc = model.sess.run(model.lambda_u)
    
    disc_eq_temp = []
    for i_lib in range(len(model.lib_descr)):
        if lambda_u_disc[i_lib] != 0:
            disc_eq_temp.append(str(lambda_u_disc[i_lib,0]) + model.lib_descr[i_lib])
    diff_u = model.sess.run(model.diff_coeff_u)
    disc_eq_temp.append(str(diff_u) + '*u_xx')
    disc_eq_u = '+'.join(disc_eq_temp)        
    print('The discovered equation: u_t = ' + disc_eq_u)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_u_history_Pretrain':model.lambda_u_history_Pretrain[:, 1:],
                                            'lambda_u_history_STRidge':model.lambda_u_history_STRidge[:,1:],
                                            'ridge_u_append_counter_STRidge':model.ridge_u_append_counter_STRidge[1:],
                                            'diff_coeff_u_history_Pretrain': model.diff_coeff_u_history_Pretrain[1:]}) 
