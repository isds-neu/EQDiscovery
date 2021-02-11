# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the discovery of Navier Stokes Vorticity equation with a single dataset 
# =============================================================================

import tensorflow as tf # tensorflow version 1.15.0
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from scipy.spatial import distance
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
#    import sobol_seq
import os
from Utils_Vorticity import *

with tf.device('/device:GPU:2'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data = scipy.io.loadmat('Vorticity_ALL.mat')        
    steps = 151
    n = 449
    m = 199
    dt = 0.2
    dx = 0.02
    dy = 0.02
    
    W = data['VORTALL'].reshape(n,m,steps)   # vorticity
    U = data['UALL'].reshape(n,m,steps)      # x-component of velocity
    V = data['VALL'].reshape(n,m,steps)      # y-component of velocity

    # Cut out the portion of the data before the cylinder
    xmin = 100
    xmax = 425
    ymin = 15
    ymax = 185        
    W = W[xmin:xmax,ymin:ymax,:]
    U = U[xmin:xmax,ymin:ymax,:]
    V = V[xmin:xmax,ymin:ymax,:]
    n,m,steps = W.shape
    
    # Preprocess data #1(First dimensiton is space and the second dimension is time. Y varies first in the first dimension)
    w_data = W.reshape(n*m, steps)
    u_data = U.reshape(n*m, steps)
    v_data = V.reshape(n*m, steps)
    
    t_data = np.arange(steps).reshape((1, -1))*dt         
    t_data = np.tile(t_data,(m*n,1))
        
    # This part reset the coordinates
    x_data = np.arange(n).reshape((-1, 1))*dx 
    x_data = np.tile(x_data, (1, m))
    x_data = np.reshape(x_data, (-1, 1))
    x_data = np.tile(x_data, (1, steps))    
    y_data = np.arange(m).reshape((1, -1))*dy 
    y_data = np.tile(y_data, (n, 1))
    y_data = np.reshape(y_data, (-1, 1))
    y_data = np.tile(y_data, (1, steps)) 
    
    t_star = np.reshape(t_data,(-1,1))
    x_star = np.reshape(x_data,(-1,1))
    y_star = np.reshape(y_data,(-1,1))        
    u_star = np.reshape(u_data,(-1,1))
    v_star = np.reshape(v_data,(-1,1))
    w_star = np.reshape(w_data,(-1,1))        
    X_star = np.hstack((x_star, y_star, t_star))

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    ## measurements
    N_s = 500         
    N_t = 60
    idx_s = np.random.choice(x_data.shape[0], N_s, replace = False)
    idx_t = np.random.choice(steps, N_t, replace = False)
    
    t_meas = t_data[idx_s, :]
    t_meas = t_meas[:, idx_t].reshape((-1,1))
    x_meas = x_data[idx_s, :]
    x_meas = x_meas[:, idx_t].reshape((-1,1))
    y_meas = y_data[idx_s, :]
    y_meas = y_meas[:, idx_t].reshape((-1,1))
    u_meas = u_data[idx_s, :]
    u_meas = u_meas[:, idx_t].reshape((-1,1))
    v_meas = v_data[idx_s, :]
    v_meas = v_meas[:, idx_t].reshape((-1,1))
    w_meas = w_data[idx_s, :]
    w_meas = w_meas[:, idx_t].reshape((-1,1))
    X_meas = np.hstack((x_meas, y_meas, t_meas))
    
    # Training measurements
    Split_TrainVal = 0.8
    N_train = int(N_s*N_t*Split_TrainVal)
    idx_train = np.random.choice(X_meas.shape[0], N_train, replace=False)
    X_train = X_meas[idx_train,:]
    u_train = u_meas[idx_train,:]
    v_train = v_meas[idx_train,:]
    w_train = w_meas[idx_train,:]        
    
    # Validation Measurements
    idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
    X_val = X_meas[idx_val,:]
    u_val = u_meas[idx_val,:]
    v_val = v_meas[idx_val,:]
    w_val = w_meas[idx_val,:]
    
    # Collocation points
    N_f = 60000

    X_f = lb + (ub-lb)*lhs(3, N_f)
#    X_f_train = lb + (ub-lb)*sobol_seq.i4_sobol_generate(2, N_f)        
    X_f = np.vstack((X_f, X_train))
    
    # Optional: Add noise
    noise = 0.1
    u_train = u_train + noise*np.std(u_star)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise*np.std(v_star)*np.random.randn(v_train.shape[0], v_train.shape[1])
    w_train = w_train + noise*np.std(w_star)*np.random.randn(w_train.shape[0], w_train.shape[1])
    u_val = u_val + noise*np.std(u_star)*np.random.randn(u_val.shape[0], u_val.shape[1])
    v_val = v_val + noise*np.std(v_star)*np.random.randn(v_val.shape[0], v_val.shape[1])
    w_val = w_val + noise*np.std(w_star)*np.random.randn(w_val.shape[0], w_val.shape[1])

# =============================================================================
#   define and train model
# =============================================================================
    layers = [3] + [60]*8 + [3] # 8 hidden layers of 40 nodes

    Adam_epochs_Pre = 5000 
    BFGS_epochs_Pre = 10000 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 6
    Adam_epochs_ADO = 500
    BFGS_epochs_ADO = 1000 # maximum allowable BFGS epochs in each ADO iteration 

    # Adam_epochs_Pre = 20 
    # BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 2
    # Adam_epochs_ADO = 20
    # BFGS_epochs_ADO = 20 # maximum allowable BFGS epochs in each ADO iteration 
    
    start_time = time.time()

    model = PiDL(layers, lb, ub, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO)
    model.train(X_train, u_train, v_train, w_train, X_f, X_val, u_val, v_val, w_val)
    
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    u_train_pred, v_train_pred, w_train_pred = model.inference(X_train)
    error_u_train = np.linalg.norm(u_train-u_train_pred,2)/np.linalg.norm(u_train,2)        
    error_v_train = np.linalg.norm(v_train-v_train_pred,2)/np.linalg.norm(v_train,2)
    error_w_train = np.linalg.norm(w_train-w_train_pred,2)/np.linalg.norm(w_train,2)
    print('Training Error u: %e \n' % (error_u_train))    
    print('Training Error v: %e \n' % (error_v_train))
    print('Training Error w: %e \n' % (error_w_train))
        
    u_val_pred, v_val_pred, w_val_pred = model.inference(X_val)
    error_u_val = np.linalg.norm(u_val-u_val_pred,2)/np.linalg.norm(u_val,2)        
    error_v_val = np.linalg.norm(v_val-v_val_pred,2)/np.linalg.norm(v_val,2)
    error_w_val = np.linalg.norm(w_val-w_val_pred,2)/np.linalg.norm(w_val,2)
    print('Val Error u: %e \n' % (error_u_val))    
    print('Val Error v: %e \n' % (error_v_val))   
    print('Val Error w: %e \n' % (error_w_val))        

# =============================================================================
#   inference the full-field system response (if training is sufficient)
# =============================================================================
    # infer the full-field system response  
    u_full_pred, v_full_pred, w_full_pred = model.inference(X_star)
    error_u_full = np.linalg.norm(u_star-u_full_pred,2)/np.linalg.norm(u_star,2)        
    error_v_full = np.linalg.norm(v_star-v_full_pred,2)/np.linalg.norm(v_star,2)
    error_w_full = np.linalg.norm(w_star-w_full_pred,2)/np.linalg.norm(w_star,2)
    print('Full Error u: %e \n' % (error_u_full))    
    print('Full Error v: %e \n' % (error_v_full))   
    print('Full Error w: %e \n' % (error_w_full))    
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'u_full_pred':u_full_pred, 'v_full_pred': v_full_pred, 'w_full_pred':w_full_pred}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_disc = model.sess.run(model.lambda1)
    lambda_true = np.zeros_like(lambda_disc)
    lambda_true[3] = 0.01 # 0.01w_xx
    lambda_true[5] = 0.01 # 0.01*w_yy
    lambda_true[7] = -1 # -uw_x   
    lambda_true[20] = -1 # -vw_y

    nonzero_ind = np.nonzero(lambda_true)
    lambda_error_vector = np.absolute((lambda_true[nonzero_ind]-lambda_disc[nonzero_ind])/lambda_true[nonzero_ind])
    lambda_error_mean = np.mean(lambda_error_vector)*100
    lambda_error_std = np.std(lambda_error_vector)*100
        
    print('lambda_error_mean: %.2f%% \n' % (lambda_error_mean))
    print('lambda_error_std: %.2f%% \n' % (lambda_error_std))
    
    disc_eq_temp = []
    for i_lib in range(len(model.lib_descr)):
        if lambda_disc[i_lib] != 0:
            disc_eq_temp.append(str(lambda_disc[i_lib,0]) + model.lib_descr[i_lib])
    disc_eq = '+'.join(disc_eq_temp)        
    print('The discovered equation: w_t = ' + disc_eq)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_history_Pretrain':model.lambda_history_Pretrain[:, 1:],
                                            'lambda_history_STRidge':model.lambda_history_STRidge[:,1:],
                                            'ridge_append_counter_STRidge':model.ridge_append_counter_STRidge[1:]}) 
