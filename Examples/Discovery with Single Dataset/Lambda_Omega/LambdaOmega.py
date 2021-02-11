# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the discovery of lambda-omega equation with a single dataset 
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
from Utils_LambdaOmega import *

with tf.device('/device:GPU:3'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data = scipy.io.loadmat('reaction_diffusion_standard.mat') # grid 256*256*201
    
    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    y = np.real(data['y'].flatten()[:,None])
    Exact_u = data['u']
    Exact_v = data['v']
        
    X, Y, T = np.meshgrid(x, y, t)
    
    X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None] 
    v_star = Exact_v.flatten()[:,None]        

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    ## measurements
    N_uv_s = 2500         
    # Use these commands when N_uv_s is larger than X.shape[0] or X.shape[1]
    idx = np.random.choice(X.shape[0]*X.shape[1], N_uv_s, replace = False)
    idx_remainder = idx%(X.shape[0])
    idx_s_y = np.floor(idx/(X.shape[0]))
    idx_s_y = idx_s_y.astype(np.int32)
    idx_idx_remainder = np.where(idx_remainder == 0)[0]
    idx_remainder[idx_idx_remainder] = X.shape[0]
    idx_s_x = idx_remainder-1 
    
    N_t_s = 15 
    idx_t = np.random.choice(X.shape[2], N_t_s, replace=False)
    idx_t = idx_t.astype(np.int32)   
    
    X1 = X[idx_s_x, idx_s_y, :]
    X2 = X1[:, idx_t]
    Y1 = Y[idx_s_x, idx_s_y, :]
    Y2 = Y1[:, idx_t]
    T1 = T[idx_s_x, idx_s_y, :]
    T2 = T1[:, idx_t]
    Exact_u1 = Exact_u[idx_s_x, idx_s_y, :]
    Exact_u2 = Exact_u1[:, idx_t]
    Exact_v1 = Exact_v[idx_s_x, idx_s_y, :]
    Exact_v2 = Exact_v1[:, idx_t]
    
    X_star_meas = np.hstack((X2.flatten()[:,None], Y2.flatten()[:,None],
                              T2.flatten()[:,None]))
    u_star_meas = Exact_u2.flatten()[:,None] 
    v_star_meas = Exact_v2.flatten()[:,None] 
    
    # free cpu memory
    del Exact_u1, Exact_u2, Exact_v1, Exact_v2, X1, Y1, T1, X2, Y2, T2
    
    # Training measurements
    Split_TrainVal = 0.8
    N_u_train = int(N_uv_s*N_t_s*Split_TrainVal)
    idx_train = np.random.choice(X_star_meas.shape[0], N_u_train, replace=False)
    X_star_train = X_star_meas[idx_train,:]
    u_star_train = u_star_meas[idx_train,:]
    v_star_train = v_star_meas[idx_train,:]
    
    # Validation Measurements
    idx_val = np.setdiff1d(np.arange(X_star_meas.shape[0]), idx_train, assume_unique=True)
    X_star_val = X_star_meas[idx_val,:]
    u_star_val = u_star_meas[idx_val,:]
    v_star_val = v_star_meas[idx_val,:]
    
    # Collocation points
    N_f = 10000

    X_f = lb + (ub-lb)*lhs(3, N_f)
#    X_f_train = lb + (ub-lb)*sobol_seq.i4_sobol_generate(2, N_f)        
    # X_f = np.vstack((X_f, X_train))
    
    # Optional: Add noise
    noise = 0.1
    u_star_train = u_star_train + noise*np.std(u_star)*np.random.randn(u_star_train.shape[0], u_star_train.shape[1])
    v_star_train = v_star_train + noise*np.std(v_star)*np.random.randn(v_star_train.shape[0], v_star_train.shape[1])
    u_star_val = u_star_val + noise*np.std(u_star)*np.random.randn(u_star_val.shape[0], u_star_val.shape[1])
    v_star_val = v_star_val + noise*np.std(v_star)*np.random.randn(v_star_val.shape[0], v_star_val.shape[1])

# =============================================================================
#   define and train model
# =============================================================================
    layers = [3] + [60]*8 + [2] # 8 hidden layers of 40 nodes

    # Adam_epochs_Pre = 10000 
    # BFGS_epochs_Pre = 40000 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 30
    # Adam_epochs_ADO = 1000
    # BFGS_epochs_ADO = 4000 # maximum allowable BFGS epochs in each ADO iteration 

    Adam_epochs_Pre = 20 
    BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 2
    Adam_epochs_ADO = 20
    BFGS_epochs_ADO = 20 # maximum allowable BFGS epochs in each ADO iteration 
    
    start_time = time.time()

    model = PiDL(layers, lb, ub, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO)
    model.train(X_star_train, u_star_train, v_star_train, X_f, X_star_val, u_star_val, v_star_val)
        
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    # optional post-training: refinement based on non-zero library coefficients
    # pt_ADO_iterations = 10
    # Adam_epochs_Pt = 1000
    
    pt_ADO_iterations = 2
    Adam_epochs_Pt = 20

    # model.post_train(pt_ADO_iterations, Adam_epochs_Pt)

    u_train_pred, v_train_pred = model.inference(X_star_train)
    error_u_train = np.linalg.norm(u_star_train-u_train_pred,2)/np.linalg.norm(u_star_train,2)        
    error_v_train = np.linalg.norm(v_star_train-v_train_pred,2)/np.linalg.norm(v_star_train,2)
    print('Training Error u: %e \n' % (error_u_train))    
    print('Training Error v: %e \n' % (error_v_train))
        
    u_val_pred, v_val_pred = model.inference(X_star_val)
    error_u_val = np.linalg.norm(u_star_val-u_val_pred,2)/np.linalg.norm(u_star_val,2)        
    error_v_val = np.linalg.norm(v_star_val-v_val_pred,2)/np.linalg.norm(v_star_val,2)
    print('Val Error u: %e \n' % (error_u_val))    
    print('Val Error v: %e \n' % (error_v_val))   

# =============================================================================
#   inference the full-field system response (if training is sufficient)
# =============================================================================
    # infer the full-field system response  
    u_full_pred, v_full_pred = model.inference(X_star)
    error_u_full = np.linalg.norm(u_star-u_full_pred,2)/np.linalg.norm(u_star,2)        
    error_v_full = np.linalg.norm(v_star-v_full_pred,2)/np.linalg.norm(v_star,2)
    print('Full Error u: %e \n' % (error_u_full))    
    print('Full Error v: %e \n' % (error_v_full))   
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'u_full_pred':u_full_pred, 'v_full_pred': v_full_pred}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_u_disc = model.sess.run(model.lambda_u)
    lambda_u_true = np.zeros_like(lambda_u_disc)
    lambda_u_true[2] = 0.1 # 0.1*u_xx
    lambda_u_true[4] = 0.1 # 0.1*u_yy
    lambda_u_true[11] = 1 # u   
    lambda_u_true[33] = -1 # -u^3
    lambda_u_true[66] = 1 # v^3
    lambda_u_true[88] = -1 # -u*v^2
    lambda_u_true[99] = 1 # u^2*v

    nonzero_ind_u = np.nonzero(lambda_u_true)
    lambda_error_vector_u = np.absolute((lambda_u_true[nonzero_ind_u]-lambda_u_disc[nonzero_ind_u])/lambda_u_true[nonzero_ind_u])
    lambda_error_mean_u = np.mean(lambda_error_vector_u)*100
    lambda_error_std_u = np.std(lambda_error_vector_u)*100
        
    print('lambda_error_mean_u: %.2f%% \n' % (lambda_error_mean_u))
    print('lambda_error_std_u: %.2f%% \n' % (lambda_error_std_u))
    
    disc_eq_temp = []
    for i_lib in range(len(model.lib_descr)):
        if lambda_u_disc[i_lib] != 0:
            disc_eq_temp.append(str(lambda_u_disc[i_lib,0]) + model.lib_descr[i_lib])
    disc_eq_u = '+'.join(disc_eq_temp)        
    print('The discovered equation: u_t = ' + disc_eq_u)

    lambda_v_disc = model.sess.run(model.lambda_v)
    lambda_v_true = np.zeros_like(lambda_v_disc)
    lambda_v_true[7] = 0.1 # 0.1*v_xx
    lambda_v_true[9] = 0.1 # 0.1*v_yy
    lambda_v_true[33] = -1 # -u^3
    lambda_v_true[44] = 1 # v
    lambda_v_true[66] = -1 # -v^3
    lambda_v_true[88] = -1 # -u*v^2
    lambda_v_true[99] = -1 # -u^2*v  
    
    nonzero_ind_v = np.nonzero(lambda_v_true)
    lambda_error_vector_v = np.absolute((lambda_v_true[nonzero_ind_v]-lambda_v_disc[nonzero_ind_v])/lambda_v_true[nonzero_ind_v])
    lambda_error_mean_v = np.mean(lambda_error_vector_v)*100
    lambda_error_std_v = np.std(lambda_error_vector_v)*100
        
    print('lambda_error_mean_v: %.2f%% \n' % (lambda_error_mean_v))
    print('lambda_error_std_v: %.2f%% \n' % (lambda_error_std_v))

    disc_eq_temp = []
    for i_lib in range(len(model.lib_descr)):
        if lambda_v_disc[i_lib] != 0:
            disc_eq_temp.append(str(lambda_v_disc[i_lib,0]) + model.lib_descr[i_lib])
    disc_eq_v = '+'.join(disc_eq_temp)        
    print('The discovered equation: v_t = ' + disc_eq_v)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_u_history_Pretrain':model.lambda_u_history_Pretrain[:, 1:],
                                            'lambda_v_history_Pretrain':model.lambda_v_history_Pretrain[:, 1:],
                                            'lambda_u_history_STRidge':model.lambda_u_history_STRidge[:,1:],
                                            'lambda_v_history_STRidge':model.lambda_v_history_STRidge[:,1:],
                                            'ridge_u_append_counter_STRidge':model.ridge_u_append_counter_STRidge[1:],
                                            'ridge_v_append_counter_STRidge':model.ridge_v_append_counter_STRidge[1:]}) 
