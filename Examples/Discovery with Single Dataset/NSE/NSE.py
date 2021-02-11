# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the discovery of nonlinear Schrodinger equation with a single dataset 
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
from Utils_NSE import *

with tf.device('/device:GPU:1'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data = scipy.io.loadmat('nse.mat')

    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    Exact_r = np.real(data['U_real']) # real part
    Exact_i = data['U_imag'] # imaginary part
    
    X, T = np.meshgrid(x,t)    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    r_star = Exact_r.flatten()[:,None] 
    i_star = Exact_i.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    # Measurement data
    N_u_s = int(0.5*X.shape[1])
    idx_s = np.random.choice(X.shape[1], N_u_s, replace=False)
    N_u_t = int(0.75*X.shape[0])
    idx_t = np.random.choice(X.shape[0], N_u_t, replace=False)
    X1 = X[:, idx_s]
    X2 = X1[idx_t, :]
    T1 = T[:, idx_s]
    T2 = T1[idx_t, :]
    Exact_r1 = Exact_r[:, idx_s]
    Exact_r2 = Exact_r1[idx_t, :]
    Exact_i1 = Exact_i[:, idx_s]
    Exact_i2 = Exact_i1[idx_t, :]
    
    X_meas = np.hstack((X2.flatten()[:,None], T2.flatten()[:,None]))
    r_meas = Exact_r2.flatten()[:,None]   
    i_meas = Exact_i2.flatten()[:,None]    
    
    # Training measurements
    Split_TrainVal = 0.8
    N_u_train = int(X_meas.shape[0]*Split_TrainVal)
    idx_train = np.random.choice(X_meas.shape[0], N_u_train, replace=False)
    X_train = X_meas[idx_train,:]
    r_train = r_meas[idx_train,:]
    i_train = i_meas[idx_train,:]
    
    # Validation Measurements
    idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
    X_val = X_meas[idx_val,:]
    r_val = r_meas[idx_val,:]
    i_val = i_meas[idx_val,:]
    
    # Collocation points
    N_f = 50000

    X_f_train = lb + (ub-lb)*lhs(2, N_f)
#    X_f_train = lb + (ub-lb)*sobol_seq.i4_sobol_generate(2, N_f)        
    # X_f_train = np.vstack((X_f_train, X_train))
    
    # Optional: Add noise
    noise = 0.1
    r_train = r_train + noise*np.std(r_train)*np.random.randn(r_train.shape[0], r_train.shape[1])
    i_train = i_train + noise*np.std(i_train)*np.random.randn(i_train.shape[0], i_train.shape[1])
    r_val = r_val + noise*np.std(r_val)*np.random.randn(r_val.shape[0], r_val.shape[1])
    i_val = i_val + noise*np.std(i_val)*np.random.randn(i_val.shape[0], i_val.shape[1])

# =============================================================================
#   define and train model
# =============================================================================
    layers = [2] + [40]*8 + [2] # 8 hidden layers of 40 nodes

    Adam_epochs_Pre = 160000 
    BFGS_epochs_Pre = 160000 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 30
    Adam_epochs_ADO = 1000
    BFGS_epochs_ADO = 4000 # maximum allowable BFGS epochs in each ADO iteration 

    # Adam_epochs_Pre = 20 
    # BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 2
    # Adam_epochs_ADO = 20
    # BFGS_epochs_ADO = 20 # maximum allowable BFGS epochs in each ADO iteration 
    
    start_time = time.time()

    model = PiDL(layers, lb, ub, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO)
    model.train(X_train, r_train, i_train, X_f_train, X_val, r_val, i_val)
    
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    r_train_pred, i_train_pred = model.inference(X_train)
    error_r_train = np.linalg.norm(r_train-r_train_pred,2)/np.linalg.norm(r_train,2)        
    error_i_train = np.linalg.norm(i_train-i_train_pred,2)/np.linalg.norm(i_train,2)
    print('Training Error r: %e \n' % (error_r_train))    
    print('Training Error i: %e \n' % (error_i_train))
        
    r_val_pred, i_val_pred = model.inference(X_val)
    error_r_val = np.linalg.norm(r_val-r_val_pred,2)/np.linalg.norm(r_val,2)        
    error_i_val = np.linalg.norm(i_val-i_val_pred,2)/np.linalg.norm(i_val,2)
    print('Val Error r: %e \n' % (error_i_val))    
    print('Val Error i: %e \n' % (error_i_val))        

# =============================================================================
#   inference the full-field system response (if training is sufficient)
# =============================================================================
    # infer the full-field system response  
    r_full_pred, i_full_pred = model.inference(X_star)
    error_r_full = np.linalg.norm(r_star-r_full_pred,2)/np.linalg.norm(r_star,2)        
    error_i_full = np.linalg.norm(i_star-i_full_pred,2)/np.linalg.norm(i_star,2)
    print('Full Error r: %e \n' % (error_r_full))    
    print('Full Error i: %e \n' % (error_i_full))    

    # visualize the prediction: the real part
    U_pred = griddata(X_star, r_full_pred.flatten(), (X, T), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('r')
    plt.title('Model Result: real part')       
    plt.savefig('FullField_ModelPred_r.png')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, Exact_r, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('r')
    plt.title('Ground Truth: real part')
    plt.savefig('FullField_GroundTruth_r.png')
    
    # visualize the prediction: the imaginary part
    U_pred = griddata(X_star, i_full_pred.flatten(), (X, T), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('imag')
    plt.title('Model Result: imaginary part')       
    plt.savefig('FullField_ModelPred_i.png')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, Exact_i, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('imag')
    plt.title('Ground Truth: imaginary part')
    plt.savefig('FullField_GroundTruth_i.png')
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'r_full_pred':r_full_pred, 'i_full_pred': i_full_pred}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_disc = model.sess.run(model.lambda1)
    lambda_true = np.zeros_like(lambda_disc)
    lambda_true[2] = -0.5 # -0.5u_xx
    lambda_true[12] = -1 # -|u|**2*u

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
    print('The discovered equation: i*u_t = ' + disc_eq)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_history_Pretrain':model.lambda_history_Pretrain[:, 1:],
                                            'lambda_history_STRidge':model.lambda_history_STRidge[:,1:],
                                            'ridge_append_counter_STRidge':model.ridge_append_counter_STRidge[1:]}) 
