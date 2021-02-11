# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the discovery of Burgers equation with multiple datasets 
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
# import os
from Utils_Burgers import *

with tf.device('/device:GPU:1'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data_Sine = scipy.io.loadmat('Burgers_SineIC_new.mat')
    t = np.real(data_Sine['t'].flatten()[:,None])
    x = np.real(data_Sine['x'].flatten()[:,None])
    Exact_Sine = np.real(data_Sine['u'])
  
    data_Cube = scipy.io.loadmat('Burgers_CubeIC_new.mat')
    Exact_Cube = np.real(data_Cube['u'])

    data_Gauss = scipy.io.loadmat('Burgers_GaussIC_new.mat')
    Exact_Gauss = np.real(data_Gauss['u'])
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star_Sin = Exact_Sine.flatten()[:,None]              
    u_star_Cube = Exact_Cube.flatten()[:,None]              
    u_star_Gauss = Exact_Gauss.flatten()[:,None]                         

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    # Measurement data
    N_u_s = 30 
    idx_s = np.random.choice(x.shape[0], N_u_s, replace=False)
    X0 = X[:, idx_s]
    T0 = T[:, idx_s]
    
    N_u_t = 500
    dt = np.floor(t.shape[0]/N_u_t)
    idx_t = (np.arange(N_u_t)*dt).astype(int)
    
    X_u_meas = np.hstack((X0[idx_t, :].flatten()[:,None], T0[idx_t, :].flatten()[:,None]))
        
    Exact0 = Exact_Sine[:, idx_s]
    u_meas_S = Exact0[idx_t, :].flatten()[:,None]   
 
    Exact1 = Exact_Cube[:, idx_s]
    u_meas_C = Exact1[idx_t, :].flatten()[:,None]   

    Exact2 = Exact_Gauss[:, idx_s]
    u_meas_G = Exact2[idx_t, :].flatten()[:,None]   
    
    u_meas = np.hstack((u_meas_S, u_meas_C, u_meas_G))
    
    # Training measurements
    Split_TrainVal = 0.8
    N_u_train = int(X_u_meas.shape[0]*Split_TrainVal)
    idx_train = np.arange(N_u_train)
    X_u_train = X_u_meas[idx_train,:]
    u_train = u_meas[idx_train,:]
    
    # Validation Measurements
    idx_val = np.setdiff1d(np.arange(X_u_meas.shape[0]), idx_train, assume_unique=True)
    X_u_val = X_u_meas[idx_val,:]
    u_val = u_meas[idx_val,:]
    
    # Collocation points
    N_f = 45000    
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_meas))
    
    # Optional: Add noise
    noise = 0.1
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])

# =============================================================================
#   define and train model
# =============================================================================
    layers_s = [2, 20, 20, 20, 20] # shared/root NN
    layers_i = [20, 30, 30, 30, 30, 1] # individual/branch NN
    
    BFGS_epochs_Pre = 30000 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 6
    Adam_epochs_ADO = 1000
    BFGS_epochs_ADO = 30000 # maximum allowable BFGS epochs in each ADO iteration 

    # BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 2
    # Adam_epochs_ADO = 20
    # BFGS_epochs_ADO = 20 # maximum allowable BFGS epochs in each ADO iteration 
    
    start_time = time.time()

    model = PiDL(layers_s, layers_i, lb, ub, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO)
    model.train(X_u_train, u_train, X_f_train, X_u_val, u_val)
    
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    u_train_Pred_Sin, u_train_Pred_Cube, u_train_Pred_Gauss = model.inference(X_u_train)                
    u_train_Pred = np.hstack((u_train_Pred_Sin, u_train_Pred_Cube, u_train_Pred_Gauss))
    Error_u_Train = np.linalg.norm(u_train-u_train_Pred,2)/np.linalg.norm(u_train,2)   
    print('Training Error u: %e \n' % (Error_u_Train))     
    
    u_val_Pred_Sin, u_val_Pred_Cube, u_val_Pred_Gauss = model.inference(X_u_val)                
    u_val_Pred = np.hstack((u_val_Pred_Sin, u_val_Pred_Cube, u_val_Pred_Gauss))
    Error_u_val = np.linalg.norm(u_val-u_val_Pred,2)/np.linalg.norm(u_val,2)   
    print('Validation Error u: %e \n' % (Error_u_val))             

# =============================================================================
#   inference the full-field system response (if training is sufficient)
# =============================================================================
    # infer the full-field system response  
    u_Full_Pred_Sin, u_Full_Pred_Cube, u_Full_Pred_Gauss = model.inference(X_star)  
    u_Full_Pred = np.hstack((u_Full_Pred_Sin, u_Full_Pred_Cube, u_Full_Pred_Gauss))
    u_star = np.hstack((u_star_Sin, u_star_Cube, u_star_Gauss))
    error_u = np.linalg.norm(u_star-u_Full_Pred,2)/np.linalg.norm(u_star,2)   
    print('Full Field Error u: %e \n' % (error_u))     

    # visualize the prediction
    ## Sine
    U_pred_sine = griddata(X_star, u_Full_Pred_Sin.flatten(), (X, T), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, U_pred_sine, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Model Result: Sine')       
    plt.savefig('ModelResult_Sine.png')
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, Exact_Sine, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Ground Truth: Sine')
    plt.savefig('GroundTruth_Sine.png')

    ## Cube
    U_pred_cube = griddata(X_star, u_Full_Pred_Cube.flatten(), (X, T), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, U_pred_cube, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Model Result: Cube') 
    plt.savefig('ModelResult_Cube.png')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, Exact_Cube, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Ground Truth: Cube')
    plt.savefig('GroundTruth_Cube.png')

    ## Sine
    U_pred_Gauss = griddata(X_star, u_Full_Pred_Gauss.flatten(), (X, T), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, U_pred_Gauss, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Model Result: Gauss')       
    plt.savefig('ModelResult_Gauss.png')
    
    # plot the whole domain truth
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, T, Exact_Gauss, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title('Ground Truth: Gauss')
    plt.savefig('GroundTruth_Gauss.png')
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'u_pred_sin': U_pred_sine, 'u_pred_cube': U_pred_cube,
                                 'u_pred_gauss': U_pred_Gauss}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_disc = model.sess.run(model.lambda1)
    lambda_true = np.zeros_like(lambda_disc)
    lambda_true[5] = -1 # uu_x
    lambda_true[8] = 0.01/3.1415926 # u_xx

    nonzero_ind = np.nonzero(lambda_true)
    lambda_error_vector = np.absolute((lambda_true[nonzero_ind]-lambda_disc[nonzero_ind])/lambda_true[nonzero_ind])
    lambda_error_mean = np.mean(lambda_error_vector)*100
    lambda_error_std = np.std(lambda_error_vector)*100
        
    print('lambda_error_mean: %.2f%% \n' % (lambda_error_mean))
    print('lambda_error_std: %.2f%% \n' % (lambda_error_std))
    
    disc_eq_temp = []
    for i_lib in range(len(model.library_description)):
        if lambda_disc[i_lib] != 0:
            disc_eq_temp.append(str(lambda_disc[i_lib,0]) + model.library_description[i_lib])
    disc_eq = '+'.join(disc_eq_temp)        
    print('The discovered equation: u_t = ' + disc_eq)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_history_Pretrain':model.lambda_history_Pretrain[:, 1:],
                                            'lambda_history_STRidge':model.lambda_history_STRidge[:,1:],
                                            'ridge_append_counter_STRidge':model.ridge_append_counter_STRidge[1:]}) 
