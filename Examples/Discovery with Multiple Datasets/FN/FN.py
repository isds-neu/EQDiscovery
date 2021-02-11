# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Main script for the discovery of FN equation with multiple datasets
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
from Utils_FN import *

with tf.device('/device:GPU:0'):  # run on GPU

    np.random.seed(1234)
    tf.set_random_seed(1234)
            
# =============================================================================
#     load data
# =============================================================================
    data_IC1 = scipy.io.loadmat('FN_IC2_Avoid104TS.mat')
    Exact_u_IC1 = np.real(data_IC1['u'])
    Exact_v_IC1 = np.real(data_IC1['v'])
    t = np.real(data_IC1['t'].flatten()[:,None])
    x = np.real(data_IC1['x'].flatten()[:,None])
    y = np.real(data_IC1['y'].flatten()[:,None])

    data_IC2 = scipy.io.loadmat('FN_IC3_Avoid104TS.mat')
    Exact_u_IC2 = np.real(data_IC2['u'])
    Exact_v_IC2 = np.real(data_IC2['v'])

    data_IC3 = scipy.io.loadmat('FN_IC4_Avoid104TS.mat')
    Exact_u_IC3 = np.real(data_IC3['u'])
    Exact_v_IC3 = np.real(data_IC3['v'])

    xx, yy, tt = np.meshgrid(x, y, t)
    
    X_star = np.hstack((xx.flatten()[:,None], yy.flatten()[:,None], tt.flatten()[:,None]))
    U_star_IC1 = np.hstack((Exact_u_IC1.flatten()[:,None], Exact_v_IC1.flatten()[:,None]))            
    U_star_IC2 = np.hstack((Exact_u_IC2.flatten()[:,None], Exact_v_IC2.flatten()[:,None]))            
    U_star_IC3 = np.hstack((Exact_u_IC3.flatten()[:,None], Exact_v_IC3.flatten()[:,None]))    
    U_star = np.stack((U_star_IC1, U_star_IC2, U_star_IC3), axis = -1)

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    ## measurements
    N_U_t = 31
        
    # continuous snapshots
    d_t = np.floor((t.shape[0])/(N_U_t - 1))
    idx_t = (np.arange(N_U_t)*d_t).astype(int)

    # downsize measurement grid
    N_U_x = 31
    d_x = np.floor((x.shape[0])/(N_U_x - 1))
    idx_x = (np.arange(1, N_U_x - 1)*d_x).astype(int) # remove bc
    X_U_meas, U_meas = DownsampleMeas(idx_x, idx_t, xx, yy, tt, Exact_u_IC1, Exact_v_IC1, Exact_u_IC2, Exact_v_IC2, Exact_u_IC3, Exact_v_IC3, XU_flag = True)
    
    # Training measurements, which are randomly sampled spatio-temporally
    Split_TrainVal = 1
    N_u_train = int(X_U_meas.shape[0]*Split_TrainVal)
    idx_train = np.random.choice(X_U_meas.shape[0], N_u_train, replace = False)
    np.random.shuffle(idx_train)
    X_U_train = X_U_meas[idx_train,:]
    U_train = U_meas[idx_train,:]
                            
    # bc meas
    idx_x = np.append((np.arange(N_U_x)*d_x).astype(int), -1) # include bc
    
    xx2, yy2, tt2, Exact_u_IC1_2, Exact_v_IC1_2, Exact_u_IC2_2, Exact_v_IC2_2, Exact_u_IC3_2, Exact_v_IC3_2 = DownsampleMeas(idx_x, idx_t, xx, yy, tt, Exact_u_IC1, Exact_v_IC1, Exact_u_IC2, Exact_v_IC2, Exact_u_IC3, Exact_v_IC3, XU_flag = False)

    # select the BCs                                                                                                                                 
    idx_x = np.array((0, -1)) # select the BCs
    xx_UB = xx2[idx_x, :, :]
    xx_LR = xx2[:, idx_x, :]
    
    yy_UB = yy2[idx_x, :, :]
    yy_LR = yy2[:, idx_x, :]

    tt_UB = tt2[idx_x, :, :]
    tt_LR = tt2[:, idx_x, :]    

    X_bc_meas_UB = np.hstack((xx_UB.flatten()[:,None], yy_UB.flatten()[:,None], tt_UB.flatten()[:,None]))  
    X_bc_meas_LR = np.hstack((xx_LR.flatten()[:,None], yy_LR.flatten()[:,None], tt_LR.flatten()[:,None]))  
    X_bc_meas = np.vstack((X_bc_meas_UB, X_bc_meas_LR))
    
    Exact_u_IC1_UB = Exact_u_IC1_2[idx_x, :, :]
    Exact_u_IC1_LR = Exact_u_IC1_2[:, idx_x, :]
    Exact_v_IC1_UB = Exact_v_IC1_2[idx_x, :, :]
    Exact_v_IC1_LR = Exact_v_IC1_2[:, idx_x, :]
            
    U_bc_meas_IC1_UB = np.hstack((Exact_u_IC1_UB.flatten()[:,None], Exact_v_IC1_UB.flatten()[:,None]))  
    U_bc_meas_IC1_LR = np.hstack((Exact_u_IC1_LR.flatten()[:,None], Exact_v_IC1_LR.flatten()[:,None]))  
    U_bc_meas_IC1 = np.vstack((U_bc_meas_IC1_UB, U_bc_meas_IC1_LR))

    Exact_u_IC2_UB = Exact_u_IC2_2[idx_x, :, :]
    Exact_u_IC2_LR = Exact_u_IC2_2[:, idx_x, :]
    Exact_v_IC2_UB = Exact_v_IC2_2[idx_x, :, :]
    Exact_v_IC2_LR = Exact_v_IC2_2[:, idx_x, :]
            
    U_bc_meas_IC2_UB = np.hstack((Exact_u_IC2_UB.flatten()[:,None], Exact_v_IC2_UB.flatten()[:,None]))  
    U_bc_meas_IC2_LR = np.hstack((Exact_u_IC2_LR.flatten()[:,None], Exact_v_IC2_LR.flatten()[:,None]))  
    U_bc_meas_IC2 = np.vstack((U_bc_meas_IC2_UB, U_bc_meas_IC2_LR))

    Exact_u_IC3_UB = Exact_u_IC3_2[idx_x, :, :]
    Exact_u_IC3_LR = Exact_u_IC3_2[:, idx_x, :]
    Exact_v_IC3_UB = Exact_v_IC3_2[idx_x, :, :]
    Exact_v_IC3_LR = Exact_v_IC3_2[:, idx_x, :]
            
    U_bc_meas_IC3_UB = np.hstack((Exact_u_IC3_UB.flatten()[:,None], Exact_v_IC3_UB.flatten()[:,None]))  
    U_bc_meas_IC3_LR = np.hstack((Exact_u_IC3_LR.flatten()[:,None], Exact_v_IC3_LR.flatten()[:,None]))  
    U_bc_meas_IC3 = np.vstack((U_bc_meas_IC3_UB, U_bc_meas_IC3_LR))
   
    U_bc_meas = np.stack((U_bc_meas_IC1, U_bc_meas_IC2, U_bc_meas_IC3), axis = -1)
    
    # Collocation points
    N_f = 40000    
    X_f_train = lb + (ub-lb)*lhs(X_U_meas.shape[1], N_f)
    
    # periodic bc
    X_l = np.hstack((xx[:, 0, :].flatten()[:,None], yy[:, 0, :].flatten()[:,None],
                          tt[:, 0, :].flatten()[:,None]))
    X_r = np.hstack((xx[:, -1, :].flatten()[:,None], yy[:, -1, :].flatten()[:,None],
                          tt[:, -1, :].flatten()[:,None]))
    X_u = np.hstack((xx[-1, :, :].flatten()[:,None], yy[-1, :, :].flatten()[:,None],
                          tt[-1, :, :].flatten()[:,None]))
    X_b = np.hstack((xx[0, :, :].flatten()[:,None], yy[0, :, :].flatten()[:,None],
                          tt[0, :, :].flatten()[:,None]))

    N_bc_sampled = 2500
    idx_l = np.random.choice(X_l.shape[0], N_bc_sampled, replace = False)
    idx_r = np.random.choice(X_r.shape[0], N_bc_sampled, replace = False)
    idx_u = np.random.choice(X_u.shape[0], N_bc_sampled, replace = False)
    idx_b = np.random.choice(X_b.shape[0], N_bc_sampled, replace = False)
    X_bc_sampled = np.vstack((X_l[idx_l, :], X_r[idx_r, :], X_u[idx_u, :], X_b[idx_b, :]))
    X_f_train = np.vstack((X_f_train, X_U_meas, X_bc_sampled, X_bc_meas))
    
    # Optional: Add noise
    noise = 0.1
    U_train = U_train + noise*np.std(U_train, axis = 0)*np.random.randn(U_train.shape[0], U_train.shape[1], U_train.shape[2])
    U_bc_meas = U_bc_meas + noise*np.std(U_bc_meas, axis = 0)*np.random.randn(U_bc_meas.shape[0], U_bc_meas.shape[1], U_bc_meas.shape[2])

# =============================================================================
#   define and train model
# =============================================================================
    layers_s = [3] + 2*[60] + [60] # root NN
    layers_i = 3*[60] + [2] # shared NN

    # pre_ADO_iterations = 3
    # Adam_epochs_Pre = 2000 
    # BFGS_epochs_Pre = 20000 # maximum allowable BFGS epochs in pretraining 
    # ADO_iterations = 10
    # Adam_epochs_ADO = 10000
    # BFGS_epochs_ADO = 10000 # maximum allowable BFGS epochs in each ADO iteration 
    
    pre_ADO_iterations = 2
    Adam_epochs_Pre = 20 
    BFGS_epochs_Pre = 20 # maximum allowable BFGS epochs in pretraining 
    ADO_iterations = 2
    Adam_epochs_ADO = 20
    BFGS_epochs_ADO = 20 # maximum allowable BFGS epochs in each ADO iteration 
    
    start_time = time.time()

    model = PiDL(layers_s, layers_i, lb, ub, pre_ADO_iterations, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO)
    model.train(X_U_train, U_train, X_f_train, X_l, X_r, X_u, X_b, X_bc_meas, U_bc_meas)
        
    elapsed = time.time() - start_time                
    print('Training time: %.4f \n' % (elapsed))
# =============================================================================
#   diagnostics
# =============================================================================
    # determine whether the training is sufficient
    model.visualize_training()
    
    # optional post-training: refinement based on non-zero library coefficients
    # pt_ADO_iterations = 10
    # Adam_epochs_Pt = 10000
    
    pt_ADO_iterations = 2
    Adam_epochs_Pt = 20

    model.post_train(pt_ADO_iterations, Adam_epochs_Pt)

    U_train_Pred = model.inference(X_U_train)   
    error_U_train = np.linalg.norm(np.reshape(U_train_Pred - U_train, (-1,1)),2) / \
        np.linalg.norm(np.reshape(U_train, (-1,1)),2)   
    print('Train Error u: %e \n' % (error_U_train)) 

# =============================================================================
#   inference the broader system response (if training is sufficient)
# =============================================================================
    # infer the broader system response due to memory limit
    N_U_t = 61
    d_t = np.floor((t.shape[0])/(N_U_t - 1))
    idx_t = (np.arange(N_U_t)*d_t).astype(int)

    N_U_x = 101
    d_x = np.floor((x.shape[0])/(N_U_x - 1))
    idx_x = (np.arange(N_U_x)*d_x).astype(int) 
    X_U_full, U_full = DownsampleMeas(idx_x, idx_t, xx, yy, tt, Exact_u_IC1, Exact_v_IC1, Exact_u_IC2, Exact_v_IC2, Exact_u_IC3, Exact_v_IC3, XU_flag = True)
    
    U_Full_Pred = model.inference(X_U_full)  
    error_U = np.linalg.norm(np.reshape(U_full - U_Full_Pred, (-1,1)),2)/np.linalg.norm(np.reshape(U_full, (-1,1)),2)   
    print('Full Field Error u: %e \n' % (error_U))
    
    # save inferred system response for plotting manuscript figures in MATLAB.
    scipy.io.savemat('Pred.mat',{'U_Full_Pred':U_Full_Pred}) 
    
# =============================================================================
#   compare discovered eq. with the ground truth (if training is sufficient)
# =============================================================================
    lambda_u_disc = model.sess.run(model.lambda_u)
    lambda_u_true = np.zeros_like(lambda_u_disc)
    lambda_u_true[0] = 0.01 # 1
    lambda_u_true[7] = 1 # u
    lambda_u_true[21] = -1 # u**3
    lambda_u_true[28] = -1 # v

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
    diff_u = model.sess.run(model.diff_coeff_u)
    disc_eq_temp.append(str(diff_u) + '*(u_xx + u_yy)')
    disc_eq_u = '+'.join(disc_eq_temp)        
    print('The discovered equation: u_t = ' + disc_eq_u)

    lambda_v_disc = model.sess.run(model.lambda_v)
    lambda_v_true = np.zeros_like(lambda_v_disc)
    lambda_v[7] = 0.25 # u
    lambda_v[28] = -0.25 # v
    
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
    diff_v = model.sess.run(model.diff_coeff_v)
    disc_eq_temp.append(str(diff_v) + '*(v_xx + v_yy)')
    disc_eq_v = '+'.join(disc_eq_temp)        
    print('The discovered equation: v_t = ' + disc_eq_v)

    # save lambda evolution during training for plotting manuscript figures in MATLAB.
    scipy.io.savemat('LambdaEvolution.mat',{'lambda_u_history_Pretrain':model.lambda_u_history_Pretrain[:, 1:],
                                            'lambda_v_history_Pretrain':model.lambda_v_history_Pretrain[:, 1:],
                                            'lambda_u_history_STRidge':model.lambda_u_history_STRidge[:,1:],
                                            'lambda_v_history_STRidge':model.lambda_v_history_STRidge[:,1:],
                                            'ridge_u_append_counter_STRidge':model.ridge_u_append_counter_STRidge[1:],
                                            'ridge_v_append_counter_STRidge':model.ridge_v_append_counter_STRidge[1:],
                                            'diff_coeff_u_history_Pretrain': model.diff_coeff_u_history_Pretrain[1:],
                                            'diff_coeff_v_history_Pretrain': model.diff_coeff_v_history_Pretrain[1:]}) 
