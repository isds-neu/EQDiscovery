# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# =============================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import distance
import time
#import sobol_seq
from pyDOE import lhs
from tqdm import tqdm
    
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with tf.device('/device:GPU:1'):
    
    # loss history (Pt)
    loss_history_Pt = np.empty([0])
    loss_val_history_Pt = np.empty([0])
    loss_r_history_Pt = np.empty([0])
    loss_i_history_Pt = np.empty([0])
    loss_f_history_Pt = np.empty([0])
    Lambda_history_Pt = np.empty([2, 1])
    step_Pt = 0
        
    lib_fun = []
    lib_descr = []
            
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, r, imag, X_f, X_val, r_val, imag_val, layers, lb, ub, Lamu_init):
            
            self.lb = lb.astype(np.float32)
            self.ub = ub.astype(np.float32)
            self.layers = layers
            
            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters
            # Assume all candidate functions are multiplied by a complex number i
            self.Lambda = tf.Variable(tf.zeros([40, 1], dtype=tf.float32), dtype=tf.float32) 
            self.Lambda2 = tf.Variable(Lamu_init, dtype=tf.float32) 

            # Specify the list of trainable variables 
            var_list_1 = self.biases + self.weights
            
            var_list_Pretrain = self.biases + self.weights
            var_list_Pretrain.append(self.Lambda)
            
            var_list_Pt = self.biases + self.weights + [self.Lambda2]
            
            ######### Training data ################
            self.x = X[:,0:1]
            self.t = X[:,1:2]
            self.r = r
            self.i = imag
            # Collocation points
            self.x_f = X_f[:,0:1]
            self.t_f = X_f[:,1:2]
            
            self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
            self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
            self.r_tf = tf.placeholder(tf.float32, shape=[None, 1])
            self.i_tf = tf.placeholder(tf.float32, shape=[None, 1])
            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
            
            self.r_pred, self.i_pred = self.net_real_imag(self.x_tf, self.t_tf)
            self.f_pred, self.Phi, self.iu_t_pred = self.net_f(self.x_f_tf, self.t_f_tf)
            
            self.loss_r = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
            self.loss_i = tf.reduce_mean(tf.square(self.i_tf - self.i_pred))
            
            self.loss_f_coeff_tf = tf.placeholder(tf.float32)
            
            self.loss_f = self.loss_f_coeff_tf*tf.reduce_mean(tf.square(tf.abs(self.f_pred))) # the average of square modulus
                        
            self.loss = tf.log(self.loss_r  + self.loss_i + self.loss_f)
                        
            ######### Validation data ###############
            self.x_val = X_val[:,0:1]
            self.t_val = X_val[:,1:2]
            self.r_val = r_val
            self.i_val = i_val
            
            self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
            self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
            self.r_val_tf = tf.placeholder(tf.float32, shape=[None, self.r_val.shape[1]])
            self.i_val_tf = tf.placeholder(tf.float32, shape=[None, self.i_val.shape[1]])
            
            self.r_val_pred, self.i_val_pred = self.net_real_imag(self.x_val_tf, self.t_val_tf)
            
            self.loss_r_val = tf.reduce_mean(tf.square(self.r_val_tf - self.r_val_pred))
            self.loss_i_val = tf.reduce_mean(tf.square(self.i_val_tf - self.i_val_pred))
            self.loss_val = tf.log(self.loss_r_val  + self.loss_i_val)     
                        
            ######### Optimizor #########################                                                                    
            self.optimizer_Pt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                            var_list = var_list_Pt,
            #                                                                    L-BFGS-B
                                                                            method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 10000,
                                                                                       'maxfun': 10000,
                                                                                       'maxcor': 50,
                                                                                       'maxls': 50,
                                                                                       'ftol' : 0.01 * np.finfo(float).eps})
            
            # the default learning rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08
            self.optimizer_Adam_Pt = tf.train.AdamOptimizer(learning_rate = 1e-3)
            self.train_op_Adam_Pt = self.optimizer_Adam_Pt.minimize(self.loss, var_list = var_list_Pt)
            
            # Save the model after pretraining
            self.saver = tf.train.Saver(var_list_Pretrain)
            self.saver_pt = tf.train.Saver(var_list_Pt)
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
    
            self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.r_tf: self.r, self.i_tf: self.i, 
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.r_val_tf: self.r_val,
                       self.i_val_tf: self.i_val}

        def initialize_NN(self, layers):        
            weights = []
            biases = []
            num_layers = len(layers) 
            for l in range(0,num_layers-1):
                W = self.xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name = 'b')
                weights.append(W)
                biases.append(b)        
            return weights, biases
            
        def xavier_init(self, size):
            in_dim = size[0]
            out_dim = size[1]        
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name = 'W')
        
        def neural_net(self, X, weights, biases):
            num_layers = len(weights) + 1            
            H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
            for l in range(0,num_layers-2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = weights[-1]
            b = biases[-1]
            Y = tf.add(tf.matmul(H, W), b)            
            return Y
                     
        def net_real_imag(self, x, t):  
            Y = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
            r = Y[:, 0:1]
            imag = Y[:, 1:2]
            return r, imag
        
        def net_f(self, x, t):            
            r, imag = self.net_real_imag(x,t)
            u = tf.complex(r, imag)
            
            ## Compute complex gradient library
            r_t = tf.gradients(r, t)[0]
            imag_t = tf.gradients(imag, t)[0]
            u_t = tf.complex(r_t, imag_t)
            iu_t = u_t*tf.constant(1j, dtype = tf.complex64)
                                    
            # Derivative terms
            r_x = tf.gradients(r, x)[0]
            imag_x = tf.gradients(imag, x)[0]
            u_x = tf.complex(r_x, imag_x)
            
            r_xx = tf.gradients(r_x, x)[0] 
            imag_xx = tf.gradients(imag_x, x)[0] 
            u_xx = tf.complex(r_xx, imag_xx)
            
            r_xxx = tf.gradients(r_xx, x)[0]
            imag_xxx = tf.gradients(imag_xx, x)[0]
            u_xxx = tf.complex(r_xxx, imag_xxx)
                        
            u_ABS = tf.complex(tf.abs(u), tf.zeros(shape=[N_f, 1]))
            u_ABS_SQ = u_ABS**2
            
                                        
            global lib_fun
            global lib_descr
            lib_fun = [u_xx, u*u_ABS_SQ]
            self.lib_descr = ['u_xx', 'u|u|**2']
                        
            Phi = tf.concat(lib_fun, axis = 1)            
            f = iu_t - tf.matmul(Phi, tf.complex(self.Lambda2, tf.zeros([2, 1], dtype=tf.float32))) 
            
            return f, Phi, iu_t
                    
        def callback_Pt(self, loss, loss_r, loss_i, loss_f, loss_val, Lambda):
            global step_Pt
            step_Pt += 1
            if step_Pt%10 == 0:
                
                global loss_history_Pt
                global loss_val_history_Pt
                global loss_r_history_Pt
                global loss_i_history_Pt
                global loss_f_history_Pt
                global Lambda_history_Pt
                
                loss_history_Pt = np.append(loss_history_Pt, loss)
                loss_val_history_Pt = np.append(loss_val_history_Pt, loss_val)
                loss_r_history_Pt = np.append(loss_r_history_Pt, loss_r)
                loss_i_history_Pt = np.append(loss_i_history_Pt, loss_i)
                loss_f_history_Pt = np.append(loss_f_history_Pt, loss_f)
                Lambda_history_Pt = np.append(Lambda_history_Pt, Lambda, axis = 1)                
                
        def Pt(self):                            
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            
            self.tf_dict[self.loss_f_coeff_tf] = 0.5
                        
            print('Adam(Pt) starts')
            for it_Adam in tqdm(range(10000)):
                self.sess.run(self.train_op_Adam_Pt, self.tf_dict, options=run_options)
                
                # Print
                if it_Adam % 10 == 0:
                    loss, loss_r, loss_i, loss_f, Lambda, loss_val = \
                        self.sess.run([self.loss, self.loss_r, self.loss_i, self.loss_f,
                                       self.Lambda2, self.loss_val], self.tf_dict)
                    
                    global loss_history_Pt
                    global loss_val_history_Pt
                    global loss_r_history_Pt
                    global loss_i_history_Pt
                    global loss_f_history_Pt
                    global Lambda_history_Pt
                    
                    loss_history_Pt = np.append(loss_history_Pt, loss)
                    loss_val_history_Pt = np.append(loss_val_history_Pt, loss_val)
                    loss_r_history_Pt = np.append(loss_r_history_Pt, loss_r)
                    loss_i_history_Pt = np.append(loss_i_history_Pt, loss_i)
                    loss_f_history_Pt = np.append(loss_f_history_Pt, loss_f)
                    Lambda_history_Pt = np.append(Lambda_history_Pt, Lambda, axis = 1)
                    
                    # L-BFGS-B optimizer(Pretraining)
            print('L-BFGS-B(Pt) starts')
            self.optimizer_Pt.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss, self.loss_r, self.loss_i, self.loss_f, 
                                                self.loss_val, self.Lambda2],
                                    loss_callback = self.callback_Pt)
                                       
        def predict(self, X_star):            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}            
            r_star = self.sess.run(self.r_pred, tf_dict)
            i_star = self.sess.run(self.i_pred, tf_dict)
            return r_star, i_star
    
    if __name__ == "__main__": 
         
        start_time = time.time()
        
        layers = [2] + 8*[40] + [2]
        
# =============================================================================
#         load data
# =============================================================================
        data = scipy.io.loadmat('nse.mat') 
            
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact_r = data['U_real'] # real part
        Exact_i = data['U_imag'] # imaginary part
        
        X, T = np.meshgrid(x,t)
        
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        r_star = Exact_r.flatten()[:,None] 
        i_star = Exact_i.flatten()[:,None]              
    
        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)    
            
        # In this case, measurements are from N_u_s points and continuously sampled all the time.
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
        
        # Training measurements, which are randomly sampled spatio-temporally
        Split_TrainVal = 0.8
        N_u_train = int(N_u_s*N_u_t*Split_TrainVal)
        idx_train = np.random.choice(X_meas.shape[0], N_u_train, replace=False)
        X_train = X_meas[idx_train,:]
        r_train = r_meas[idx_train,:]
        i_train = i_meas[idx_train,:]
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
        X_val = X_meas[idx_val,:]
        r_val = r_meas[idx_val,:]
        i_val = i_meas[idx_val,:]
        
        # Collocation points
        N_f = 50000
    
#        X_f_train = lb + (ub-lb)*sobol_seq.i4_sobol_generate(2, N_f)        
        X_f_train = lb + (ub-lb)*lhs(2, N_f)        
        
        # add noise
        noise = 0.1
        r_train = r_train + noise*np.std(r_train)*np.random.randn(r_train.shape[0], r_train.shape[1])
        i_train = i_train + noise*np.std(i_train)*np.random.randn(i_train.shape[0], i_train.shape[1])
        r_val = r_val + noise*np.std(r_val)*np.random.randn(r_val.shape[0], r_val.shape[1])
        i_val = i_val + noise*np.std(i_val)*np.random.randn(i_val.shape[0], i_val.shape[1])
        
        X_train = X_train.astype(np.float32)
        r_train = r_train.astype(np.float32)
        i_train = i_train.astype(np.float32)
        X_f_train = X_f_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        r_val = r_val.astype(np.float32)
        i_val = i_val.astype(np.float32)
        
# =============================================================================
#         train model
# =============================================================================
        # inherit eq coeffs(non-zeros) from previous training
        eq_coeff_data = scipy.io.loadmat('DiscLam_ADO.mat')
        
        Lamu_init = eq_coeff_data['Lamu_Disc']
        Lamu_init = np.reshape(Lamu_init[np.nonzero(Lamu_init)], (-1, 1))

        ## Restore model
        model = PhysicsInformedNN(X_train, r_train, i_train, X_f_train, X_val, r_val, i_val, layers, lb, ub, Lamu_init)
        model.saver.restore(model.sess, './saved_variable_ADO')

        ## Pt
        model.Pt()
        
        # save model
        model.saver_pt.save(model.sess, './saved_variable_Pt')
        
# =============================================================================
#         evaluate training efforts
# =============================================================================
        f = open("stdout.txt", "a+") 
        
        r_train_pred, i_train_pred = model.predict(X_train)
        error_r_train = np.linalg.norm(r_train-r_train_pred,2)/np.linalg.norm(r_train,2)        
        error_i_train = np.linalg.norm(i_train-i_train_pred,2)/np.linalg.norm(i_train,2)
        f.write('Training Error r: %e \n' % (error_r_train))    
        f.write('Training Error i: %e \n' % (error_i_train))   
        
        r_val_pred, i_val_pred = model.predict(X_val)
        error_r_val = np.linalg.norm(r_val-r_val_pred,2)/np.linalg.norm(r_val,2)        
        error_i_val = np.linalg.norm(i_val-i_val_pred,2)/np.linalg.norm(i_val,2)
        f.write('Val Error r: %e \n' % (error_i_val))    
        f.write('Val Error i: %e \n' % (error_i_val))
                
        ######################## Plots for Pt #################
        fig = plt.figure()
        plt.plot(loss_history_Pt[1:])
        plt.plot(loss_val_history_Pt[1:])
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of Pt')  
        plt.savefig('6.png')
        
        fig = plt.figure()
        plt.plot(loss_r_history_Pt[1:])
        plt.plot(loss_i_history_Pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history of Pt')  
        plt.savefig('7.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of Pt')     
        plt.savefig('8.png')
        
        fig = plt.figure()
        for i in range(Lambda_history_Pt.shape[0]):
            plt.plot(Lambda_history_Pt[i, 1:])
        plt.title('Lambda_history_Pt')
        plt.savefig('9.png')          

# =============================================================================
#         compare w/ ground truth if training is sufficient
# =============================================================================        
        r_full_pred, i_full_pred = model.predict(X_star)
        error_r_full = np.linalg.norm(r_star-r_full_pred,2)/np.linalg.norm(r_star,2)        
        error_i_full = np.linalg.norm(i_star-i_full_pred,2)/np.linalg.norm(i_star,2)
        f.write('Full Error r: %e \n' % (error_r_full))    
        f.write('Full Error i: %e \n' % (error_i_full))
        
        Lambda_value = model.sess.run(model.Lambda2)        
        Lambda_true = np.zeros((2,1))
        Lambda_true[0] = -0.5 # -0.5u_xx
        Lambda_true[1] = -1 # -|u|**2*u
        error_Lambda = np.linalg.norm(Lambda_true-Lambda_value,2)/np.linalg.norm(Lambda_true,2)
        f.write('Lambda Error: %.2f \n' % (error_Lambda))
        nonzero_ind = np.nonzero(Lambda_true)
        Lambda_error_vector = np.absolute((Lambda_true[nonzero_ind]-Lambda_value[nonzero_ind])/Lambda_true[nonzero_ind])
        error_Lambda_mean = np.mean(Lambda_error_vector)
        error_Lambda_std = np.std(Lambda_error_vector)
        f.write('Lambda Mean Error: %.4f \n' % (error_Lambda_mean))
        f.write('Lambda Std Error: %.4f \n' % (error_Lambda_std))
        
        elapsed = time.time() - start_time                
        f.write('Training time: %.4f \n' % (elapsed))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_descr)):
            if Lambda_value[i_lib] != 0:
                disc_eq_temp.append(str(Lambda_value[i_lib,0]) + model.lib_descr[i_lib])
        disc_eq = '+'.join(disc_eq_temp)        
        f.write('The discovered equation: i*u_t = ' + disc_eq)
    
        # save lambda evolution during training for plotting manuscript figures in MATLAB.
        scipy.io.savemat('LambdaEvolution.mat',{'Lambda_history_Pt':Lambda_history_Pt[:, 1:]}) 
        
        f.close()        
        
        ######################## Plots for Lambda #################
        fig = plt.figure()
        plt.plot(Lambda_value, 'ro-')
        plt.plot(Lambda_true)
        plt.legend(('the pred', 'the true'))
        plt.title('Lambda')
        plt.savefig('26.png')