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
    
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with tf.device('/device:GPU:0'):
    
    # L-BFGS-S loss history (Pretrain)
    loss_history_Pretrain = np.empty([0])
    loss_val_history_Pretrain = np.empty([0])
    loss_r_history_Pretrain = np.empty([0])
    loss_i_history_Pretrain = np.empty([0])
    loss_f_history_Pretrain = np.empty([0])
    loss_Lambda_history_Pretrain = np.empty([0])
    Lambda_history_Pretrain = np.empty([40, 1])
    step_Pretrain = 0
    loss_W_history_Pretrain = np.empty([0])
    
    # Adam loss history (Pretrain)
    loss_history_Adam_Pretrain = np.empty([0])
    loss_val_history_Adam_Pretrain = np.empty([0])
    loss_r_history_Adam_Pretrain = np.empty([0])
    loss_i_history_Adam_Pretrain = np.empty([0])
    loss_f_history_Adam_Pretrain = np.empty([0])
    loss_Lambda_history_Adam_Pretrain = np.empty([0])
    Lambda_history_Adam_Pretrain = np.empty([40, 1])
    loss_W_history_Adam_Pretrain = np.empty([0])
    
    # L-BFGS-S loss history
    loss_history = np.empty([0])
    loss_val_history = np.empty([0])
    loss_r_history = np.empty([0])
    loss_i_history = np.empty([0])
    loss_f_history = np.empty([0])
    loss_Lambda_history = np.empty([0])
    Lambda_history = np.empty([40, 1]) 
    step = 0
    loss_W_history = np.empty([0])
    
    # Adam loss history
    loss_history_Adam = np.empty([0])
    loss_val_history_Adam = np.empty([0])
    loss_r_history_Adam = np.empty([0])
    loss_i_history_Adam = np.empty([0])
    loss_f_history_Adam = np.empty([0])
    loss_Lambda_history_Adam = np.empty([0])
    Lambda_history_Adam = np.empty([40, 1])
    loss_W_history_Adam = np.empty([0])
    
    # STRidge loss histroy
    loss_history_STRidge = np.empty([0])
    loss_f_history_STRidge = np.empty([0])
    loss_Lambda_history_STRidge = np.empty([0])
    tol_history_STRidge = np.empty([0])
    
    Phi_rank_history_STRidge = np.empty([0])
        
    Lambda_history_STRidge = np.empty([40, 1])
    ridge_append_counter_STRidge = np.empty([0])
    
    lib_fun = []
    lib_descr = []
        
    # Alter loss history
    loss_history_Alter = np.empty([0])
    loss_val_history_Alter = np.empty([0])
    loss_r_history_Alter = np.empty([0])
    loss_i_history_Alter = np.empty([0])
    loss_f_history_Alter = np.empty([0])
    loss_Lambda_history_Alter = np.empty([0])
    Lambda_history_Alter = np.empty([40, 1])
    loss_W_history_Alter = np.empty([0])
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, r, imag, X_f, X_val, r_val, imag_val, layers, lb, ub):
            
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
            
            # Specify the list of trainable variables 
            var_list_1 = self.biases + self.weights
            
            var_list_Pretrain = self.biases + self.weights
            var_list_Pretrain.append(self.Lambda)
            
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
            self.f_pred, self.Phi, self.iu_t_pred = self.net_f(self.x_f_tf, self.t_f_tf,
                                                               self.x_f.shape[0])
            
            self.loss_r = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
            self.loss_i = tf.reduce_mean(tf.square(self.i_tf - self.i_pred))
            
            self.loss_f_coeff_tf = tf.placeholder(tf.float32)
            
            self.loss_f = self.loss_f_coeff_tf*tf.reduce_mean(tf.square(tf.abs(self.f_pred))) # the average of square modulus
            
            self.loss_Lambda = 1e-7*tf.norm(self.Lambda, ord=1)  
            
            # L1 norm error for loss_W
            self.loss_W = tf.norm(self.weights[0], ord = 1) # We can't regularize self.biases due to its intialization as zeros.
            for i in range(1, len(self.weights)):
                self.loss_W = self.loss_W + tf.norm(self.weights[i], ord = 1)  
            # self.loss_W = self.loss_W*1e-8
            self.loss_W = self.loss_W*0

            self.loss = tf.log(self.loss_r  + self.loss_i + self.loss_f + self.loss_Lambda)
                        
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
    #         self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
    #                                                                 var_list = var_list_1,
    # #                                                                    L-BFGS-B
    #                                                                 method = 'L-BFGS-B', 
    #                                                                 options = {'maxiter': 1000,
    #                                                                            'maxfun': 1000,
    #                                                                            'maxcor': 50,
    #                                                                            'maxls': 50})
                                                                    
            self.optimizer_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                            var_list = var_list_Pretrain,
            #                                                                    L-BFGS-B
                                                                            method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 160000,
                                                                                       'maxfun': 160000,
                                                                                       'maxcor': 50,
                                                                                       'maxls': 50,
                                                                                       'ftol' : 0.01 * np.finfo(float).eps})
            
            # the default learning rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08
            self.optimizer_Adam = tf.contrib.opt.NadamOptimizer(learning_rate = 5e-4) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = var_list_1)
                        
            self.optimizer_Adam_Pretrain = tf.contrib.opt.NadamOptimizer(learning_rate = 1e-3)
            self.train_op_Adam_Pretrain = self.optimizer_Adam_Pretrain.minimize(self.loss, var_list = var_list_Pretrain)
            
            # Save the model after pretraining
            self.saver = tf.train.Saver(var_list_Pretrain)
            
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
        
        def net_f(self, x, t, N_f):            
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
            
            deri = [u_x, u_xx, u_xxx]
            deri_descr = ['u_x', 'u_xx', 'u_xxx']
            
            # Polynomial terms            
            poly = [u, u**2, u**3]
            poly_descr = ['u', 'u**2', 'u**3']
            
            # Absolute terms
            u_ABS = tf.complex(tf.abs(u), tf.zeros(shape=[N_f, 1]))
            u_ABS_SQ = u_ABS**2
            u_ABS_CUBE = u_ABS**3
            abs_u = [u_ABS, u_ABS_SQ, u_ABS_CUBE]
            abs_u_descr = ['|u|', '|u|**2', '|u|**3']
            
            # Poly-Abs terms
            poly_abs = [u_ABS*u, u_ABS*u**2, u_ABS_SQ*u]
            poly_abs_descr = ['u|u|', 'u**2|u|', 'u|u|**2']
                    
            # Multiplication of Derivative terms and Polynomial terms
            deri_poly = []
            deri_poly_descr = []
            for i in range(len(deri)):
                for j in range(len(poly)):                                        
                    deri_poly.append(deri[i]*poly[j])
                    deri_poly_descr.append(deri_descr[i]+poly_descr[j])
                    
            # Multiplication of Derivative terms and Absolute terms
            deri_abs = []
            deri_abs_descr = []
            for i in range(len(deri)):
                for j in range(len(abs_u)):
                    deri_abs.append(deri[i]*abs_u[j])
                    deri_abs_descr.append(deri_descr[i]+abs_u_descr[j])
                    
            # Multiplication of Derivative terms and Poly-Abs terms
            deri_poly_abs = []
            deri_poly_abs_descr = []
            for i in range(len(deri)):
                for j in range(len(poly_abs)):
                    deri_poly_abs.append(deri[i]*poly_abs[j])
                    deri_poly_abs_descr.append(deri_descr[i]+poly_descr[j])
                    
            global lib_fun
            global lib_descr
            lib_fun = [tf.constant(1, shape=[N_f, 1], dtype=tf.complex64)] + deri + poly + abs_u + \
                        poly_abs + deri_poly + deri_abs + deri_poly_abs
            self.lib_descr = ['1'] + deri_descr + poly_descr + abs_u_descr + poly_abs_descr + \
                        deri_poly_descr + deri_abs_descr + deri_poly_abs_descr
                        
            Phi = tf.transpose(tf.reshape(tf.convert_to_tensor(lib_fun), [40, -1]))            
            f = iu_t - tf.matmul(Phi, tf.complex(self.Lambda, tf.zeros([40, 1], dtype=tf.float32))) 
            
            return f, Phi, iu_t
                    
        def callback_Pretrain(self, loss, loss_r, loss_i, loss_f, loss_Lambda, loss_val, Lambda, loss_W):
            global step_Pretrain
            step_Pretrain += 1
            if step_Pretrain%10 == 0:
                
                global loss_history_Pretrain
                global loss_val_history_Pretrain
                global loss_r_history_Pretrain
                global loss_i_history_Pretrain
                global loss_f_history_Pretrain
                global loss_Lambda_history_Pretrain
                global Lambda_history_Pretrain
                global loss_W_history_Pretrain
                
                loss_history_Pretrain = np.append(loss_history_Pretrain, loss)
                loss_val_history_Pretrain = np.append(loss_val_history_Pretrain, loss_val)
                loss_r_history_Pretrain = np.append(loss_r_history_Pretrain, loss_r)
                loss_i_history_Pretrain = np.append(loss_i_history_Pretrain, loss_i)
                loss_f_history_Pretrain = np.append(loss_f_history_Pretrain, loss_f)
                loss_Lambda_history_Pretrain = np.append(loss_Lambda_history_Pretrain, loss_Lambda)
                Lambda_history_Pretrain = np.append(Lambda_history_Pretrain, Lambda, axis = 1)
                loss_W_history_Pretrain = np.append(loss_W_history_Pretrain, loss_W)
                
        def callback(self, loss, loss_r, loss_i, loss_f, loss_Lambda, loss_val, Lambda, loss_W):
            global step
            step = step+1
            if step%10 == 0:
                
                global loss_history
                global loss_val_history
                global loss_r_history
                global loss_i_history
                global loss_f_history
                global loss_Lambda_history
                global Lambda_history
                global loss_W_history
                
                loss_history = np.append(loss_history, loss)
                loss_val_history = np.append(loss_val_history, loss_val)
                loss_r_history = np.append(loss_r_history, loss_r)
                loss_i_history = np.append(loss_i_history, loss_i)
                loss_f_history = np.append(loss_f_history, loss_f)
                loss_Lambda_history = np.append(loss_Lambda_history, loss_Lambda)
                Lambda_history = np.append(Lambda_history, Lambda, axis = 1)
                loss_W_history = np.append(loss_W_history, loss_W)
                
        def pretrain(self):     
            global loss_history_Adam_Pretrain
            global loss_val_history_Adam_Pretrain
            global loss_r_history_Adam_Pretrain
            global loss_i_history_Adam_Pretrain
            global loss_f_history_Adam_Pretrain
            global loss_Lambda_history_Adam_Pretrain
            global Lambda_history_Adam_Pretrain
            global loss_W_history_Adam_Pretrain
                        
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            
            self.tf_dict[self.loss_f_coeff_tf] = 0.1
            print('Adam(Pretraining) starts')
            for it_Adam in range(80000):
                self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict, options=run_options)
                
                # Print
                if it_Adam % 10 == 0:
                    loss, loss_r, loss_i, loss_f, loss_Lambda, Lambda, loss_val = \
                        self.sess.run([self.loss, self.loss_r, self.loss_i, self.loss_f, self.loss_Lambda, 
                                       self.Lambda, self.loss_val], self.tf_dict)
                    loss_W = self.sess.run(self.loss_W)
                    
                    loss_history_Adam_Pretrain = np.append(loss_history_Adam_Pretrain, loss)
                    loss_val_history_Adam_Pretrain = np.append(loss_val_history_Adam_Pretrain, loss_val)
                    Lambda_history_Adam_Pretrain = np.append(Lambda_history_Adam_Pretrain, Lambda, axis=1)
                    loss_r_history_Adam_Pretrain = np.append(loss_r_history_Adam_Pretrain, loss_r)
                    loss_i_history_Adam_Pretrain = np.append(loss_i_history_Adam_Pretrain, loss_i)
                    loss_f_history_Adam_Pretrain = np.append(loss_f_history_Adam_Pretrain, loss_f)
                    loss_Lambda_history_Adam_Pretrain = np.append(loss_Lambda_history_Adam_Pretrain, loss_Lambda)
                    loss_W_history_Adam_Pretrain = np.append(loss_W_history_Adam_Pretrain, loss_W)
                    
    #                # L-BFGS-B optimizer(Pretraining)
            print('L-BFGS-B(Pretraining) starts')
            self.optimizer_Pretrain.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss, self.loss_r, self.loss_i, self.loss_f, 
                                               self.loss_Lambda, self.loss_val, self.Lambda, self.loss_W],
                                    loss_callback = self.callback_Pretrain)
                        
        def ASO(self, nIter):
            global loss_history_Adam
            global loss_val_history_Adam
            global loss_r_history_Adam
            global loss_i_history_Adam
            global loss_f_history_Adam
            global loss_Lambda_history_Adam
            global Lambda_history_Adam
            global loss_W_history_Adam
            
            global loss_history_Alter
            global loss_val_history_Alter 
            global loss_r_history_Alter 
            global loss_i_history_Alter 
            global loss_f_history_Alter 
            global loss_Lambda_history_Alter  
            global Lambda_history_Alter 
            global loss_W_history_Alter
            
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            self.tf_dict[self.loss_f_coeff_tf] = 0.5
            for self.it in range(nIter):    
                # Adam optimizer                         
                print('Adam starts')
                for it_Adam in range(1000):
                    self.sess.run(self.train_op_Adam, self.tf_dict, options=run_options)
                    
                    # Print
                    if it_Adam % 10 == 0:
                        loss, loss_r, loss_i, loss_f, loss_Lambda, Lambda, loss_val = \
                            self.sess.run([self.loss, self.loss_r, self.loss_i, self.loss_f, self.loss_Lambda, 
                                       self.Lambda, self.loss_val], self.tf_dict)
                        loss_W = self.sess.run(self.loss_W)
                        
                        loss_history_Adam = np.append(loss_history_Adam, loss)
                        loss_val_history_Adam = np.append(loss_val_history_Adam, loss_val)
                        loss_r_history_Adam = np.append(loss_r_history_Adam, loss_r)
                        loss_i_history_Adam = np.append(loss_i_history_Adam, loss_i)
                        loss_f_history_Adam = np.append(loss_f_history_Adam, loss_f)
                        loss_Lambda_history_Adam = np.append(loss_Lambda_history_Adam, loss_Lambda)
                        Lambda_history_Adam = np.append(Lambda_history_Adam, Lambda, axis = 1)
                        loss_W_history_Adam = np.append(loss_W_history_Adam, loss_W)
    
                    # if it_Adam % 999 == 0:
                    #     loss, loss_r, loss_i, loss_f, loss_Lambda, Lambda, loss_val = \
                    #         self.sess.run([self.loss, self.loss_r, self.loss_i, self.loss_f, self.loss_Lambda, 
                    #                    self.Lambda, self.loss_val], self.tf_dict)
                    #     loss_W = self.sess.run(self.loss_W)
                        
                    #     loss_history_Alter  = np.append(loss_history_Alter, loss)
                    #     loss_val_history_Alter = np.append(loss_val_history_Alter, loss_val)
                    #     loss_r_history_Alter = np.append(loss_r_history_Alter, loss_r)
                    #     loss_i_history_Alter = np.append(loss_i_history_Alter, loss_i)
                    #     loss_f_history_Alter = np.append(loss_f_history_Alter, loss_f)
                    #     loss_Lambda_history_Alter = np.append(loss_Lambda_history_Alter, loss_Lambda)
                    #     Lambda_history_Alter = np.append(Lambda_history_Alter, Lambda, axis = 1)
                    #     loss_W_history_Alter = np.append(loss_W_history_Alter, loss_W)
                                                                  
                # L-BFGS-B optimizer
                # print('L-BFGS-B starts')
                # self.optimizer.minimize(self.sess,
                #                         feed_dict = self.tf_dict,
                #                         fetches = [self.loss, self.loss_r, self.loss_i, self.loss_f, 
                #                                    self.loss_Lambda, self.loss_val, self.Lambda, self.loss_W],
                #                         loss_callback = self.callback)
                
                # STRidge optimizer
                print('STRidge starts')
                self.callTrainSTRidge()
                
        def predict(self, X_star):            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}            
            r_star = self.sess.run(self.r_pred, tf_dict)
            i_star = self.sess.run(self.i_pred, tf_dict)
            return r_star, i_star
        
        def callTrainSTRidge(self):
            lam = 1e-5
            d_tol = 100
            maxit = 100    
            
            # Process of Lambda            
            Phi, iu_t_pred = self.sess.run([self.Phi, self.iu_t_pred], self.tf_dict) 
            Lambda2, loss_history_STRidge2, loss_f_history_STRidge2, loss_Lambda_history_STRidge2, \
                tol_history_STRidge2 = self.TrainSTRidge(Phi, iu_t_pred, lam, d_tol, maxit)
                
            self.Lambda = tf.assign(self.Lambda, tf.convert_to_tensor(Lambda2, dtype = tf.float32))
            
            rank_Phi = np.linalg.matrix_rank(Phi)
            global Phi_rank_history_STRidge
            Phi_rank_history_STRidge = np.append(Phi_rank_history_STRidge, rank_Phi)
                        
            global loss_history_STRidge
            global loss_f_history_STRidge
            global loss_Lambda_history_STRidge
            global tol_history_STRidge
            
            loss_history_STRidge = np.append(loss_history_STRidge, loss_history_STRidge2)
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, loss_f_history_STRidge2)
            loss_Lambda_history_STRidge = np.append(loss_Lambda_history_STRidge, loss_Lambda_history_STRidge2)
            tol_history_STRidge = np.append(tol_history_STRidge, tol_history_STRidge2)
            
        def TrainSTRidge(self, R0, Ut, lam, d_tol, maxit, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8):                        
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================      
            n,d = R0.shape
            R = np.zeros((n,d), dtype = np.complex64)
            # First normalize data
            if normalize != 0:
                Mreg = np.zeros((d,1))
                for i in range(0,d):
                    Mreg[i] = 1.0/(np.linalg.norm(np.abs(R0[:,i]),normalize)) # L2 norm
                    # Mreg[i] = 1.0/np.amax(np.abs(R0[:,i])) # infinity norm
                    R[:,i] = Mreg[i]*R0[:,i]
                    
            else: R = R0
            
            # normalize Ut
            # Mreg_Phi = Mreg
            # Mreg_Ut = 1/np.amax(np.abs(Ut))
            # Ut0 = Ut
            # Ut = Ut*Mreg_Ut
            # Mreg = Mreg_Phi*Mreg_Ut

            # Split data into 80% training and 20% test, then search for the best tolderance.
            np.random.seed(0) # for consistancy
            n,_ = R.shape
            train = np.random.choice(n, int(n*split), replace = False)
            test = [i for i in np.arange(n) if i not in train]
            TrainR = R[train,:]
            TestR = R[test,:]
            TrainY = Ut[train,:]
            TestY = Ut[test,:]
            D = TrainR.shape[1]       
        
            # TestR0 = R0[test,:]
            # TestY0 = Ut0[test,:]

            # Set up the initial tolerance and l0 penalty
            d_tol = float(d_tol)
            tol = d_tol            
            
            w_best = self.sess.run(self.Lambda)/Mreg + 0*1j
                
            # err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
            err_f = np.mean((np.abs(TestY - TestR.dot(w_best)))**2)            
            
            # err_f = np.mean((np.abs(TestY0 - TestR0.dot(w_best*Mreg)))**2)            

            if l0_penalty == None and self.it == 0: 
                # l0_penalty = 0.05*np.linalg.cond(R)
                self.l0_penalty_0 = 100*err_f
                l0_penalty = self.l0_penalty_0
            elif l0_penalty == None:
                l0_penalty = self.l0_penalty_0

            err_Lambda = l0_penalty*np.count_nonzero(w_best)
            err_best = err_f + err_Lambda
            tol_best = 0
                        
            loss_history_STRidge = np.empty([0])
            loss_f_history_STRidge = np.empty([0])
            loss_Lambda_history_STRidge = np.empty([0])
            tol_history_STRidge = np.empty([0])
            loss_history_STRidge = np.append(loss_history_STRidge, err_best)
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
            loss_Lambda_history_STRidge = np.append(loss_Lambda_history_STRidge, err_Lambda)
            tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
        
            # Now increase tolerance until test performance decreases
            for iter in range(maxit):
        
                # Get a set of coefficients and error
                w = self.STRidge(TrainR, TrainY, lam, STR_iters, tol, Mreg)    
                
                # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
                
                err_f = np.mean((np.abs(TestY - TestR.dot(w)))**2)            
                err_Lambda = l0_penalty*np.count_nonzero(w)
                err = err_f + err_Lambda
                
                # err_f = np.mean((np.abs(TestY0 - TestR0.dot(w*Mreg)))**2)            
                # err_Lambda = l0_penalty*np.count_nonzero(w)
                # err = err_f + err_Lambda
                
                # Has the accuracy improved?
                if err <= err_best:
                    err_best = err
                    w_best = w
                    
                    tol_best = tol
                    tol = tol + d_tol
                    
                    loss_history_STRidge = np.append(loss_history_STRidge, err_best)
                    loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
                    loss_Lambda_history_STRidge = np.append(loss_Lambda_history_STRidge, err_Lambda)
                    tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
        
                else:
                    
                    tol = max([0,tol - 2*d_tol])
                    d_tol = d_tol/1.618
                    tol = tol + d_tol
                            
            w_best = w_best*Mreg
            
            return np.real(w_best), loss_history_STRidge, loss_f_history_STRidge, loss_Lambda_history_STRidge, tol_history_STRidge
        
        def STRidge(self, X0, y, lam, maxit, tol, Mreg, normalize = 2, print_results = False):      
        
            n,d = X0.shape
            
            X = X0
                        
            # Inherit w from previous trainning
            w = self.sess.run(self.Lambda)/Mreg + 0*1j
            
            num_relevant = d
            biginds = np.where(abs(w) > tol)[0]
            
            global ridge_append_counter_STRidge
            ridge_append_counter = 0
            
            global Lambda_history_STRidge
            Lambda_history_STRidge = np.append(Lambda_history_STRidge, np.multiply(Mreg, w), axis = 1)
            ridge_append_counter += 1
                
            # Threshold and continue
            for j in range(maxit):
        
                # Figure out which items to cut out
                smallinds = np.where(abs(w) < tol)[0]
                new_biginds = [i for i in range(d) if i not in smallinds]
                    
                # If nothing changes then stop
                if num_relevant == len(new_biginds): break
                else: num_relevant = len(new_biginds)
                    
                if len(new_biginds) == 0:
                    if j == 0: 
                        
                        if normalize != 0: 
                            Lambda_history_STRidge = np.append(Lambda_history_STRidge, w*Mreg, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return w
                        else:                             
                            Lambda_history_STRidge = np.append(Lambda_history_STRidge, w*Mreg, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return w
                    else: break
                biginds = new_biginds
                
                # Otherwise get a new guess
                w[smallinds] = 0
                
                if lam != 0: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
                    Lambda_history_STRidge = np.append(Lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
                    ridge_append_counter += 1
                else: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
                    Lambda_history_STRidge = np.append(Lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
                    ridge_append_counter += 1
                        
        
            # Now that we have the sparsity pattern, use standard least squares to get w
            if biginds != []: 
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            
            if normalize != 0: 
                Lambda_history_STRidge = np.append(Lambda_history_STRidge, w*Mreg, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return w
            else:
                Lambda_history_STRidge = np.append(Lambda_history_STRidge, w*Mreg, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return w
        
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
        ## Pretrain
        model = PhysicsInformedNN(X_train, r_train, i_train, X_f_train, X_val, r_val, i_val, layers, lb, ub)
        model.pretrain()
        
        # Checkpoint model        
        saved_path = model.saver.save(model.sess, './saved_variable')
        # print('model saved in {}'.format(saved_path))                
        
        ## Restore pretrained model
#        # in case we need to create graph 
        # model = PhysicsInformedNN(X_train, r_train, i_train, X_f_train, X_val, r_val, i_val, layers, lb, ub)
#        # restore the saved vairable
        # model.saver.restore(model.sess, './saved_variable')


        # Phi, iu_t_pred = model.sess.run([model.Phi, model.iu_t_pred], model.tf_dict) 
        # Mreg = np.zeros((Phi.shape[1],1))
        # for i in range(0, Phi.shape[1]):
        #     Mreg[i] = 1.0/np.amax(np.abs(Phi[:,i])) # infinity norm
            
        # # normalize Ut
        # Mreg_Phi = Mreg
        # Mreg_Ut = 1/np.amax(np.abs(iu_t_pred))
        # Mreg = Mreg_Phi*Mreg_Ut
            
        # Lambda_value = model.sess.run(model.Lambda)        

        # scipy.io.savemat('Phi_ut.mat',{'Phi':Phi,
        #                                'iu_t_pred':iu_t_pred,
        #                                'Mreg':Mreg,
        #                                'Lambda_value':Lambda_value}) 

        ## ASO
        model.ASO(6)
        
        # Checkpoint model        
        saved_path = model.saver.save(model.sess, './saved_variable_ADO')
        
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

        ######################## Plots for Adam(pretrain) #################
        fig = plt.figure()
        plt.plot(loss_history_Adam_Pretrain[1:])
        plt.plot(loss_val_history_Adam_Pretrain[1:])
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of Adam_Pretrain')
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(loss_r_history_Adam_Pretrain[1:])
        plt.plot(loss_i_history_Adam_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history of Adam_Pretrain')  
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Adam_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of Adam_Pretrain') 
        plt.savefig('3.png')
                
        fig = plt.figure()
        plt.plot(loss_Lambda_history_Adam_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_Lambda history of Adam_Pretrain')  
        plt.savefig('4.png')
        
        fig = plt.figure()
        plt.plot(loss_W_history_Adam_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_W history of Adam_Pretrain')  
        plt.savefig('5.png')
                
        ######################## Plots for BFGS(pretrain) #################
        fig = plt.figure()
        plt.plot(loss_history_Pretrain[1:])
        plt.plot(loss_val_history_Pretrain[1:])
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of BFGS_Pretrain')  
        plt.savefig('6.png')
        
        fig = plt.figure()
        plt.plot(loss_r_history_Pretrain[1:])
        plt.plot(loss_i_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history of BFGS_Pretrain')  
        plt.savefig('7.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS_Pretrain')     
        plt.savefig('8.png')
        
        fig = plt.figure()
        plt.plot(loss_Lambda_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_Lambda history of BFGS_Pretrain')  
        plt.savefig('9.png')
        
        fig = plt.figure()
        plt.plot(loss_W_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_W history of BFGS_Pretrain')  
        plt.savefig('10.png')
        
        ######################## Plots for Adam #################
        fig = plt.figure()
        plt.plot(loss_history_Adam[1:])
        plt.plot(loss_val_history_Adam[1:])
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of Adam')
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(loss_r_history_Adam[1:])
        plt.plot(loss_i_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history of Adam')  
        plt.savefig('12.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of Adam') 
        plt.savefig('13.png')
                
        fig = plt.figure()
        plt.plot(loss_Lambda_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_Lambda history of Adam')  
        plt.savefig('14.png')
        
        fig = plt.figure()
        plt.plot(loss_W_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_W history of Adam')  
        plt.savefig('15.png')
                
        ######################## Plots for BFGS #################
        fig = plt.figure()
        plt.plot(loss_history[1:])
        plt.plot(loss_val_history[1:])
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of BFGS')  
        plt.savefig('16.png')
        
        fig = plt.figure()
        plt.plot(loss_r_history[1:])
        plt.plot(loss_i_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history of BFGS')  
        plt.savefig('17.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS')     
        plt.savefig('18.png')
        
        fig = plt.figure()
        plt.plot(loss_Lambda_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_Lambda history of BFGS')  
        plt.savefig('19.png')
        
        fig = plt.figure()
        plt.plot(loss_W_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_W history of BFGS')  
        plt.savefig('20.png')
        
        ######################## Plots for STRidge #################     
        fig = plt.figure()
        plt.plot(loss_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('History of STRidge(loss_f+loss_lambda )')  
        plt.savefig('21.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f history of STRidge') 
        plt.savefig('22.png')
        
        fig = plt.figure()
        plt.plot(loss_Lambda_history_STRidge[1:])
        plt.yscale('log')      
        plt.title('loss_Lambda history of STRidge')  
        plt.savefig('23.png')
        
        fig = plt.figure()
        plt.plot(tol_history_STRidge[1:])
        plt.title('tol history of STRidge')  
        plt.savefig('24.png')
        
        fig = plt.figure()
        plt.plot(Phi_rank_history_STRidge[1:])
        plt.title('rank history of STRidge')  
        plt.savefig('25.png')

# =============================================================================
#         compare w/ ground truth if training is sufficient
# =============================================================================        
        r_full_pred, i_full_pred = model.predict(X_star)
        error_r_full = np.linalg.norm(r_star-r_full_pred,2)/np.linalg.norm(r_star,2)        
        error_i_full = np.linalg.norm(i_star-i_full_pred,2)/np.linalg.norm(i_star,2)
        f.write('Full Error r: %e \n' % (error_r_full))    
        f.write('Full Error i: %e \n' % (error_i_full))
        
        Lambda_value = model.sess.run(model.Lambda)        
        Lambda_true = np.zeros((40,1))
        Lambda_true[2] = -0.5 # -0.5u_xx
        Lambda_true[12] = -1 # -|u|**2*u
        cosine_similarity = 1-distance.cosine(Lambda_true, Lambda_value)
        f.write('Cosine similarity of Lambda: %.2f \n' % (cosine_similarity))
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
        scipy.io.savemat('LambdaEvolution.mat',{'Lambda_history_Pretrain':Lambda_history_Pretrain[:, 1:],
                                                'Lambda_history_Adam_Pretrain':Lambda_history_Adam_Pretrain[:,1:],
                                                'Lambda_history_STRidge':Lambda_history_STRidge[:,1:],
                                                'Lambda_history_Alter':Lambda_history_Alter[:,1:],
                                                'ridge_append_counter_STRidge': ridge_append_counter_STRidge[1:]}) 
        
        f.close()        
        
        ######################## Plots for Lambda #################
        fig = plt.figure()
        plt.plot(Lambda_value, 'ro-')
        plt.plot(Lambda_true)
        plt.legend(('the pred', 'the true'))
        plt.title('Lambda')
        plt.savefig('26.png')