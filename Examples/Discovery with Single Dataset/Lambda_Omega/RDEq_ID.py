# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Please run RD_Pretrain_ADO and RDEq_ID in order.
# =============================================================================

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
#import sobol_seq
from pyDOE import lhs
    
with tf.device('/device:GPU:1'):        
    # Adam loss history(NN Weights)
    loss_history_Adam_Pretrain = np.empty([0])
    loss_val_history_Adam_Pretrain = np.empty([0])
    loss_u_history_Adam_Pretrain = np.empty([0])
    loss_v_history_Adam_Pretrain = np.empty([0])
    loss_f_u_history_Adam_Pretrain = np.empty([0])
    loss_f_v_history_Adam_Pretrain = np.empty([0])
    
    # L-BFGS-B loss history(NN Weights)
    loss_history_Pretrain = np.empty([0])
    loss_val_history_Pretrain = np.empty([0])
    loss_u_history_Pretrain = np.empty([0])
    loss_v_history_Pretrain = np.empty([0])
    loss_f_u_history_Pretrain = np.empty([0])
    loss_f_v_history_Pretrain = np.empty([0]) 
    step_Pretrain = 0
    
    # L-BFGS-S loss history
    loss_history = np.empty([0])
    loss_val_history = np.empty([0])
    loss_u_history = np.empty([0])
    loss_v_history = np.empty([0])
    loss_f_u_history = np.empty([0])
    loss_f_v_history = np.empty([0])
    step = 0
    
    # Adam loss history
    loss_history_Adam = np.empty([0])
    loss_val_history_Adam = np.empty([0])
    loss_u_history_Adam = np.empty([0])
    loss_v_history_Adam = np.empty([0])
    loss_f_u_history_Adam = np.empty([0])
    loss_f_v_history_Adam = np.empty([0])
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, v, X_f, X_val, u_val, v_val, layers, lb, ub, BatchNo):
            
            self.lb = lb
            self.ub = ub
            self.layers = layers
            
            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters based on ADO results
            self.lambda_u2 = tf.Variable(0.0888854563236237, dtype=tf.float32)
            self.lambda_u4 = tf.Variable(0.0921253487467766, dtype=tf.float32)
            self.lambda_u11 = tf.Variable(0.785800397396088, dtype=tf.float32)
            self.lambda_u33 = tf.Variable(-0.767779946327210, dtype=tf.float32)
            self.lambda_u66 = tf.Variable(0.999720275402069, dtype=tf.float32)
            self.lambda_u88 = tf.Variable(-0.774374604225159, dtype=tf.float32)
            self.lambda_u99 = tf.Variable(0.996893763542175, dtype=tf.float32)
            self.lambda_u = [self.lambda_u2, self.lambda_u4, self.lambda_u11, self.lambda_u33, self.lambda_u66, self.lambda_u88,
                             self.lambda_u99]
            
            self.lambda_v7 = tf.Variable(0.0922485068440437, dtype=tf.float32)
            self.lambda_v9 = tf.Variable(0.0898747444152832, dtype=tf.float32)
            self.lambda_v33 = tf.Variable(-0.995353758335114, dtype=tf.float32)
            self.lambda_v44 = tf.Variable(0.821019947528839, dtype=tf.float32)
            self.lambda_v66 = tf.Variable(-0.811196208000183, dtype=tf.float32)
            self.lambda_v88 = tf.Variable(-0.996607720851898, dtype=tf.float32)
            self.lambda_v99 = tf.Variable(-0.804214298725128, dtype=tf.float32)
            self.lambda_v = [self.lambda_v7, self.lambda_v9, self.lambda_v33, self.lambda_v44, self.lambda_v66, self.lambda_v88, 
                             self.lambda_v99]
            
            # Specify the list of trainable variables 
            var_list_Pretrain = self.biases + self.weights
            
            ######### Training data ################
            self.x = X[:,0:1]
            self.y = X[:,1:2]
            self.t = X[:,2:3]
            self.u = u
            self.v = v
            # Collocation points
            self.x_f = X_f[:,0:1]
            self.y_f = X_f[:,1:2]
            self.t_f = X_f[:,2:3]
            
            self.BatchNo = BatchNo
            self.batchsize_train = np.floor(self.x.shape[0]/self.BatchNo)
            self.batchsize_f = np.floor(self.x_f.shape[0]/self.BatchNo)            
            
            self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
            self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
            self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
            self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
            self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
            
            self.u_pred, self.v_pred = self.net_uv(self.x_tf, self.y_tf, self.t_tf)
            self.f_u_pred, self.f_v_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf, self.batchsize_f)
            
            self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
            self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
            self.loss_f_u = tf.reduce_mean(tf.square(self.f_u_pred))
            self.loss_f_v = tf.reduce_mean(tf.square(self.f_v_pred))
            
            self.loss = tf.log(self.loss_u  + self.loss_v + self.loss_f_u + self.loss_f_v) 
                        
            ######### Validation data ################
            self.x_val = X_val[:,0:1]
            self.y_val = X_val[:,1:2]
            self.t_val = X_val[:,2:3]
            self.u_val = u_val
            self.v_val = v_val
            
            self.batchsize_val = np.floor(self.x_val.shape[0]/self.BatchNo)
            
            self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
            self.y_val_tf = tf.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
            self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
            self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
            self.v_val_tf = tf.placeholder(tf.float32, shape=[None, self.v_val.shape[1]])
            
            self.u_val_pred, self.v_val_pred = self.net_uv(self.x_val_tf, self.y_val_tf, self.t_val_tf)
    
            self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
            self.loss_v_val = tf.reduce_mean(tf.square(self.v_val_tf - self.v_val_pred))
            self.loss_val = tf.log(self.loss_u_val  + self.loss_v_val)     
            
            ######### Optimizor #########################
            self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 20000,
                                                                               'maxfun': 20000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 0.1 * np.finfo(float).eps})
        
            self.optimizer_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                            var_list = var_list_Pretrain,
                                                                            method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 20000,
                                                                                       'maxfun': 20000,
                                                                                       'maxcor': 50,
                                                                                       'maxls': 50,
                                                                                       'ftol' : 0.1 * np.finfo(float).eps})
            
            self.optimizer_Adam = tf.contrib.opt.NadamOptimizer(learning_rate = 0.001) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                        
            self.optimizer_Adam1_Pretrain = tf.contrib.opt.NadamOptimizer(learning_rate = 0.001)
            self.train_op_Adam1_Pretrain = self.optimizer_Adam1_Pretrain.minimize(self.loss, var_list = var_list_Pretrain)
                        
            init = tf.global_variables_initializer()
            self.sess.run(init)
    
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
                     
        def net_uv(self, x, y, t):  
            Y = self.neural_net(tf.concat([x,y,t],1), self.weights, self.biases)
            u = Y[:, 0:1]
            v = Y[:, 1:2]
            return u,v
        
        def net_f(self, x, y, t, N_f):       
            u, v = self.net_uv(x,y,t)
                        
            u_x = tf.gradients(u,x)[0]
            u_xx = tf.gradients(u_x,x)[0]
            u_y = tf.gradients(u,y)[0]
            u_yy = tf.gradients(u_y,y)[0]
            u_t = tf.gradients(u,t)[0]
            
            v_x = tf.gradients(v,x)[0]
            v_xx = tf.gradients(v_x,x)[0]
            v_y = tf.gradients(v,y)[0]
            v_yy = tf.gradients(v_y,y)[0]
            v_t = tf.gradients(v,t)[0]
                        
            lib_fun_u = [u_xx, u_yy, u, u**3, v**3, u*v**2, u**2*v]
            lib_fun_v = [v_xx, v_yy, u**3, v, v**3, u*v**2, u**2*v]
            
            self.lib_fun_u_descr = ['u_xx', 'u_yy', 'u', 'u**3', 'v**3', 'u*v**2', 'u**2*v']
            self.lib_fun_v_descr = ['v_xx', 'v_yy', 'u**3', 'v', 'v**3', 'u*v**2', 'u**2*v']

            f_u = u_t
            f_v = v_t
            for i in range(len(lib_fun_u)):
                f_u = f_u - lib_fun_u[i]*self.lambda_u[i] # Note that the minus sign instead of the plus sign is uesd here.
                f_v = f_v - lib_fun_v[i]*self.lambda_v[i] # Note that the minus sign instead of the plus sign is uesd here.
                                    
            return f_u, f_v
                            
        def callback(self, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val):
            global step
            step = step+1
            if step%100 == 0:
                print('It: %d, log Loss: %e, loss_u: %e, loss_v: %e, loss_f_u: %e, loss_f_v: %e, log loss_val: %e' % 
                      (step, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val))
                
                global loss_history
                global loss_val_history
                global loss_u_history
                global loss_v_history
                global loss_f_u_history
                global loss_f_v_history
                
                loss_history = np.append(loss_history, loss)
                loss_val_history = np.append(loss_val_history, loss_val)
                loss_u_history = np.append(loss_u_history, loss_u)
                loss_v_history = np.append(loss_v_history, loss_v)
                loss_f_u_history = np.append(loss_f_u_history, loss_f_u)
                loss_f_v_history = np.append(loss_f_v_history, loss_f_v)
                
        def callback_Pretrain(self, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val):
            global step_Pretrain
            step_Pretrain += 1
            if step_Pretrain % 100 == 0:
                print('It: %d, log Loss: %e, loss_u: %e, loss_v: %e, loss_f_u: %e, loss_f_v: %e, log loss_val: %e' % 
                      (step_Pretrain, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val))
                
                global loss_history_Pretrain
                global loss_val_history_Pretrain
                global loss_u_history_Pretrain
                global loss_v_history_Pretrain
                global loss_f_u_history_Pretrain
                global loss_f_v_history_Pretrain
                
                loss_history_Pretrain = np.append(loss_history_Pretrain, loss)
                loss_val_history_Pretrain = np.append(loss_val_history_Pretrain, loss_val)
                loss_u_history_Pretrain = np.append(loss_u_history_Pretrain, loss_u)
                loss_v_history_Pretrain = np.append(loss_v_history_Pretrain, loss_v)
                loss_f_u_history_Pretrain = np.append(loss_f_u_history_Pretrain, loss_f_u)
                loss_f_v_history_Pretrain = np.append(loss_f_v_history_Pretrain, loss_f_v)
            
        def train(self):       
            # With batches
            for i in range(self.BatchNo):                
                x_batch = self.x[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                y_batch = self.y[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                t_batch = self.t[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                u_batch = self.u[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                v_batch = self.v[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                
                x_f_batch = self.x_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                y_f_batch = self.y_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                t_f_batch = self.t_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                
                x_val_batch = self.x_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                y_val_batch = self.y_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                t_val_batch = self.t_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                u_val_batch = self.u_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                v_val_batch = self.v_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                
                self.tf_dict = {self.x_tf: x_batch, self.y_tf: y_batch, self.t_tf: t_batch, self.u_tf: u_batch, self.v_tf: v_batch,
                                self.x_f_tf: x_f_batch, self.y_f_tf: y_f_batch, self.t_f_tf: t_f_batch, 
                                self.x_val_tf: x_val_batch, self.y_val_tf: y_val_batch, self.t_val_tf: t_val_batch,
                                self.u_val_tf: u_val_batch, self.v_val_tf: v_val_batch}
            
                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                # Adam1 optimizer(Pretraining)
                print('Adam1(Pretraining) starts')
                start_time = time.time()
                for it_Adam in range(20000):
                    self.sess.run(self.train_op_Adam1_Pretrain, self.tf_dict, options=run_options)
                    
                    # Print
                    if it_Adam % 100 == 0:
                        elapsed = time.time() - start_time
                        loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val = \
                            self.sess.run([self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_val], self.tf_dict)
                        print('It: %d, Log Loss: %e, loss_u: %e, loss_v: %e, loss_f_u: %e, loss_f_v: %e, loss_val: %e, Time: %.2f' 
                              % (it_Adam, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val, elapsed))
                        
                        global loss_history_Adam_Pretrain
                        global loss_val_history_Adam_Pretrain
                        global loss_u_history_Adam_Pretrain
                        global loss_v_history_Adam_Pretrain
                        global loss_f_u_history_Adam_Pretrain
                        global loss_f_v_history_Adam_Pretrain
                        
                        loss_history_Adam_Pretrain = np.append(loss_history_Adam_Pretrain, loss)
                        loss_val_history_Adam_Pretrain = np.append(loss_val_history_Adam_Pretrain, loss_val)
                        loss_u_history_Adam_Pretrain = np.append(loss_u_history_Adam_Pretrain, loss_u)
                        loss_v_history_Adam_Pretrain = np.append(loss_v_history_Adam_Pretrain, loss_v)
                        loss_f_u_history_Adam_Pretrain = np.append(loss_f_u_history_Adam_Pretrain, loss_f_u)
                        loss_f_v_history_Adam_Pretrain = np.append(loss_f_v_history_Adam_Pretrain, loss_f_v)
                
                        start_time = time.time()
                                                
                # L-BFGS-B optimizer(Pretraining)
                print('L-BFGS-B(Pretraining) starts')
                self.optimizer_Pretrain.minimize(self.sess,
                                        feed_dict = self.tf_dict,
                                        fetches = [self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_val],
                                        loss_callback = self.callback_Pretrain)
                
                # Adam optimizer                
                print('Adam starts')
                start_time = time.time()
                for it_Adam in range(20000):
                    self.sess.run(self.train_op_Adam, self.tf_dict, options=run_options)
                    
                    # Print
                    if it_Adam % 100 == 0:
                        elapsed = time.time() - start_time
                        loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val = self.sess.run([self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_val], self.tf_dict)
                        print('It: %d, Log Loss: %e, loss_u: %e, loss_v: %e, loss_f_u: %e, loss_f_v: %e, loss_val: %e, Time: %.2f' 
                              % (it_Adam, loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_val, elapsed))
                        
                        global loss_history_Adam
                        global loss_val_history_Adam
                        global loss_u_history_Adam
                        global loss_v_history_Adam
                        global loss_f_u_history_Adam
                        global loss_f_v_history_Adam
                        
                        loss_history_Adam = np.append(loss_history_Adam, loss)
                        loss_val_history_Adam = np.append(loss_val_history_Adam, loss_val)
                        loss_u_history_Adam = np.append(loss_u_history_Adam, loss_u)
                        loss_v_history_Adam = np.append(loss_v_history_Adam, loss_v)
                        loss_f_u_history_Adam = np.append(loss_f_u_history_Adam, loss_f_u)
                        loss_f_v_history_Adam = np.append(loss_f_v_history_Adam, loss_f_v)
                
                        start_time = time.time()
                        
                 # L-BFGS-B optimizer
                print('L-BFGS-B1 starts')
                self.optimizer1.minimize(self.sess,
                                        feed_dict = self.tf_dict,
                                        fetches = [self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_val],
                                        loss_callback = self.callback)
                                        
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.y_tf: X_star[:,1:2], self.t_tf: X_star[:,2:3]}
            
            u_star = self.sess.run(self.u_pred, tf_dict)
            v_star = self.sess.run(self.v_pred, tf_dict)
            return u_star, v_star           
        
    if __name__ == "__main__": 
         
        
        start_time = time.time()
        
        layers = [3] + 8*[60] + [2]
        
# =============================================================================
#         load data
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
                    
        # Measurements: Spatially random 
        N_uv_s = 2500 # The total measurement points are 5000*30, about 1.14% of the total grid numbers
        
        # Use these commands when N_uv_s is larger than X.shape[0] or X.shape[1]
        idx = np.random.choice(X.shape[0]*X.shape[1], N_uv_s, replace = False)
        idx_remainder = idx%(X.shape[0])
        idx_s_y = np.floor(idx/(X.shape[0]))
        idx_s_y = idx_s_y.astype(np.int32)
        idx_idx_remainder = np.where(idx_remainder == 0)[0]
        idx_remainder[idx_idx_remainder] = X.shape[0]
        idx_s_x = idx_remainder-1            
                
        # Random sample temporally
        N_t_s = 15 # Original value is 20Hz.
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
        
        # Training measurements, which are randomly sampled spatio-temporally
        Split_TrainVal = 0.8
        N_u_train = int(N_uv_s*N_t_s*Split_TrainVal)
        idx_train = np.random.choice(X_star_meas.shape[0], N_u_train, replace=False)
        X_star_train = X_star_meas[idx_train,:]
        u_star_train = u_star_meas[idx_train,:]
        v_star_train = v_star_meas[idx_train,:]
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_star_meas.shape[0]), idx_train, assume_unique=True)
        X_star_val = X_star_meas[idx_val,:]
        u_star_val = u_star_meas[idx_val,:]
        v_star_val = v_star_meas[idx_val,:]
                
        # Collocation points
        N_f = 40000
#        X_f = lb + (ub-lb)*sobol_seq.i4_sobol_generate(3, N_f)
        X_f = lb + (ub-lb)*lhs(3, N_f)
#        X_f = np.vstack((X_f, X_star_train))
         
        # add noise
        noise = 0.1
        u_star_train = u_star_train + noise*np.std(u_star_train)*np.random.randn(u_star_train.shape[0], u_star_train.shape[1])
        v_star_train = v_star_train + noise*np.std(v_star_train)*np.random.randn(v_star_train.shape[0], v_star_train.shape[1])
        u_star_val = u_star_val + noise*np.std(u_star_val)*np.random.randn(u_star_val.shape[0], u_star_val.shape[1])
        v_star_val = v_star_val + noise*np.std(v_star_val)*np.random.randn(v_star_val.shape[0], v_star_val.shape[1])
                
        BatchNo = 1
        
        X_star_train = X_star_train.astype(np.float32)
        u_star_train = u_star_train.astype(np.float32)
        v_star_train = v_star_train.astype(np.float32)
        X_f = X_f.astype(np.float32)
        X_star_val = X_star_val.astype(np.float32)
        u_star_val = u_star_val.astype(np.float32)
        v_star_val = v_star_val.astype(np.float32)
        
# =============================================================================
#         train model
# =============================================================================
        model = PhysicsInformedNN(X_star_train, u_star_train, v_star_train, X_f, X_star_val, u_star_val, v_star_val, layers, lb, ub, BatchNo)
        model.train()
        
# =============================================================================
#       check if training efforts are sufficient                    
# =============================================================================
        u_train_pred, v_train_pred = model.predict(X_star_train)
        error_u_train = np.linalg.norm(u_star_train-u_train_pred,2)/np.linalg.norm(u_star_train,2)        
        error_v_train = np.linalg.norm(v_star_train-v_train_pred,2)/np.linalg.norm(v_star_train,2)
        f = open("stdout.txt", "a+")
        f.write('Training Error u: %e' % (error_u_train))    
        f.write('Training Error v: %e' % (error_v_train))   
        
        u_val_pred, v_val_pred = model.predict(X_star_val)
        error_u_val = np.linalg.norm(u_star_val-u_val_pred,2)/np.linalg.norm(u_star_val,2)        
        error_v_val = np.linalg.norm(v_star_val-v_val_pred,2)/np.linalg.norm(v_star_val,2)
        f.write('Val Error u: %e' % (error_u_val))    
        f.write('Val Error v: %e' % (error_v_val))   
        
        ######################## Plots for Adam(Pretraining) #################
        fig = plt.figure()
        plt.plot(loss_history_Adam_Pretrain)
        plt.plot(loss_val_history_Adam_Pretrain)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('100x')
        plt.title('log loss history of Adam(Pretraining)')
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Adam_Pretrain)
        plt.plot(loss_v_history_Adam_Pretrain)
        plt.yscale('log')       
        plt.xlabel('100x')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u and loss_v history of Adam(Pretraining)')  
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(loss_f_u_history_Adam_Pretrain)
        plt.plot(loss_f_v_history_Adam_Pretrain)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f_u and loss_f_v history of Adam(Pretraining)')  
        plt.savefig('3.png')
                        
        ######################## Plots for L-BFGS-B(Pretraining) #################
        fig = plt.figure()
        plt.plot(loss_history_Pretrain)
        plt.plot(loss_val_history_Pretrain)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('100X')
        plt.title('log loss history of BFGS(Pretrain)')  
        plt.savefig('4.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Pretrain)
        plt.plot(loss_v_history_Pretrain)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u and loss_v history of BFGS(Pretrain)')  
        plt.savefig('5.png')
        
        fig = plt.figure()
        plt.plot(loss_f_u_history_Pretrain)
        plt.plot(loss_f_v_history_Pretrain)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f_u and loss_f_v history of BFGS(Pretrain)')     
        plt.savefig('6.png')
            
        ######################## Plots for Adam #################
        fig = plt.figure()
        plt.plot(loss_history_Adam)
        plt.plot(loss_val_history_Adam)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('100X')
        plt.title('log loss history of Adam')
        plt.savefig('7.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Adam)
        plt.plot(loss_v_history_Adam)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u and loss_v history of Adam')  
        plt.savefig('8.png')
        
        fig = plt.figure()
        plt.plot(loss_f_u_history_Adam)
        plt.plot(loss_f_v_history_Adam)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f_u and loss_f_v history of Adam') 
        plt.savefig('9.png')
                                
        ######################## Plots for BFGS #################
        fig = plt.figure()
        plt.plot(loss_history)
        plt.plot(loss_val_history)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('100X')
        plt.title('log loss history of BFGS')  
        plt.savefig('10.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history)
        plt.plot(loss_v_history)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u and loss_v history of BFGS')  
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(loss_f_u_history)
        plt.plot(loss_f_v_history)
        plt.yscale('log')       
        plt.xlabel('100X')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f_u and loss_f_v history of BFGS')     
        plt.savefig('12.png')

# =============================================================================
#       compare with ground truth if training efforts are sufficient
# =============================================================================
        u_pred_full, v_pred_full = model.predict(X_star)        
        scipy.io.savemat('PredResponse_full.mat', {'u_pred_full':u_pred_full, 'v_pred_full':v_pred_full})
        
        lambda_u_value = model.sess.run(model.lambda_u)   
        lambda_u_value = np.asarray(lambda_u_value)
        lambda_u_true = np.array([0.1, 0.1, 1, -1, 1, -1, 1])
                
        lambda_u_error_vector = np.absolute((lambda_u_true-lambda_u_value)/lambda_u_true)
        error_lambda_u_mean = np.mean(lambda_u_error_vector)
        error_lambda_u_std = np.std(lambda_u_error_vector)
        f.write('lambda_u Mean Error: %.4f' % (error_lambda_u_mean))
        f.write('lambda_u Std Error: %.4f' % (error_lambda_u_std))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_fun_u_descr)):
            if lambda_u_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda_u_value[i_lib,0]) + model.lib_fun_u_descr[i_lib])
        disc_eq_u = '+'.join(disc_eq_temp)        
        print('The discovered equation: u_t = ' + disc_eq_u)

        lambda_v_value = model.sess.run(model.lambda_v)
        lambda_v_value = np.asarray(lambda_v_value)
        lambda_v_true = np.array([0.1, 0.1, -1, 1, -1, -1, -1])
                
        lambda_v_error_vector = np.absolute((lambda_v_true-lambda_v_value)/lambda_v_true)
        error_lambda_v_mean = np.mean(lambda_v_error_vector)
        error_lambda_v_std = np.std(lambda_v_error_vector)
        f.write('lambda_v Mean Error: %.4f' % (error_lambda_v_mean))
        f.write('lambda_v Std Error: %.4f' % (error_lambda_v_std))
        
        lambda_total_error_vector = np.concatenate((lambda_u_error_vector, lambda_v_error_vector))
        error_lambda_total_mean = np.mean(lambda_total_error_vector)
        error_lambda_total_std = np.std(lambda_total_error_vector)
        f.write('lambda_total Mean Error: %.4f' % (error_lambda_total_mean))
        f.write('lambda_total Std Error: %.4f' % (error_lambda_total_std))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_fun_v_descr)):
            if lambda_v_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda_v_value[i_lib,0]) + model.lib_fun_v_descr[i_lib])
        disc_eq_v = '+'.join(disc_eq_temp)        
        print('The discovered equation: v_t = ' + disc_eq_v)
        
        elapsed = time.time() - start_time                
        f.write('Training time: %.4f' % (elapsed))
        f.close()
        
        scipy.io.savemat('Results.mat', {'lambda_u_value': lambda_u_value, 'lambda_u_true': lambda_u_true, 'lambda_v_value': lambda_v_value, 'lambda_v_true': lambda_v_true})
        
        
        
                        
        ######################## Plots for lambdas #################
        fig = plt.figure()
        plt.plot(lambda_u_value, 'ro-')
        plt.plot(lambda_u_true)
        plt.legend(('the pred', 'the true'))
        plt.title('lambda_u ')
        plt.savefig('13.png')
        
        fig = plt.figure()
        plt.plot(lambda_v_value, 'ro-')
        plt.plot(lambda_v_true)
        plt.legend(('the pred', 'the true'))
        plt.title('lambda_v ')        
        plt.savefig('14.png')