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
    
with tf.device('/device:GPU:0'):
    
    # Adam loss history(Pretraining)
    loss_history_Adam_Pretrain = np.empty([0])
    loss_val_history_Adam_Pretrain = np.empty([0])
    loss_u_history_Adam_Pretrain = np.empty([0])
    loss_v_history_Adam_Pretrain = np.empty([0])
    loss_w_history_Adam_Pretrain = np.empty([0])
    loss_f_w_history_Adam_Pretrain = np.empty([0])
    loss_f_wxy_history_Adam_Pretrain = np.empty([0])
    loss_lambda_w_history_Adam_Pretrain = np.empty([0])
    lambda_w_history_Adam_Pretrain = np.empty([60, 1])
    
    # L-BFGS-B loss history(Pretraining)
    loss_history_Pretrain = np.empty([0])
    loss_val_history_Pretrain = np.empty([0])
    loss_u_history_Pretrain = np.empty([0])
    loss_v_history_Pretrain = np.empty([0])
    loss_w_history_Pretrain = np.empty([0])
    loss_f_w_history_Pretrain = np.empty([0])
    loss_f_wxy_history_Pretrain = np.empty([0])
    loss_lambda_w_history_Pretrain = np.empty([0])
    lambda_w_history_Pretrain = np.empty([60, 1])
    step_Pretrain = 0
    
    # L-BFGS-S loss history
    loss_history = np.empty([0])
    loss_val_history = np.empty([0])
    loss_u_history = np.empty([0])
    loss_v_history = np.empty([0])
    loss_w_history = np.empty([0])
    loss_f_w_history = np.empty([0])
    loss_f_wxy_history = np.empty([0])
    loss_lambda_w_history = np.empty([0])
    lambda_w_history = np.empty([60, 1])
    step = 0
    
    # Adam loss history
    loss_history_Adam = np.empty([0])
    loss_val_history_Adam = np.empty([0])
    loss_u_history_Adam = np.empty([0])
    loss_v_history_Adam = np.empty([0])
    loss_w_history_Adam = np.empty([0])
    loss_f_w_history_Adam = np.empty([0])
    loss_f_wxy_history_Adam = np.empty([0])
    loss_lambda_w_history_Adam = np.empty([0])
    lambda_w_history_Adam = np.empty([60, 1])
    
    # Alter loss history
    loss_history_Alter = np.empty([0])
    loss_val_history_Alter = np.empty([0])
    loss_u_history_Alter = np.empty([0])
    loss_v_history_Alter = np.empty([0])
    loss_w_history_Alter = np.empty([0])
    loss_f_w_history_Alter = np.empty([0])
    loss_f_wxy_history_Alter = np.empty([0])
    loss_lambda_w_history_Alter = np.empty([0])
    lambda_w_history_Alter = np.empty([60, 1])
    
    # STRidge loss histroy
    loss_w_history_STRidge = np.empty([0])
    loss_f_w_history_STRidge = np.empty([0])
    loss_lambda_w_history_STRidge = np.empty([0])
    tol_w_history_STRidge = np.empty([0])
    lambda_history_STRidge = np.zeros((60, 1))
    ridge_append_counter_STRidge = np.array([0])
        
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
        def __init__(self, X, u, v, w, X_f, X_val, u_val, v_val, w_val, layers, lb, ub, BatchNo):
            
            self.lb = lb
            self.ub = ub
            self.layers = layers
            
            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters
            self.lambda_w = tf.Variable(tf.zeros([60, 1], dtype=tf.float32), dtype=tf.float32)
            
            # Specify the list of trainable variables 
            var_list_1 = self.biases + self.weights
            
            var_list_Pretrain = self.biases + self.weights
            var_list_Pretrain.append(self.lambda_w)
            
            ######### Training data ################
			
            self.x = X[:,0:1]
            self.y = X[:,1:2]
            self.t = X[:,2:3]
            self.u = u
            self.v = v
            self.w = w
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
            self.w_tf = tf.placeholder(tf.float32, shape=[None, self.w.shape[1]])
            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
            
            self.u_pred, self.v_pred, self.w_pred = self.net_U(self.x_tf, self.y_tf, self.t_tf)
            self.f_w_pred, self.f_wxy_pred, self.Phi, self.w_t_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf,
                                                                                 self.batchsize_f)

            self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
            self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
            self.loss_w = tf.reduce_mean(tf.square(self.w_tf - self.w_pred))
            self.loss_f_w = tf.reduce_mean(tf.square(self.f_w_pred))
            self.loss_f_wxy = tf.reduce_mean(tf.square(self.f_wxy_pred))
            
            self.loss_lambda_w = 1e-10*tf.norm(self.lambda_w, ord=1)    
            
            self.loss_NNparas = tf.square(tf.norm(self.weights[0])) # We can't regularize self.biases due to its intialization as zeros.
            for i in range(1, len(self.weights)):
                self.loss_NNparas = self.loss_NNparas + tf.square(tf.norm(self.weights[i]))  
            self.loss_NNparas = self.loss_NNparas*1e-8
            
            self.loss = tf.log(self.loss_u  + self.loss_v + self.loss_w + self.loss_f_w + self.loss_f_wxy + self.loss_lambda_w + \
                               self.loss_NNparas) 
                        
            ######### Validation data ################
            self.x_val = X_val[:,0:1]
            self.y_val = X_val[:,1:2]
            self.t_val = X_val[:,2:3]
            self.u_val = u_val
            self.v_val = v_val
            self.w_val = w_val
            
            self.batchsize_val = np.floor(self.x_val.shape[0]/self.BatchNo)
            
            self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
            self.y_val_tf = tf.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
            self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
            self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
            self.v_val_tf = tf.placeholder(tf.float32, shape=[None, self.v_val.shape[1]])
            self.w_val_tf = tf.placeholder(tf.float32, shape=[None, self.w_val.shape[1]])
            
            self.u_val_pred, self.v_val_pred, self.w_val_pred = self.net_U(self.x_val_tf, self.y_val_tf, self.t_val_tf)
    
            self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
            self.loss_v_val = tf.reduce_mean(tf.square(self.v_val_tf - self.v_val_pred))
            self.loss_w_val = tf.reduce_mean(tf.square(self.w_val_tf - self.w_val_pred))
            self.loss_val = tf.log(self.loss_u_val  + self.loss_v_val + self.loss_w_val)     
            
            ######### Optimizor #########################
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = var_list_1,
    #                                                                    L-BFGS-B
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 1000,
                                                                               'maxfun': 1000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 0.1 * np.finfo(float).eps})
    
            self.optimizer_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                            var_list = var_list_Pretrain,
            #                                                                    L-BFGS-B
                                                                            method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 10000,
                                                                                       'maxfun': 10000,
                                                                                       'maxcor': 50,
                                                                                       'maxls': 50,
                                                                                       'ftol' : 0.1 * np.finfo(float).eps})
                        
            self.optimizer_Adam = tf.contrib.opt.NadamOptimizer(learning_rate = 1e-4) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = var_list_1)
            
            
            self.optimizer_Adam_Pretrain = tf.contrib.opt.NadamOptimizer(learning_rate = 1e-3)
            self.train_op_Adam_Pretrain = self.optimizer_Adam_Pretrain.minimize(self.loss, var_list = var_list_Pretrain)
            
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
                     
        def net_U(self, x, y, t):  
            Y = self.neural_net(tf.concat([x,y,t],1), self.weights, self.biases)
            u = Y[:, 0:1]
            v = Y[:, 1:2]
            w = Y[:, 2:3]
            return u, v, w
        
        def net_f(self, x, y, t, N_f):    
            u, v, w = self.net_U(x,y,t)
            data = [u, v, w]            
            
            ## derivatives     
            w_x = tf.gradients(w, x)[0]
            w_y = tf.gradients(w, y)[0]
            w_xx = tf.gradients(w_x, x)[0]
            w_yy = tf.gradients(w_y, y)[0]
            w_xy = tf.gradients(w_x, y)[0]
            w_yx = tf.gradients(w_y, x)[0]
                                   
            derivatives = [1]
            derivatives.append(w_x)
            derivatives.append(w_y)
            derivatives.append(w_xx)
            derivatives.append(w_xy)
            derivatives.append(w_yy)
            
            derivatives_description = ['', 'w_{x}', 'w_{y}', 'w_{xx}', 'w_{xy}', 'w_{yy}']
            
            global lib_fun
            global lib_descr
            lib_fun, self.lib_descr = self.build_library(data, derivatives, derivatives_description, PolyOrder = 2, 
                                                    data_description = ['u','v', 'w'])      
            w_t = tf.gradients(w, t)[0]
            f_w = w_t
            Phi = tf.constant(1, shape=[N_f, 1], dtype=tf.float32)
            for i in range(len(lib_fun)):
                f_w = f_w - lib_fun[i]*self.lambda_w[i:i+1,0:1] # Note that the minus sign instead of the plus sign is uesd here.
                if i != 0:
                    Phi = tf.concat([Phi, lib_fun[i]], 1)
                    
            f_wxy = w_xy - w_yx
                                    
            return f_w, f_wxy, Phi, w_t
        
        def build_library(self, data, derivatives, derivatives_description, PolyOrder = 2, data_description = None):         
            ## polynomial terms
            P = PolyOrder
            lib_poly = [1]
            lib_poly_descr = [''] # it denotes '1'
            for i in range(len(data)): # polynomial terms of univariable
                for j in range(1, P+1):
                    lib_poly.append(data[i]**j)
                    lib_poly_descr.append(data_description[i]+"**"+str(j))                    
            
            lib_poly.append(data[0]*data[1])
            lib_poly_descr.append(data_description[0]+data_description[1])
            lib_poly.append(data[0]*data[2])
            lib_poly_descr.append(data_description[0]+data_description[2])
            lib_poly.append(data[1]*data[2])
            lib_poly_descr.append(data_description[1]+data_description[2])
                    
            ## derivative terms
            lib_deri = derivatives
            lib_deri_descr = derivatives_description
            
            ## Multiplication of derivatives and polynomials (including the multiplication with '1')
            lib_poly_deri = []
            lib_poly_deri_descr = []
            for i in range(len(lib_poly)):
                for j in range(len(lib_deri)):
                    lib_poly_deri.append(lib_poly[i]*lib_deri[j])
                    lib_poly_deri_descr.append(lib_poly_descr[i]+lib_deri_descr[j])
                    
            return lib_poly_deri,lib_poly_deri_descr
                    
        def callback(self, loss, loss_u, loss_v, loss_w, loss_f_w, loss_f_wxy, loss_lambda_w, loss_val, lambda_w):
            global step
            step = step+1
            if step%10 == 0:
                
                global loss_history
                global loss_val_history
                global loss_u_history
                global loss_v_history
                global loss_w_history
                global loss_f_w_history
                global loss_f_wxy_history
                global loss_lambda_w_history
                global lambda_w_history
                
                loss_history = np.append(loss_history, loss)
                loss_val_history = np.append(loss_val_history, loss_val)
                loss_u_history = np.append(loss_u_history, loss_u)
                loss_v_history = np.append(loss_v_history, loss_v)
                loss_w_history = np.append(loss_w_history, loss_w)
                loss_f_w_history = np.append(loss_f_w_history, loss_f_w)
                loss_f_wxy_history = np.append(loss_f_wxy_history, loss_f_wxy)
                loss_lambda_w_history = np.append(loss_lambda_w_history, loss_lambda_w)
                lambda_w_history = np.append(lambda_w_history, lambda_w, axis = 1)
                
        def callback_Pretrain(self, loss, loss_u, loss_v, loss_w, loss_f_w, loss_f_wxy, loss_lambda_w, loss_val, lambda_w):
            global step_Pretrain
            step_Pretrain += 1
            if step_Pretrain % 10 == 0:
                
                global loss_history_Pretrain
                global loss_val_history_Pretrain
                global loss_u_history_Pretrain
                global loss_v_history_Pretrain
                global loss_w_history_Pretrain
                global loss_f_w_history_Pretrain
                global loss_f_wxy_history_Pretrain
                global loss_lambda_w_history_Pretrain
                global lambda_w_history_Pretrain
                
                loss_history_Pretrain = np.append(loss_history_Pretrain, loss)
                loss_val_history_Pretrain = np.append(loss_val_history_Pretrain, loss_val)
                loss_u_history_Pretrain = np.append(loss_u_history_Pretrain, loss_u)
                loss_v_history_Pretrain = np.append(loss_v_history_Pretrain, loss_v)
                loss_w_history_Pretrain = np.append(loss_w_history_Pretrain, loss_w)
                loss_f_w_history_Pretrain = np.append(loss_f_w_history_Pretrain, loss_f_w)
                loss_f_wxy_history_Pretrain = np.append(loss_f_wxy_history_Pretrain, loss_f_wxy)
                loss_lambda_w_history_Pretrain = np.append(loss_lambda_w_history_Pretrain, loss_lambda_w)
                lambda_w_history_Pretrain = np.append(lambda_w_history_Pretrain, lambda_w, axis = 1)
            
        def train(self, nIter):                   
            
            # With batches
            for i in range(self.BatchNo):                
                x_batch = self.x[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                y_batch = self.y[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                t_batch = self.t[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                u_batch = self.u[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                v_batch = self.v[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                w_batch = self.w[int(i*self.batchsize_train):int((i+1)*self.batchsize_train), :]
                
                x_f_batch = self.x_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                y_f_batch = self.y_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                t_f_batch = self.t_f[int(i*self.batchsize_f):int((i+1)*self.batchsize_f), :]
                
                x_val_batch = self.x_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                y_val_batch = self.y_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                t_val_batch = self.t_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                u_val_batch = self.u_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                v_val_batch = self.v_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                w_val_batch = self.w_val[int(i*self.batchsize_val):int((i+1)*self.batchsize_val), :]
                
                self.tf_dict = {self.x_tf: x_batch, self.y_tf: y_batch, self.t_tf: t_batch, 
                                self.u_tf: u_batch, self.v_tf: v_batch, self.w_tf: w_batch,
                                self.x_f_tf: x_f_batch, self.y_f_tf: y_f_batch, self.t_f_tf: t_f_batch, 
                                self.x_val_tf: x_val_batch, self.y_val_tf: y_val_batch, self.t_val_tf: t_val_batch,
                                self.u_val_tf: u_val_batch, self.v_val_tf: v_val_batch, self.w_val_tf: w_val_batch}
            
                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                # Adam optimizer(Pretraining)
                print('Adam(Pretraining) starts')
                for it_Adam in tqdm(range(5000)):
                    self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict, options=run_options)
                    
                    # Print
                    if it_Adam % 10 == 0:
                        loss, loss_u, loss_v, loss_w, loss_f_w, loss_f_wxy, loss_lambda_w, loss_val = \
                            self.sess.run([self.loss, self.loss_u, self.loss_v, self.loss_w, self.loss_f_w, self.loss_f_wxy, 
                                           self.loss_lambda_w, self.loss_val], self.tf_dict)
    
                        lambda_w = self.sess.run(self.lambda_w)
    
                        global loss_history_Adam_Pretrain
                        global loss_val_history_Adam_Pretrain
                        global loss_u_history_Adam_Pretrain
                        global loss_v_history_Adam_Pretrain
                        global loss_w_history_Adam_Pretrain
                        global loss_f_w_history_Adam_Pretrain
                        global loss_f_wxy_history_Adam_Pretrain
                        global loss_lambda_w_history_Adam_Pretrain
                        global lambda_w_history_Adam_Pretrain
                        
                        loss_history_Adam_Pretrain = np.append(loss_history_Adam_Pretrain, loss)
                        loss_val_history_Adam_Pretrain = np.append(loss_val_history_Adam_Pretrain, loss_val)
                        loss_u_history_Adam_Pretrain = np.append(loss_u_history_Adam_Pretrain, loss_u)
                        loss_v_history_Adam_Pretrain = np.append(loss_v_history_Adam_Pretrain, loss_v)
                        loss_w_history_Adam_Pretrain = np.append(loss_w_history_Adam_Pretrain, loss_w)
                        loss_f_w_history_Adam_Pretrain = np.append(loss_f_w_history_Adam_Pretrain, loss_f_w)
                        loss_f_wxy_history_Adam_Pretrain = np.append(loss_f_wxy_history_Adam_Pretrain, loss_f_wxy)
                        loss_lambda_w_history_Adam_Pretrain = np.append(loss_lambda_w_history_Adam_Pretrain, loss_lambda_w)
                        lambda_w_history_Adam_Pretrain = np.append(lambda_w_history_Adam_Pretrain, lambda_w, axis = 1)
                        
                # L-BFGS-B optimizer(Pretraining)
                print('L-BFGS-B(Pretraining) starts')
                self.optimizer_Pretrain.minimize(self.sess,
                                        feed_dict = self.tf_dict,
                                        fetches = [self.loss, self.loss_u, self.loss_v,  self.loss_w, self.loss_f_w, self.loss_f_wxy,
                                                   self.loss_lambda_w, self.loss_val, self.lambda_w],
                                        loss_callback = self.callback_Pretrain)
                                
                for it in tqdm(range(nIter)):    
                    # Adam optimizer                
                    print('Adam starts')
                    for it_Adam in tqdm(range(500)):
                        self.sess.run(self.train_op_Adam, self.tf_dict, options=run_options)
                        
                        # Print
                        if it_Adam % 10 == 0:
                            loss, loss_u, loss_v, loss_w, loss_f_w, loss_f_wxy, loss_lambda_w, loss_val = \
                            self.sess.run([self.loss, self.loss_u, self.loss_v, self.loss_w, self.loss_f_w, self.loss_f_wxy, 
                                           self.loss_lambda_w, self.loss_val], self.tf_dict)
                    
                            lambda_w = self.sess.run(self.lambda_w)
                            
                            global loss_history_Adam
                            global loss_val_history_Adam
                            global loss_u_history_Adam
                            global loss_v_history_Adam
                            global loss_w_history_Adam
                            global loss_f_w_history_Adam
                            global loss_f_wxy_history_Adam
                            global loss_lambda_w_history_Adam
                            global lambda_w_history_Adam
                            
                            loss_history_Adam = np.append(loss_history_Adam, loss)
                            loss_val_history_Adam = np.append(loss_val_history_Adam, loss_val)
                            loss_u_history_Adam = np.append(loss_u_history_Adam, loss_u)
                            loss_v_history_Adam = np.append(loss_v_history_Adam, loss_v)
                            loss_w_history_Adam = np.append(loss_w_history_Adam, loss_w)
                            loss_f_w_history_Adam = np.append(loss_f_w_history_Adam, loss_f_w)
                            loss_f_wxy_history_Adam = np.append(loss_f_wxy_history_Adam, loss_f_wxy)
                            loss_lambda_w_history_Adam = np.append(loss_lambda_w_history_Adam, loss_lambda_w)
                            lambda_w_history_Adam = np.append(lambda_w_history_Adam, lambda_w, axis = 1)
                            
                        if it_Adam == 499:
                            loss, loss_u, loss_v, loss_w, loss_f_w, loss_f_wxy, loss_lambda_w, loss_val = \
                            self.sess.run([self.loss, self.loss_u, self.loss_v, self.loss_w, self.loss_f_w, self.loss_f_wxy, 
                                           self.loss_lambda_w, self.loss_val], self.tf_dict)
                    
                            lambda_w = self.sess.run(self.lambda_w)
                            
                            global loss_history_Alter
                            global loss_val_history_Alter
                            global loss_u_history_Alter
                            global loss_v_history_Alter
                            global loss_w_history_Alter
                            global loss_f_w_history_Alter
                            global loss_f_wxy_history_Alter
                            global loss_lambda_w_history_Alter
                            global lambda_w_history_Alter
                            
                            loss_history_Alter = np.append(loss_history_Alter, loss)
                            loss_val_history_Alter = np.append(loss_val_history_Alter, loss_val)
                            loss_u_history_Alter = np.append(loss_u_history_Alter, loss_u)
                            loss_v_history_Alter = np.append(loss_v_history_Alter, loss_v)
                            loss_w_history_Alter = np.append(loss_w_history_Alter, loss_w)
                            loss_f_w_history_Alter = np.append(loss_f_w_history_Alter, loss_f_w)
                            loss_f_wxy_history_Alter = np.append(loss_f_wxy_history_Alter, loss_f_wxy)
                            loss_lambda_w_history_Alter = np.append(loss_lambda_w_history_Alter, loss_lambda_w)
                            lambda_w_history_Alter = np.append(lambda_w_history_Alter, lambda_w, axis = 1)
                            
                    # L-BFGS-B optimizer
                    print('L-BFGS-B starts')
                    self.optimizer.minimize(self.sess,
                                            feed_dict = self.tf_dict,
                                            fetches = [self.loss, self.loss_u, self.loss_v, self.loss_w, self.loss_f_w, 
                                                       self.loss_f_wxy, self.loss_lambda_w, self.loss_val, self.lambda_w],
                                            loss_callback = self.callback)
                    
                    # STRidge optimizer
                    print('STRidge starts')
                    self.callTrainSTRidge()
                        
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.y_tf: X_star[:,1:2], self.t_tf: X_star[:,2:3]}
            
            u_star = self.sess.run(self.u_pred, tf_dict)
            v_star = self.sess.run(self.v_pred, tf_dict)
            w_star = self.sess.run(self.w_pred, tf_dict)
            
            return u_star, v_star, w_star
        
        def callTrainSTRidge(self):
            lam = 1e-5
            d_tol = 1
            maxit = 25
            STR_iters = 10
            
            l0_penalty = None
            
            normalize = 2
            split = 0.8
            print_best_tol = False     
            
            # Process of lambda_u            
            Phi, w_t_pred = self.sess.run([self.Phi, self.w_t_pred], self.tf_dict)        
            lambda_w2, loss_w_history_STRidge2, loss_f_w_history_STRidge2, loss_lambda_w_history_STRidge2, tol_w_history_STRidge2, \
                optimaltol_w_history2 = self.TrainSTRidge(Phi, w_t_pred, lam, d_tol, maxit, STR_iters, l0_penalty, normalize, split, \
                                                          print_best_tol)
            self.lambda_w = tf.assign(self.lambda_w, tf.convert_to_tensor(lambda_w2, dtype = tf.float32))
            
            
            global loss_w_history_STRidge
            global loss_f_w_history_STRidge
            global loss_lambda_w_history_STRidge
            global tol_w_history_STRidge
            
            loss_w_history_STRidge = np.append(loss_w_history_STRidge, loss_w_history_STRidge2)
            loss_f_w_history_STRidge = np.append(loss_f_w_history_STRidge, loss_f_w_history_STRidge2)
            loss_lambda_w_history_STRidge = np.append(loss_lambda_w_history_STRidge, loss_lambda_w_history_STRidge2)
            tol_w_history_STRidge = np.append(tol_w_history_STRidge, tol_w_history_STRidge2)
                    
        def TrainSTRidge(self, R, Ut, lam, d_tol, maxit, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, 
                         print_best_tol = False):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================       
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
        
            # Set up the initial tolerance and l0 penalty
            d_tol = float(d_tol)
            tol = d_tol
            if l0_penalty == None: l0_penalty = 1e-3*np.linalg.cond(R)
                    
            w_best= self.sess.run(self.lambda_w)
            
            err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
            err_lambda = l0_penalty*np.count_nonzero(w_best)
            err_best = err_f + err_lambda
            tol_best = 0
                        
            loss_history_STRidge = np.empty([0])
            loss_f_history_STRidge = np.empty([0])
            loss_lambda_history_STRidge = np.empty([0])
            tol_history_STRidge = np.empty([0])
            loss_history_STRidge = np.append(loss_history_STRidge, err_best)
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
            loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
            tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
        
            # Now increase tolerance until test performance decreases
            for iter in range(maxit):
        
                # Get a set of coefficients and error
                w = self.STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
                err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
                err_lambda = l0_penalty*np.count_nonzero(w)
                err = err_f + err_lambda
        
                # Has the accuracy improved?
                if err <= err_best:
                    err_best = err
                    w_best = w
                    tol_best = tol
                    tol = tol + d_tol
                    
                    loss_history_STRidge = np.append(loss_history_STRidge, err_best)
                    loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
                    loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
                    tol_history_STRidge = np.append(tol_history_STRidge, tol)
        
                else:
                    tol = max([0,tol - 2*d_tol])
                    d_tol  = 2*d_tol / (maxit - iter)
                    tol = tol + d_tol
        
            if print_best_tol: print ("Optimal tolerance:", tol_best)
            
            optimaltol_history = np.empty([0])
            optimaltol_history = np.append(optimaltol_history, tol_best)
        
            return np.real(w_best), loss_history_STRidge, loss_f_history_STRidge, loss_lambda_history_STRidge, tol_history_STRidge, optimaltol_history
        
        def STRidge(self, X0, y, lam, maxit, tol, normalize = 2, print_results = False):
        
            n,d = X0.shape
            X = np.zeros((n,d), dtype=np.complex64)
            # First normalize data
            if normalize != 0:
                Mreg = np.zeros((d,1))
                for i in range(0,d):
                    Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                    X[:,i] = Mreg[i]*X0[:,i]
            else: X = X0
                        
            # Inherit w from previous trainning
            w = self.sess.run(self.lambda_w)/Mreg
            
            num_relevant = d
            biginds = np.where( abs(w) > tol)[0]
            
            global ridge_append_counter_STRidge
            ridge_append_counter = 0
            
            global lambda_history_STRidge
            lambda_history_STRidge = np.append(lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
            ridge_append_counter += 1
            
            # Threshold and continue
            for j in range(maxit):
        
                # Figure out which items to cut out
                smallinds = np.where( abs(w) < tol)[0]
                new_biginds = [i for i in range(d) if i not in smallinds]
                    
                # If nothing changes then stop
                if num_relevant == len(new_biginds): break
                else: num_relevant = len(new_biginds)
                    
                # Also make sure we didn't just lose all the coefficients
                if len(new_biginds) == 0:
                    if j == 0: 
                        if normalize != 0: 
                            w = np.multiply(Mreg,w)
                            lambda_history_STRidge = np.append(lambda_history_STRidge, w, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return w
                        else: 
                            lambda_history_STRidge = np.append(lambda_history_STRidge, w, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return w
                    else: break
                biginds = new_biginds
                
                # Otherwise get a new guess
                w[smallinds] = 0
                
                if lam != 0: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
                    lambda_history_STRidge = np.append(lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
                    ridge_append_counter += 1
                else: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
                    lambda_history_STRidge = np.append(lambda_history_STRidge, w, axis = 1)
                    ridge_append_counter += 1
    
            # Now that we have the sparsity pattern, use standard least squares to get w
            if biginds != []: 
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            
            if normalize != 0: 
                w = np.multiply(Mreg,w)
                lambda_history_STRidge = np.append(lambda_history_STRidge, w, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return w
            else:
                lambda_history_STRidge = np.append(lambda_history_STRidge, w, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return w
    
        
    if __name__ == "__main__": 
         
        start_time = time.time()
        
        layers = [3, 60, 60, 60, 60, 60, 60, 60, 60, 3]
        
# =============================================================================
#         load data
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
        # Preprocess data #2(compatible with NN format)
        t_star = np.reshape(t_data,(-1,1))
        x_star = np.reshape(x_data,(-1,1))
        y_star = np.reshape(y_data,(-1,1))        
        u_star = np.reshape(u_data,(-1,1))
        v_star = np.reshape(v_data,(-1,1))
        w_star = np.reshape(w_data,(-1,1))
        
        X_star = np.hstack((x_star, y_star, t_star))
                            
        ## Spatially randomly but temporally continuously sampled measurements
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
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_meas.shape[0]), idx_train, assume_unique=True)
        X_val = X_meas[idx_val,:]
        u_val = u_meas[idx_val,:]
        v_val = v_meas[idx_val,:]
        w_val = w_meas[idx_val,:]       
    
        # Doman bounds        
        lb = X_star.min(0)
        ub = X_star.max(0)    
        
        # Collocation points
        N_f = 60000
#        X_f = lb + (ub-lb)*sobol_seq.i4_sobol_generate(3, N_f)
        X_f = lb + (ub-lb)*lhs(3, N_f)
        X_f = np.vstack((X_f, X_train))
        
        # add noise
        noise = 0.1
        u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
        w_train = w_train + noise*np.std(w_train)*np.random.randn(w_train.shape[0], w_train.shape[1])
        u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
        v_val = v_val + noise*np.std(v_val)*np.random.randn(v_val.shape[0], v_val.shape[1])
        w_val = w_val + noise*np.std(w_val)*np.random.randn(w_val.shape[0], w_val.shape[1])

# =============================================================================
        # model training
# =============================================================================
        batchNo = 1
        model = PhysicsInformedNN(X_train, u_train, v_train, w_train, X_f, X_val, u_val, v_val, w_val, layers, lb, ub, batchNo)
        model.train(6)
                
# =============================================================================
#         evaluate training efforts
# =============================================================================
        f = open("stdout.txt", "a+") 
        
        # training error
        u_train_pred, v_train_pred, w_train_pred = model.predict(X_train)
        error_u_train = np.linalg.norm(u_train-u_train_pred,2)/np.linalg.norm(u_train,2)        
        error_v_train = np.linalg.norm(v_train-v_train_pred,2)/np.linalg.norm(v_train,2)
        error_w_train = np.linalg.norm(w_train-w_train_pred,2)/np.linalg.norm(w_train,2)
        
        f.write('Training Error u: %e \n' % (error_u_train))    
        f.write('Training Error v: %e \n' % (error_v_train))   
        f.write('Training Error w: %e \n' % (error_w_train))   
        
        # validation error
        u_val_pred, v_val_pred, w_val_pred = model.predict(X_val)
        error_u_val = np.linalg.norm(u_val-u_val_pred,2)/np.linalg.norm(u_val,2)        
        error_v_val = np.linalg.norm(v_val-v_val_pred,2)/np.linalg.norm(v_val,2)
        error_w_val = np.linalg.norm(w_val-w_val_pred,2)/np.linalg.norm(w_val,2)
        f.write('Val Error u: %e \n' % (error_u_val))    
        f.write('Val Error v: %e \n' % (error_v_val))   
        f.write('Val Error w: %e \n' % (error_w_val))   
        
        # loss histories
        ######################## Plots for Adam(Pretraining) #################
        fig = plt.figure()
        plt.plot(loss_history_Adam_Pretrain)
        plt.plot(loss_val_history_Adam_Pretrain)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of Adam(Pretraining)')
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Adam_Pretrain)
        plt.plot(loss_v_history_Adam_Pretrain)
        plt.plot(loss_w_history_Adam_Pretrain)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.title('loss_u,v,w history of Adam(Pretraining)')  
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(loss_f_w_history_Adam_Pretrain)
        plt.plot(loss_f_wxy_history_Adam_Pretrain)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_w', 'loss_f_wxy'))
        plt.title('loss_f_w and loss_f_wxy history of Adam(Pretraining)')  
        plt.savefig('3.png')
                
        fig = plt.figure()
        plt.plot(loss_lambda_w_history_Adam_Pretrain)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda_whistory of Adam(Pretraining)')  
        plt.savefig('4.png')
        
        ######################## Plots for L-BFGS-B(Pretraining) #################
        fig = plt.figure()
        plt.plot(loss_history_Pretrain)
        plt.plot(loss_val_history_Pretrain)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of BFGS(Pretrain)')  
        plt.savefig('5.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Pretrain)
        plt.plot(loss_v_history_Pretrain)
        plt.plot(loss_w_history_Pretrain)
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u, v, w history of BFGS(Pretrain)')  
        plt.savefig('6.png')
        
        fig = plt.figure()
        plt.plot(loss_f_w_history_Pretrain)
        plt.plot(loss_f_wxy_history_Pretrain)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_w', 'loss_f_wxy'))
        plt.title('loss_f_w and loss_f_wxy history of BFGS(Pretrain)')     
        plt.savefig('7.png')
        
        fig = plt.figure()
        plt.plot(loss_lambda_w_history_Pretrain)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda_w history of BFGS(Pretrain)') 
        plt.savefig('8.png')
    
        ######################## Plots for Adam #################
        fig = plt.figure()
        plt.plot(loss_history_Adam)
        plt.plot(loss_val_history_Adam)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of Adam')
        plt.savefig('9.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_Adam)
        plt.plot(loss_v_history_Adam)
        plt.plot(loss_w_history_Adam)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.title('loss_u, v, w history of Adam')  
        plt.savefig('10.png')
        
        fig = plt.figure()
        plt.plot(loss_f_w_history_Adam)
        plt.plot(loss_f_wxy_history_Adam)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_w', 'loss_f_wxy'))
        plt.title('loss_f_w and loss_f_wxy history of Adam') 
        plt.savefig('11.png')
                
        fig = plt.figure()
        plt.plot(loss_lambda_w_history_Adam)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda_w history of Adam')  
        plt.savefig('12.png')
                
        ######################## Plots for BFGS #################
        fig = plt.figure()
        plt.plot(loss_history)
        plt.plot(loss_val_history)
        plt.legend(('train loss', 'val loss'))
        plt.xlabel('10x')
        plt.title('log loss history of BFGS')  
        plt.savefig('13.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history)
        plt.plot(loss_v_history)
        plt.plot(loss_w_history)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.title('loss_u, v, w history of BFGS')  
        plt.savefig('14.png')
        
        fig = plt.figure()
        plt.plot(loss_f_w_history)
        plt.plot(loss_f_wxy_history)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_w', 'loss_f_wxy'))
        plt.title('loss_f_w and loss_f_wxy history of BFGS')     
        plt.savefig('15.png')
        
        fig = plt.figure()
        plt.plot(loss_lambda_w_history)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda_w history of BFGS')  
        plt.savefig('16.png')
        
        ######################## Plots for STRidge #################        
        fig = plt.figure()        
        plt.plot(loss_w_history_STRidge)
        plt.yscale('log')    
        plt.title('History of STRidge(loss_f+loss_lambda )')  
        plt.savefig('17.png')
        
        fig = plt.figure()
        plt.plot(loss_f_w_history_STRidge)
        plt.yscale('log')       
        plt.title('loss_f_w history of STRidge') 
        plt.savefig('18.png')
        
        fig = plt.figure()
        plt.plot(loss_lambda_w_history_STRidge)
        plt.yscale('log')      
        plt.legend(('loss_lambda_w'))
        plt.title('loss_lambda_w history of STRidge')  
        plt.savefig('19.png')
        
        fig = plt.figure()
        plt.plot(tol_w_history_STRidge)
        plt.title('tol_w history of STRidge')  
        plt.savefig('20.png')

            
# =============================================================================
#         compare with ground truth
# =============================================================================
        u_full_pred, v_full_pred, w_full_pred = model.predict(X_star)
        error_u_full = np.linalg.norm(u_star-u_full_pred,2)/np.linalg.norm(u_star,2)        
        error_v_full = np.linalg.norm(v_star-v_full_pred,2)/np.linalg.norm(v_star,2)
        error_w_full = np.linalg.norm(w_star-w_full_pred,2)/np.linalg.norm(w_star,2)
        f.write('Full Error u: %e \n' % (error_u_full))    
        f.write('Full Error v: %e \n' % (error_v_full))   
        f.write('Full Error w: %e \n' % (error_w_full))   
        
        lambda_w_pred = model.sess.run(model.lambda_w)        
        lambda_w_true = np.zeros((60,1))
        lambda_w_true[3] = 0.01 # 0.01w_xx
        lambda_w_true[5] = 0.01 # 0.01*w_yy
        lambda_w_true[7] = -1 # -uw_x   
        lambda_w_true[20] = -1 # -vw_y
        cosine_similarity_w = 1-distance.cosine(lambda_w_true,lambda_w_pred)
        f.write('Cosine similarity of lambda_w: %.2f \n' % (cosine_similarity_w))
        error_lambda_w = np.linalg.norm(lambda_w_true-lambda_w_pred,2)/np.linalg.norm(lambda_w_true,2)
        f.write('lambda_w Error: %.2f \n' % (error_lambda_w))
        nonzero_ind_w = np.nonzero(lambda_w_true)
        lambda_w_error_vector = np.absolute((lambda_w_true[nonzero_ind_w]-lambda_w_pred[nonzero_ind_w])/lambda_w_true[nonzero_ind_w])
        error_lambda_w_mean = np.mean(lambda_w_error_vector)
        error_lambda_w_std = np.std(lambda_w_error_vector)
        f.write('lambda_w Mean Error: %.4f \n' % (error_lambda_w_mean))
        f.write('lambda_w Std Error: %.4f \n' % (error_lambda_w_std))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_descr)):
            if lambda_w_pred[i_lib] != 0:
                disc_eq_temp.append(str(lambda_w_pred[i_lib,0]) + model.lib_descr[i_lib])
        disc_eq = '+'.join(disc_eq_temp)        
        f.write('The discovered equation: w_t = ' + disc_eq)
        
        elapsed = time.time() - start_time                
        f.write('Training time: %.4f \n' % (elapsed))
        
        f.close()
        
        x = np.reshape(np.tile(np.arange(n).reshape((-1, 1))*dx , (1, m)), (-1, 1)) 
        y = np.reshape(np.tile(np.arange(m).reshape((-1, 1))*dy , (1, n)), (-1, 1))
        t = np.arange(steps)*dt 
            
        scipy.io.savemat('PredictedResponse_Full.mat', {'u_full_pred': u_full_pred, 'v_full_pred': v_full_pred,
                                                        'w_full_pred': w_full_pred})
    
        scipy.io.savemat('History.mat', {'lambda_w_history_Adam_Pretrain': lambda_w_history_Adam_Pretrain, 
                                         'lambda_w_history_Pretrain': lambda_w_history_Pretrain,
                                         'lambda_w_history_Alter': lambda_w_history_Alter,
                                         'lambda_history_STRidge': lambda_history_STRidge,
                                         'ridge_append_counter_STRidge': ridge_append_counter_STRidge,
                                         'lambda_w_pred': lambda_w_pred})
        
        
        ######################## Plots for lambdas #################
        fig = plt.figure()
        plt.plot(lambda_w_pred, 'ro-')
        plt.plot(lambda_w_true)
        plt.legend(('the pred', 'the true'))
        plt.title('lambda_w ')
        plt.savefig('21.png')