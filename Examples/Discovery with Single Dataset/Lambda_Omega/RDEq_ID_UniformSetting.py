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
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # limite all operations on this GPU
with tf.device('/device:GPU:0'):        
    # pt loss history
    loss_history = np.empty([0])
    loss_val_history = np.empty([0])
    loss_u_history = np.empty([0])
    loss_v_history = np.empty([0])
    loss_f_u_history = np.empty([0])
    loss_f_v_history = np.empty([0])
    step_Pt = 0
    lamu_history = np.zeros((7, 1))
    lamv_history = np.zeros((7, 1))

    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, v, X_f, X_val, u_val, v_val, layers, lb, ub, BatchNo, Lamu_init, Lamv_init):
            
            self.lb = lb
            self.ub = ub
            self.layers = layers
            
            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters based on ADO results
            self.lambda_u2 = tf.Variable(Lamu_init, dtype=tf.float32)
            self.lambda_v2 = tf.Variable(Lamv_init, dtype=tf.float32)
            
            # Specify the list of trainable variables 
            self.lambda_u = tf.Variable(tf.zeros([110, 1], dtype=tf.float32), dtype=tf.float32, name = 'lambda_u')
            self.lambda_v = tf.Variable(tf.zeros([110, 1], dtype=tf.float32), dtype=tf.float32, name = 'lambda_v')

            self.var_list_Pretrain = self.biases + self.weights + [self.lambda_u, self.lambda_v]
            self.var_list_Pt = self.biases + self.weights + [self.lambda_u2, self.lambda_v2]
            
            # Save the model after pretraining
            self.saver = tf.train.Saver(self.var_list_Pretrain)
            self.saver_pt = tf.train.Saver(self.var_list_Pt)

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
            self.loss_f_u = 10*tf.reduce_mean(tf.square(self.f_u_pred))
            self.loss_f_v = 10*tf.reduce_mean(tf.square(self.f_v_pred))
            
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
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 1e-3
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.8,
                                                         staircase=True)

            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = self.var_list_Pt, 
                                                              global_step = self.global_step)
            
            self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                             var_list = self.var_list_Pt,
        #                                                                    L-BFGS-B
                                                                            method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 10000,
                                                                                       'maxfun': 10000,
                                                                                       'maxcor': 50,
                                                                                       'maxls': 50,
                                                                                       'ftol' : 0.1 * np.finfo(float).eps})
                                    
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

            Phi_u = tf.concat(lib_fun_u, axis = 1)
            Phi_v = tf.concat(lib_fun_v, axis = 1)
            
            f_u = Phi_u@self.lambda_u2 - u_t
            f_v = Phi_v@self.lambda_v2 - v_t
                                                
            return f_u, f_v
                                                        
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
    
                # Loop of Adam optimization
                print('Adam begins')
                for it_Adam in tqdm(range(10000)):   
                    self.sess.run(self.train_op_Adam, self.tf_dict)                    
                    # Print
                    if it_Adam % 10 == 0:
                        loss_u, loss_v, loss_f_u, loss_f_v = self.sess.run([self.loss_u, self.loss_v,
                                                                                 self.loss_f_u,
                                                                                 self.loss_f_v], 
                                                                                self.tf_dict)
                        loss_all, loss_all_val = self.sess.run([self.loss,
                                                                self.loss_val], self.tf_dict)
                        
                        lamu, lamv = self.sess.run([self.lambda_u2, self.lambda_v2])

                        global loss_u_history
                        global loss_v_history
                        global loss_f_u_history
                        global loss_f_v_history
                        global loss_history
                        global loss_val_history
                        
                        global lamu_history
                        global lamv_history

                        loss_u_history = np.append(loss_u_history, loss_u)
                        loss_v_history = np.append(loss_v_history, loss_v)
                        loss_f_u_history = np.append(loss_f_u_history, loss_f_u)
                        loss_f_v_history = np.append(loss_f_v_history, loss_f_v)
                        loss_history = np.append(loss_history, loss_all)
                        loss_val_history = np.append(loss_val_history, loss_all_val)
                        
                        lamu_history = np.append(lamu_history, lamu, axis = 1)
                        lamv_history = np.append(lamv_history, lamv, axis = 1)
                        
                print('BFGS begins')
                self.optimizer_BFGS.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss,
                                               self.loss_val, self.lambda_u2, self.lambda_v2],
                                    loss_callback = self.callback)

        def callback(self, loss_u, loss_v, loss_f_u, loss_f_v, loss_all, loss_all_val, lamu, lamv):
            global step_Pt
            step_Pt = step_Pt+1
            if step_Pt % 10 == 0:
                global loss_u_history
                global loss_v_history
                global loss_f_u_history
                global loss_f_v_history
                global loss_history
                global loss_val_history
                
                global lamu_history
                global lamv_history

                loss_u_history = np.append(loss_u_history, loss_u)
                loss_v_history = np.append(loss_v_history, loss_v)
                loss_f_u_history = np.append(loss_f_u_history, loss_f_u)
                loss_f_v_history = np.append(loss_f_v_history, loss_f_v)
                loss_history = np.append(loss_history, loss_all)
                loss_val_history = np.append(loss_val_history, loss_all_val)
                
                lamu_history = np.append(lamu_history, lamu, axis = 1)
                lamv_history = np.append(lamv_history, lamv, axis = 1)
                
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
        # inherit eq coeffs(non-zeros) from previous training
        eq_coeff_data = scipy.io.loadmat('DiscLam_ADO.mat')
        
        Lamu_init = eq_coeff_data['Lamu_Disc']
        Lamu_init = np.reshape(Lamu_init[np.nonzero(Lamu_init)], (-1, 1))
        
        Lamv_init = eq_coeff_data['Lamv_Disc']
        Lamv_init = np.reshape(Lamv_init[np.nonzero(Lamv_init)], (-1, 1))

        model = PhysicsInformedNN(X_star_train, u_star_train, v_star_train, X_f, X_star_val, u_star_val, v_star_val, layers, lb, ub, BatchNo, Lamu_init, Lamv_init)
        model.saver.restore(model.sess, './saved_variable_ADO')
        model.train()
        
        saved_path = model.saver.save(model.sess, './saved_variable_ptADO')
        
# =============================================================================
#       check if training efforts are sufficient                    
# =============================================================================
        u_train_pred, v_train_pred = model.predict(X_star_train)
        error_u_train = np.linalg.norm(u_star_train-u_train_pred,2)/np.linalg.norm(u_star_train,2)        
        error_v_train = np.linalg.norm(v_star_train-v_train_pred,2)/np.linalg.norm(v_star_train,2)
        f = open("stdout.txt", "a+")
        f.write('Training Error u: %e \n' % (error_u_train))    
        f.write('Training Error v: %e \n' % (error_v_train))   
        
        u_val_pred, v_val_pred = model.predict(X_star_val)
        error_u_val = np.linalg.norm(u_star_val-u_val_pred,2)/np.linalg.norm(u_star_val,2)        
        error_v_val = np.linalg.norm(v_star_val-v_val_pred,2)/np.linalg.norm(v_star_val,2)
        f.write('Val Error u: %e \n' % (error_u_val))    
        f.write('Val Error v: %e \n' % (error_v_val))   
        
######################## Plots for Pt #################            
        fig = plt.figure()
        plt.plot(loss_u_history[1:])
        plt.plot(loss_v_history[1:])
        plt.legend(['loss_u', 'loss_v'])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_U history of ADO')  
        plt.savefig('1_Pt.png')
        
        fig = plt.figure()
        plt.plot(loss_f_u_history[1:])
        plt.plot(loss_f_v_history[1:])
        plt.legend(['loss_f_u', 'loss_f_v'])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of ADO')     
        plt.savefig('2_Pt.png')
        
        fig = plt.figure()
        plt.plot(loss_history[1:])
        plt.plot(loss_val_history[1:])
        # plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_all_history of ADO')  
        plt.savefig('3_Pt.png')
                                
        fig = plt.figure()
        for i in range(lamu_history.shape[0]):
            plt.plot(lamu_history[i, 1:])
        plt.title('lamu_history')
        plt.savefig('4_Pt.png')
                                                        
        fig = plt.figure()
        for i in range(lamv_history.shape[0]):
            plt.plot(lamv_history[i, 1:])
        plt.title('lamv_history')
        plt.savefig('5_Pt.png')

# =============================================================================
#       compare with ground truth if training efforts are sufficient
# =============================================================================
        u_pred_full, v_pred_full = model.predict(X_star)        
        scipy.io.savemat('PredResponse_full.mat', {'u_pred_full':u_pred_full, 'v_pred_full':v_pred_full})
        
        lambda_u_value, lambda_v_value = model.sess.run([model.lambda_u2, model.lambda_v2])
        lambda_u_value = np.asarray(lambda_u_value)
        lambda_u_true = (np.array([0.1, 0.1, 1, -1, 1, -1, 1])).T
                
        lambda_u_error_vector = np.absolute((lambda_u_true-lambda_u_value)/lambda_u_true)
        error_lambda_u_mean = np.mean(lambda_u_error_vector)
        error_lambda_u_std = np.std(lambda_u_error_vector)
        f.write('lambda_u Mean Error: %.4f \n' % (error_lambda_u_mean))
        f.write('lambda_u Std Error: %.4f \n' % (error_lambda_u_std))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_fun_u_descr)):
            if lambda_u_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda_u_value[i_lib,0]) + model.lib_fun_u_descr[i_lib])
        disc_eq_u = '+'.join(disc_eq_temp)        
        f.write('The discovered equation: u_t = ' + disc_eq_u + '\n')

        lambda_v_value = np.asarray(lambda_v_value)
        lambda_v_true = (np.array([0.1, 0.1, -1, 1, -1, -1, -1])).T
                
        lambda_v_error_vector = np.absolute((lambda_v_true-lambda_v_value)/lambda_v_true)
        error_lambda_v_mean = np.mean(lambda_v_error_vector)
        error_lambda_v_std = np.std(lambda_v_error_vector)
        f.write('lambda_v Mean Error: %.4f \n' % (error_lambda_v_mean))
        f.write('lambda_v Std Error: %.4f \n' % (error_lambda_v_std))
        
        lambda_total_error_vector = np.concatenate((lambda_u_error_vector, lambda_v_error_vector))
        error_lambda_total_mean = np.mean(lambda_total_error_vector)
        error_lambda_total_std = np.std(lambda_total_error_vector)
        f.write('lambda_total Mean Error: %.4f \n' % (error_lambda_total_mean))
        f.write('lambda_total Std Error: %.4f \n' % (error_lambda_total_std))
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_fun_v_descr)):
            if lambda_v_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda_v_value[i_lib,0]) + model.lib_fun_v_descr[i_lib])
        disc_eq_v = '+'.join(disc_eq_temp)        
        f.write('The discovered equation: v_t = ' + disc_eq_v + '\n')
        
        elapsed = time.time() - start_time                
        f.write('Training time: %.4f \n' % (elapsed))
        f.close()
        
        scipy.io.savemat('Results.mat', {'lambda_u_value': lambda_u_value, 'lambda_u_true': lambda_u_true, 'lambda_v_value': lambda_v_value, 'lambda_v_true': lambda_v_true})
        
        scipy.io.savemat('Histories.mat', {'loss_history': loss_history,
                                           'loss_val_history': loss_val_history})        
                        
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