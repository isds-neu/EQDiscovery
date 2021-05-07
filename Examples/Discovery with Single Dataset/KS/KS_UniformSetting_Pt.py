# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# =============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from scipy.spatial import distance
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.gridspec as gridspec
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import sobol_seq
from pyDOE import lhs
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # limite all operations on this GPU

with tf.device('/device:GPU:0'):
    
    # pt loss history
    loss_history = np.array([0])
    loss_u_history = np.array([0])
    loss_f_history = np.array([0])
    lambda1_history = np.zeros((3,1))    
    loss_history_val = np.array([0])
    loss_u_history_val = np.array([0])
    loss_f_history_val = np.array([0])
    
    step = 0
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, X_f, X_val, u_val, layers, lb, ub, Lamu_init): 
            self.lb = lb
            self.ub = ub
            self.layers = layers
            
            # Initialize NNs and lambda1
            self.weights, self.biases = self.initialize_NN(layers)
            self.lambda1 = tf.Variable(tf.zeros([36, 1], dtype=tf.float32), dtype=tf.float32, name = 'lambda')
            self.lambda2 = tf.Variable(Lamu_init, dtype=tf.float32)            
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
                        
            # Specify the list of trainable variables 
            var_list_1 = self.biases + self.weights
            
            self.var_list_Pretrain = self.biases + self.weights
            self.var_list_Pretrain.append(self.lambda1)
            
            self.var_list_Pt = self.biases + self.weights + [self.lambda2]
            
            ######### Training data ################
            self.x = X[:,0:1]
            self.t = X[:,1:2]
            self.u = u
            self.x_f = X_f[:, 0:1]
            self.t_f = X_f[:, 1:2]
            
            self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
            self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
            self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
            
            self.u_pred = self.net_u(self.x_tf, self.t_tf)
            self.f_pred, self.Phi_pred, self.u_t_pred = self.net_f(self.x_f_tf, self.t_f_tf, X_f.shape[0])
            
            self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
            self.loss_f_coeff_tf = tf.placeholder(tf.float32)
            self.loss_f = self.loss_f_coeff_tf*tf.reduce_mean(tf.square(self.f_pred))
            self.loss = tf.log(self.loss_u  + self.loss_f) # log loss
            
            ######### Validation data ################
            self.x_val = X_val[:,0:1]
            self.t_val = X_val[:,1:2]
            self.u_val = u_val
            
            self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
            self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
            self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
                    
            self.u_val_pred = self.net_u(self.x_val_tf, self.t_val_tf)
            self.f_val_pred, _, _ = self.net_f(self.x_val_tf, self.t_val_tf, u_val.shape[0])
            
            self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
            self.loss_f_val = tf.reduce_mean(tf.square(self.f_val_pred))
            self.loss_val = tf.log(self.loss_u_val  + self.loss_f_val) # log loss
                        
            ######### Optimizor #########################
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = self.var_list_Pt,
#                                                                    L-BFGS-B
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 20000,
                                                                               'maxfun': 20000,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 0.1 * np.finfo(float).eps})
                                                                    
            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = 1e-3) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = self.var_list_Pt)          
            
            
            self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.u_val_tf: self.u_val}
            
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
                
        def net_u(self, x, t):  
            u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
            return u
        
        def net_f(self, x, t, N_f):
            u = self.net_u(x,t)
            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            u_xxx = tf.gradients(u_xx, x)[0]  
            u_xxxx = tf.gradients(u_xxx, x)[0]
            
            Phi = tf.concat([u*u_x, u_xx, u_xxxx], 1)            
            self.library_description = ['u*u_x', 'u_xx', 'u_xxxx']
            
            f = tf.matmul(Phi, self.lambda2) - u_t
            
            return f, Phi, u_t
        
        def callback(self, loss, lambda1, loss_u, loss_f, loss_val, loss_u_val, loss_f_val):
            global step
            step = step+1
            if step % 10 == 0:
                
                global loss_history
                global lambda1_history
                global loss_u_history
                global loss_f_history
                global loss_history_val
                global loss_u_history_val
                global loss_f_history_val
                
                loss_history = np.append(loss_history, loss)
                lambda1_history = np.append(lambda1_history, lambda1, axis=1)
                loss_u_history = np.append(loss_u_history, loss_u)
                loss_f_history = np.append(loss_f_history, loss_f)
                loss_history_val = np.append(loss_history_val, loss_val)
                loss_u_history_val = np.append(loss_u_history_val, loss_u_val)
                loss_f_history_val = np.append(loss_f_history_val, loss_f_val)

        def train(self):            
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            saver = tf.train.Saver(self.var_list_Pretrain)            
            saver.restore(self.sess, './saved_variable_ADO')

            self.tf_dict[self.loss_f_coeff_tf] = 10
            
            # Adam 1
            print('Adam starts')
            for it in tqdm(range(20000)):
                self.sess.run(self.train_op_Adam, feed_dict=self.tf_dict, options=run_options)
                
                # Print
                if it % 10 == 0:
                    loss, loss_u, loss_f, lambda1_value, loss_val, loss_u_val, loss_f_val = self.sess.run([self.loss, self.loss_u, self.loss_f, self.lambda1, self.loss_val, self.loss_u_val, self.loss_f_val], self.tf_dict)
                    lambda1 = self.sess.run(self.lambda2)
                    
                    global loss_history
                    global lambda1_history
                    global loss_u_history
                    global loss_f_history
                    global loss_history_val
                    global loss_u_history_val
                    global loss_f_history_val
                    
                    loss_history = np.append(loss_history, loss)
                    lambda1_history = np.append(lambda1_history, lambda1, axis=1)
                    loss_u_history = np.append(loss_u_history, loss_u)
                    loss_f_history = np.append(loss_f_history, loss_f)
                    loss_history_val = np.append(loss_history_val, loss_val)
                    loss_u_history_val = np.append(loss_u_history_val, loss_u_val)
                    loss_f_history_val = np.append(loss_f_history_val, loss_f_val)
                                    
                        
                
                # L-BFGS-B
            print('BFGS or L-BFGS-B starts')
            self.optimizer.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss, self.lambda2, self.loss_u, self.loss_f, self.loss_val,
                                                self.loss_u_val, self.loss_f_val],
                                    loss_callback = self.callback)      
                
                
            saved_path = saver.save(self.sess, './saved_variable_Pt')
            # saver.restore(self.sess, './saved_variable_Pt')
                                
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
            
            u_star = self.sess.run(self.u_pred, tf_dict)
            return u_star    
        
    if __name__ == "__main__": 
         
        start_time = time.time()
        
        layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 1]
# =============================================================================
#         load data
# =============================================================================
        data = scipy.io.loadmat('kuramoto_sivishinky.mat') # course temporal grid
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        Exact = np.real(data['u']).T
        
        X, T = np.meshgrid(x,t)
        
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = Exact.flatten()[:,None]              
    
        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)    
        
        # In this case, measurements are from N_u_s points.
        N_u_s = 320 
        idx_s = np.random.choice(x.shape[0], N_u_s, replace=False)
        
        # Resample at a low temporal sampling rate.
        N_u_t = 101 
        idx_t = np.random.choice(t.shape[0], N_u_t, replace=False)
        idx_t = idx_t.astype(np.int32)
        
        X1 = X[:, idx_s]
        X2 = X1[idx_t, :]
        T1 = T[:, idx_s]
        T2 = T1[idx_t, :]
        Exact1 = Exact[:, idx_s]
        Exact2 = Exact1[idx_t, :]
        
        X_u_meas = np.hstack((X2.flatten()[:,None], T2.flatten()[:,None]))
        u_meas = Exact2.flatten()[:,None]   
        
        # Training measurements, which are randomly sampled spatio-temporally
        Split_TrainVal = 0.8
        N_u_train = int(N_u_s*N_u_t*Split_TrainVal)
        idx_train = np.random.choice(X_u_meas.shape[0], N_u_train, replace=False)
        X_u_train = X_u_meas[idx_train,:]
        u_train = u_meas[idx_train,:]
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_u_meas.shape[0]), idx_train, assume_unique=True)
        X_u_val = X_u_meas[idx_val,:]
        u_val = u_meas[idx_val,:]
        
        
        # Collocation points
        N_f = 20000 
        X_f_train = lb + (ub-lb)*lhs(2, N_f)   
        X_f_train = np.vstack((X_f_train, X_u_train))
        
        # Add Noise
        noise = 0.1    
        u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
        
# =============================================================================
#         train model
# =============================================================================
        # inherit eq coeffs(non-zeros) from previous training
        eq_coeff_data = scipy.io.loadmat('DiscLam_ADO.mat')
        
        Lamu_init = eq_coeff_data['Lamu_Disc']
        Lamu_init = np.reshape(Lamu_init[np.nonzero(Lamu_init)], (-1, 1))
        
        model = PhysicsInformedNN(X_u_train, u_train, X_f_train, X_u_val, u_val, layers, lb, ub, Lamu_init)
                
        model.train()
        
# =============================================================================
#         check if training efforts are sufficient
# =============================================================================
        elapsed = time.time() - start_time     
        f = open("stdout.txt", "a+")    
                   
        f.write('Training time: %.4f \n' % (elapsed))
        
        u_train_Pred = model.predict(X_u_train)                
        Error_u_Train = np.linalg.norm(u_train-u_train_Pred,2)/np.linalg.norm(u_train,2)   
        f.write('Training Error u: %e \n' % (Error_u_Train))     
        
        u_val_Pred = model.predict(X_u_val)                
        Error_u_Val = np.linalg.norm(u_val-u_val_Pred,2)/np.linalg.norm(u_val,2)   
        f.write('Validation Error u: %e \n' % (Error_u_Val))      
                        
        ######################## Plots for Pt #################
        fig = plt.figure()
        plt.plot(loss_history)
        plt.xlabel('10x')
        plt.title('log loss history of BFGS')
        plt.savefig('22.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of BFGS')  
        plt.savefig('23.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS')     
        plt.savefig('24.png')
        
        fig = plt.figure()
        for i in range(lambda1_history.shape[0]):
            plt.plot(lambda1_history[i, 1:])
        plt.title('lambda1_history')
        plt.savefig('25.png')          

        fig = plt.figure()
        plt.plot(loss_history_val)
        plt.xlabel('10x')
        plt.title('log loss_val history of BFGS')
        plt.savefig('26.png')
        
        fig = plt.figure()
        plt.plot(loss_u_history_val)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u_val history of BFGS')  
        plt.savefig('27.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_val)
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f_val history of BFGS')
        plt.savefig('28.png')
        
# =============================================================================
#       compare with the ground truth if training efforts are sufficient
# =============================================================================
        u_FullField_Pred = model.predict(X_star)                
        error_u = np.linalg.norm(u_star-u_FullField_Pred,2)/np.linalg.norm(u_star,2)   
        f.write('Full Field Error u: %e \n' % (error_u))    
        
        scipy.io.savemat('Pred.mat',{'u_FullField_Pred':u_FullField_Pred})
        
#        Save trained weights, biases and lambda
        lambda1_trained, weights_trained, biases_trained = model.sess.run([model.lambda2, model.weights, model.biases])
        
        lambda1_value = lambda1_trained        
        lambda1_true = np.zeros((3, 1))
        lambda1_true[0] = -1 # uu_x
        lambda1_true[1] = -1 # u_xx
        lambda1_true[2] = -1 # u_xxxx
        cosine_similarity = 1-distance.cosine(lambda1_true, lambda1_value)
        f.write('Cosine similarity of lambda: %.2f \n' % (cosine_similarity))     
        
        lambda7_error = np.abs((lambda1_true[0]-lambda1_value[0])/lambda1_true[0])*100
        lambda12_error = np.abs((lambda1_true[1]-lambda1_value[1])/lambda1_true[1])*100
        lambda24_error = np.abs((lambda1_true[2]-lambda1_value[2])/lambda1_true[2])*100
        
        f.write('lambda7_error: %.2f%% \n' % (lambda7_error))
        f.write('lambda12_error: %.2f%% \n' % (lambda12_error))
        f.write('lambda24_error: %.2f%% \n' % (lambda24_error))
        
        lambda_error_mean = np.mean(np.array([lambda7_error, lambda12_error, lambda24_error]))
        lambda_error_std = np.std(np.array([lambda7_error, lambda12_error, lambda24_error]))
        f.write('lambda_error_mean: %.2f%% \n' % (lambda_error_mean))
        f.write('lambda_error_std: %.2f%% \n' % (lambda_error_std))
        
        lambda_error = np.linalg.norm(lambda1_true-lambda1_value,2)/np.linalg.norm(lambda1_true,2)
        f.write('Lambda L2 Error: %e \n' % (lambda_error))   
        
        # disc_eq_temp = []
        # for i_lib in range(len(model.library_description)):
        #     if lambda1_value[i_lib] != 0:
        #         disc_eq_temp.append(str(lambda1_value[i_lib,0]) + model.library_description[i_lib])
        # disc_eq = '+'.join(disc_eq_temp)        
        # print('The discovered equation: u_t = ' + disc_eq)       
        
        f.close()
        
        scipy.io.savemat('History.mat', {'lambda1_history': lambda1_history,
                                         'lambda1_value': lambda1_value})
    
        ####################### Plot ######################################
        # plot the whole domain 
        U_pred = griddata(X_star, u_FullField_Pred.flatten(), (X, T), method='cubic')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)    
        plt.title('Model Result')        
        plt.savefig('34.png')
        
        # plot the whole domain truth
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, Exact, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        plt.title('Ground Truth')
        plt.savefig('35.png')
        
        
        ############################### Plot for Lambda ####################
        fig = plt.figure()
        plt.plot(lambda1_value, 'o-')
        plt.plot(lambda1_true)
        plt.legend(('the identified', 'the true'))
        plt.title('Identified $\lambda$')
        plt.savefig('36.png')
        