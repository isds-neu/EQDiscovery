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
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # limite all operations on this GPU
with tf.device('/device:GPU:1'): # '/device:GPU:0' is GPU1 in task manager. Likewise, '/device:GPU:1' is GPU0.
            
    # L-BFGS-S loss history
    loss_u_history = np.array([0])
    loss_f_history = np.array([0])
    loss_u_val_history = np.array([0])
    loss_lambda_history = np.array([0])
    step = 0
    
    # Adam loss history
    loss_u_history_Adam = np.array([0])
    loss_f_history_Adam = np.array([0])
    loss_u_val_history_Adam = np.array([0])
    loss_lambda_history_Adam = np.array([0])
    
    # STRidge loss history
    loss_f_history_STRidge = np.array([0])
    loss_lambda_history_STRidge = np.array([0])
    optimaltol_history = np.array([0])   
    tol_history_STRidge = np.array([0])
    lambda_normalized_history_STRidge = np.zeros((16,1))
    lambda_history_STRidge = np.zeros((16,1))
    ridge_u_append_counter_STRidge = np.array([0])
        
    # Loss histories for pretraining
    loss_u_history_Pretrain = np.array([0])
    loss_f_history_Pretrain = np.array([0])
    loss_u_val_history_Pretrain = np.array([0])
    step_Pretrain = 0
    lambda_history_Pretrain = np.zeros((16, 1)) 
    loss_lambda_history_Pretrain = np.array([0])
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, X_val, u_val, X_f, layers_s, layers_i, lb, ub):
            
            self.lb = lb
            self.ub = ub
            
            # Initialize NNs
            self.weights_s, self.biases_s = self.initialize_NN(layers_s) # root NN
            self.weights0, self.biases0 = self.initialize_NN(layers_i) # branch NN 1
            self.weights1, self.biases1 = self.initialize_NN(layers_i) # branch NN 2
            self.weights2, self.biases2 = self.initialize_NN(layers_i) # branch NN 3

            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters
            self.lambda1 = tf.Variable(tf.zeros([16, 1], dtype=tf.float64), dtype=tf.float64,
                                       name = 'lambda')
            
            # Specify the list of trainable variables 
            self.var_list_1 = self.biases0 + self.weights0 + self.biases1 + self.weights1 + self.biases2 + \
                self.weights2 + self.weights_s + self.biases_s
            
            self.var_list_Pretrain = self.var_list_1 + [self.lambda1]
            
            ######### Training data ################
            self.x = X[:,0:1]
            self.t = X[:,1:2]
            self.u0 = u[:, 0:1]
            self.u1 = u[:, 1:2]
            self.u2 = u[:, 2:3]
            self.x_f = X_f[:, 0:1]
            self.t_f = X_f[:, 1:2]

            self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
            self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
            self.u0_tf = tf.placeholder(tf.float64, shape=[None, self.u0.shape[1]])
            self.u1_tf = tf.placeholder(tf.float64, shape=[None, self.u1.shape[1]])
            self.u2_tf = tf.placeholder(tf.float64, shape=[None, self.u2.shape[1]])
            self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])
            
            self.u0_pred = self.net_u(self.x_tf, self.t_tf, 0)
            self.u1_pred = self.net_u(self.x_tf, self.t_tf, 1)
            self.u2_pred = self.net_u(self.x_tf, self.t_tf, 2)
			
            self.f_pred, self.Phi_pred, self.u_t_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.x_f.shape[0])
            self.loss_u = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred)) + \
                    tf.reduce_mean(tf.square(self.u2_tf - self.u2_pred))
            self.loss_f = tf.reduce_mean(tf.square(self.f_pred))
            
            self.loss_lambda = tf.norm(self.lambda1, ord = 1) 
                        
            self.loss_Pre = tf.log(self.loss_u  + 0.01*self.loss_f + 1e-7*self.loss_lambda) # log loss
            self.loss = tf.log(self.loss_u  + 0.1*self.loss_f) # log loss
            
            ######### Validation data ################
            self.x_val = X_val[:,0:1]
            self.t_val = X_val[:,1:2]
            self.u0_val = u_val[:, 0:1]
            self.u1_val = u_val[:, 1:2]
            self.u2_val = u_val[:, 2:3]
            
            self.x_val_tf = tf.placeholder(tf.float64, shape=[None, 1])
            self.t_val_tf = tf.placeholder(tf.float64, shape=[None, 1])
            self.u0_val_tf = tf.placeholder(tf.float64, shape=[None, 1])
            self.u1_val_tf = tf.placeholder(tf.float64, shape=[None, 1])
            self.u2_val_tf = tf.placeholder(tf.float64, shape=[None, 1])
                    
            self.u0_val_pred = self.net_u(self.x_val_tf, self.t_val_tf, 0)
            self.u1_val_pred = self.net_u(self.x_val_tf, self.t_val_tf, 1)
            self.u2_val_pred = self.net_u(self.x_val_tf, self.t_val_tf, 2)
            
            self.loss_u_val = tf.reduce_mean(tf.square(self.u0_val_tf - self.u0_val_pred)) + \
                tf.reduce_mean(tf.square(self.u1_val_tf - self.u1_val_pred)) + \
                    tf.reduce_mean(tf.square(self.u2_val_tf - self.u2_val_pred))
                        
            ######### Optimizor #########################
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = self.var_list_1,
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 1000, #10000
                                                                               'maxfun': 1000, # 10000
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 1.0 * np.finfo(float).eps})
    
            self.optimizer_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss_Pre, 
                                                                    var_list = self.var_list_Pretrain,
                                                                    method = 'L-BFGS-B', 
                                                                   options = {'maxiter': 80000, #10000
                                                                               'maxfun': 80000, #10000
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 1.0 * np.finfo(float).eps})
                             
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 1e-5
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.5,
                                                        staircase=True)

            # The default settings: learning rate = 1e-3, beta1 = 0.9, beta2 = 0.999， epsilon = 1e-8
            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = self.var_list_1,
                                                               global_step = self.global_step)
            
            self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u0_tf: self.u0, self.u1_tf: self.u1,
                            self.u2_tf: self.u2,
                           self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                           self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.u0_val_tf: self.u0_val,
                           self.u1_val_tf: self.u1_val, self.u2_val_tf: self.u2_val}
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
    
        def initialize_NN(self, layers):        
            weights = []
            biases = []
            num_layers = len(layers) 
            for l in range(0,num_layers-1):
                W = self.xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64, name = 'b')
                weights.append(W)
                biases.append(b)        
            return weights, biases
            
        def xavier_init(self, size):
            in_dim = size[0]
            out_dim = size[1]        
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), name = 'W')
        
        def neural_net(self, X, weights, biases, si_flag):
            num_layers = len(weights) + 1    
            if si_flag:
                H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # input to root NN
            else:
                H = X # input to branch NN
            for l in range(0,num_layers-2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = weights[-1]
            b = biases[-1]
            if si_flag:
                Y = tf.tanh(tf.add(tf.matmul(H, W), b)) # output from root NN
            else:
                Y = tf.add(tf.matmul(H, W), b) # output from branch NN
            return Y
        
        def net_u(self, x, t, IBC_flag):  
            u_int = self.neural_net(tf.concat([x,t],1), self.weights_s, self.biases_s, True) # root NN
            if IBC_flag == 0:
                u = self.neural_net(u_int, self.weights0, self.biases0, False) # branch NN 1
            elif IBC_flag == 1:
                u = self.neural_net(u_int, self.weights1, self.biases1, False) # branch NN 2
            elif IBC_flag == 2:
                u = self.neural_net(u_int, self.weights2, self.biases2, False) # branch NN 3
            return u
                       
        def net_f(self, x, t, N_f):
            u0 = self.net_u(x, t, 0)
            u1 = self.net_u(x, t, 1)
            u2 = self.net_u(x, t, 2)
            u0_t = tf.gradients(u0, t)[0]
            u0_x = tf.gradients(u0, x)[0]
            u0_xx = tf.gradients(u0_x, x)[0]
            u0_xxx = tf.gradients(u0_xx, x)[0]  
 
            u1_t = tf.gradients(u1, t)[0]
            u1_x = tf.gradients(u1, x)[0]
            u1_xx = tf.gradients(u1_x, x)[0]
            u1_xxx = tf.gradients(u1_xx, x)[0]  

            u2_t = tf.gradients(u2, t)[0]
            u2_x = tf.gradients(u2, x)[0]
            u2_xx = tf.gradients(u2_x, x)[0]
            u2_xxx = tf.gradients(u2_xx, x)[0]  

            u = tf.concat((u0, u1, u2), 0)
            u_t = tf.concat((u0_t, u1_t, u2_t), 0)
            u_x = tf.concat((u0_x, u1_x, u2_x), 0)
            u_xx = tf.concat((u0_xx, u1_xx, u2_xx), 0)
            u_xxx = tf.concat((u0_xxx, u1_xxx, u2_xxx), 0)
            Phi = tf.concat([tf.constant(1, shape=[3*N_f, 1], dtype=tf.float64), u, u**2, u**3, u_x, u*u_x, u**2*u_x,
                                  u**3*u_x, u_xx, u*u_xx, u**2*u_xx, u**3*u_xx, u_xxx, u*u_xxx, u**2*u_xxx, u**3*u_xxx], 1)            
            self.lib_descr = ['1', 'u', 'u**2', 'u**3', 'u_x', 'u*u_x', 'u**2*u_x',
                                  'u**3*u_x', 'u_xx', 'u*u_xx', 'u**2*u_xx', 'u**3*u_xx', 'u_xxx', 'u*u_xxx', 'u**2*u_xxx', 'u**3*u_xxx']
            f = tf.matmul(Phi, self.lambda1) - u_t      
            return f, Phi, u_t
        
        def callback(self, loss_u, loss_f, loss_u_val, loss_lambda):
            global step
            if step%10 == 0:
                global loss_u_history
                global loss_f_history
                global loss_u_val_history
                global loss_lambda_history
                
                loss_u_history = np.append(loss_u_history, loss_u)
                loss_f_history = np.append(loss_f_history, loss_f)
                loss_u_val_history = np.append(loss_u_val_history, loss_u_val)
                loss_lambda_history = np.append(loss_lambda_history, loss_lambda)
                
            step = step+1

                
        def callback_Pretrain(self, loss_u, loss_f, lambda_u, loss_u_val, loss_lambda):
            global step_Pretrain
            if step_Pretrain % 10 == 0:
                
                global loss_u_history_Pretrain
                global loss_f_history_Pretrain
                global loss_u_val_history_Pretrain                
                global lambda_history_Pretrain
                global loss_lambda_history_Pretrain
                
                loss_u_history_Pretrain = np.append(loss_u_history_Pretrain, loss_u)
                loss_f_history_Pretrain = np.append(loss_f_history_Pretrain, loss_f)
                loss_u_val_history_Pretrain = np.append(loss_u_val_history_Pretrain, loss_u_val)                
                lambda_history_Pretrain = np.append(lambda_history_Pretrain, lambda_u, axis = 1)
                loss_lambda_history_Pretrain = np.append(loss_lambda_history_Pretrain, loss_lambda)                
                
            step_Pretrain += 1


        def train(self, nIter): # nIter is the number of outer loop           
            # Pretraining,as a form of a good intialization
            print('L-BFGS-B pretraining begins')
            self.optimizer_Pretrain.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_f, self.lambda1, self.loss_u_val, self.loss_lambda],
                                    loss_callback = self.callback_Pretrain)
            
            saver_Pre = tf.train.Saver(self.var_list_Pretrain)          
            saved_path = saver_Pre.save(self.sess, './saved_variable_Pre')

            for self.it in range(nIter):
    
                # Loop of STRidge optimization
                print('STRidge begins')
                self.callTrainSTRidge()

                # Loop of Adam optimization
                print('Adam begins')                                            
                for it_Adam in tqdm(range(1000)): # 10000                    
                    self.sess.run(self.train_op_Adam, self.tf_dict)                    
                    # Print
                    if it_Adam % 10 == 0:
                        loss_u, loss_f, loss_u_val = self.sess.run([self.loss_u, self.loss_f, self.loss_u_val], 
                                                                   self.tf_dict)
                        loss_lambda = self.sess.run(self.loss_lambda)
                        
                        global loss_u_history_Adam
                        global loss_f_history_Adam
                        global loss_u_val_history_Adam
                        global loss_lambda_history_Adam
                        
                        loss_u_history_Adam = np.append(loss_u_history_Adam, loss_u)
                        loss_f_history_Adam = np.append(loss_f_history_Adam, loss_f)
                        loss_u_val_history_Adam = np.append(loss_u_val_history_Adam, loss_u_val)
                        loss_lambda_history_Adam = np.append(loss_lambda_history_Adam, loss_lambda)
                
                # Loop of L-BFGS-B optimization
                print('L-BFGS-B begins')
                self.optimizer.minimize(self.sess,
                                        feed_dict = self.tf_dict,
                                        fetches = [self.loss_u, self.loss_f, self.loss_u_val, self.loss_lambda],
                                        loss_callback = self.callback)   
                
            saver = tf.train.Saver(self.var_list_1)          
            saved_path = saver.save(self.sess, './saved_variable_ADO')
        
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
            
            u0 = self.sess.run(self.u0_pred, tf_dict)
            u1 = self.sess.run(self.u1_pred, tf_dict)
            u2 = self.sess.run(self.u2_pred, tf_dict)
            return u0, u1, u2
        
        def callTrainSTRidge(self):
            lam = 1e-5
            d_tol = 1
            maxit = 100
            STR_iters = 10
            l0_penalty = None
            
            normalize = 2
            split = 0.8
            print_best_tol = False     
            Phi_pred, u_t_pred = self.sess.run([self.Phi_pred, self.u_t_pred], self.tf_dict)
            
            lambda2 = self.TrainSTRidge(Phi_pred, u_t_pred, lam, d_tol, maxit, STR_iters, l0_penalty, normalize, split)     
            
            self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype = tf.float64))
                    
        def TrainSTRidge(self, R0, Ut, lam, d_tol, maxit, STR_iters = 10, l0_penalty = None, normalize = 2,
                         split = 0.8):   
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================                     
            # First normalize data 
            n,d = R0.shape
            R = np.zeros((n,d), dtype=np.float64)
            if normalize != 0:
                Mreg = np.zeros((d,1))
                for i in range(0,d):
                    Mreg[i] = 1.0/(np.linalg.norm(R0[:,i],normalize))
                    R[:,i] = Mreg[i]*R0[:,i]
            else: R = R0
                        
            # Split data into 80% training and 20% test, then search for the best tolderance.
            np.random.seed(0) # for consistancy
            n,_ = R.shape
            train = np.random.choice(n, int(n*split), replace = False)
            test = [i for i in np.arange(n) if i not in train]
            TrainR = R[train,:]
            TestR = R[test,:]
            TrainY = Ut[train,:]
            TestY = Ut[test,:]
        
            # Set up the initial tolerance and l0 penalty
            d_tol = float(d_tol)
            tol = d_tol
            lambda_u = self.sess.run(self.lambda1)
            w_best = lambda_u/Mreg
            
            # err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
            err_f = np.mean((TestY - TestR.dot(w_best))**2)
            
            if l0_penalty == None and self.it == 0: 
                # l0_penalty = 0.05*np.linalg.cond(R)
                self.l0_penalty_0 = err_f
                l0_penalty = self.l0_penalty_0
            elif l0_penalty == None:
                l0_penalty = self.l0_penalty_0

            err_lambda = l0_penalty*np.count_nonzero(w_best)
            err_best = err_f + err_lambda
            tol_best = 0
                        
            global loss_f_history_STRidge
            global loss_lambda_history_STRidge
            global tol_history_STRidge
            global lambda_normalized_history_STRidge
            
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
            loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
            tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
            lambda_normalized_history_STRidge = np.append(lambda_normalized_history_STRidge, np.reshape(w_best,
                                                        (-1, 1)), axis = 1)
        
            # Now increase tolerance until test performance decreases
            for iter in range(maxit):
                # Get a set of coefficients and error
                w = self.STRidge(TrainR, TrainY, lam, STR_iters, tol, Mreg, normalize = normalize)
                
                # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
                err_f = np.mean((TestY - TestR.dot(w))**2)
                
                err_lambda = l0_penalty*np.count_nonzero(w)
                err = err_f + err_lambda
        
                # Has the accuracy improved?
                if err <= err_best:
                    err_best = err
                    w_best = w
                    tol_best = tol
                    tol = tol + d_tol
                    
                    loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
                    loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
                    tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
                    lambda_normalized_history_STRidge = np.append(lambda_normalized_history_STRidge,
                                                                  np.reshape(w_best, (-1, 1)), axis = 1)                    

                else:
                    tol = max([0,tol - 2*d_tol])
                    d_tol = d_tol/1.618
                    tol = tol + d_tol
                    
            global optimaltol_history
            optimaltol_history = np.append(optimaltol_history, tol_best)
            
            if normalize != 0:
                w_best = np.multiply(Mreg, w_best)
                        
            return np.real(w_best)     
        
        def STRidge(self, X0, y, lam, STR_iters, tol, Mreg, normalize = 2):        
            n,d = X0.shape            
            X = X0            
            w = self.sess.run(self.lambda1)/Mreg           
            num_relevant = d
            
            # norm threshold
            biginds = np.where(abs(w) > tol)[0]
            
            global lambda_history_STRidge
            global ridge_u_append_counter_STRidge
            lambda_history_STRidge = np.append(lambda_history_STRidge, np.reshape(w*Mreg, (-1, 1)), axis = 1)
            ridge_u_append_counter = 1

            # Threshold and continue
            for j in range(STR_iters):        
                # Figure out which items to cut out
                
                # norm threshold
                smallinds = np.where(abs(w) < tol)[0]
                
                new_biginds = [i for i in range(d) if i not in smallinds]
                    
                # If nothing changes then stop
                if num_relevant == len(new_biginds): break
                else: num_relevant = len(new_biginds)
                    
                # Also make sure we didn't just lose all the coefficients
                if len(new_biginds) == 0:
                    if j == 0: 
                        ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
                        return w
                    else: break
                biginds = new_biginds
                
                # Otherwise get a new guess
                w[smallinds] = 0
                
                if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + \
                                                          lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
                else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
                lambda_history_STRidge = np.append(lambda_history_STRidge, np.reshape(w*Mreg, (-1, 1)), axis = 1)
                ridge_u_append_counter += 1

            # Now that we have the sparsity pattern, use standard least squares to get w
            if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
            lambda_history_STRidge = np.append(lambda_history_STRidge, np.reshape(w*Mreg, (-1, 1)), axis = 1)
            ridge_u_append_counter += 1
            
            ridge_u_append_counter_STRidge = np.append(ridge_u_append_counter_STRidge, ridge_u_append_counter)
            return w
        
    if __name__ == "__main__":                 
        
        start_time = time.time()
        
        layers_s = [2, 20, 20, 20, 20] # root NN
        layers_i = [20, 30, 30, 30, 30, 1] # branch NNs
        
# =============================================================================
#         load data
# =============================================================================       
        # data_Sine = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '\\Burgers_SineIC_new.mat')
        data_Sine = scipy.io.loadmat('Burgers_SineIC_new.mat')
        t = np.real(data_Sine['t'].flatten()[:,None])
        x = np.real(data_Sine['x'].flatten()[:,None])
        Exact_Sine = np.real(data_Sine['u'])
  
        # data_Cube = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '\\Burgers_CubeIC_new.mat')
        data_Cube = scipy.io.loadmat('Burgers_CubeIC_new.mat')
        Exact_Cube = np.real(data_Cube['u'])
        
        # data_Gauss = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '\\Burgers_GaussIC_new.mat')
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

        # Training measurements, which are randomly sampled spatio-temporally
        Split_TrainVal = 0.8
        N_u_train = int(X_u_meas.shape[0]*Split_TrainVal)
        idx_train = np.arange(N_u_train)
        X_u_train = X_u_meas[idx_train,:]
        u_train = u_meas[idx_train,:]
        
        # Validation Measurements, which are the rest of measurements
        idx_val = np.setdiff1d(np.arange(X_u_meas.shape[0]), idx_train, assume_unique=True)
        X_u_val = X_u_meas[idx_val,:]
        u_val = u_meas[idx_val,:]
                
        # Collocation points
        N_f = 45000    
        X_f_train = lb + (ub-lb)*lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_meas))
        
        # Option: Add noise
        noise = 0.1
        u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        u_val = u_val + noise*np.std(u_val)*np.random.randn(u_val.shape[0], u_val.shape[1])
        
# =============================================================================
#       train model
# =============================================================================
        model = PhysicsInformedNN(X_u_train, u_train, X_u_val, u_val, X_f_train, layers_s, layers_i, lb, ub)
        model.train(6)
        
# =============================================================================
#         check if training efforts are sufficient
# =============================================================================
        f = open("stdout.txt", "a+")
        
        elapsed = time.time() - start_time                
        f.write('Training time: %.4f \n' % (elapsed))

        u_train_Pred_Sin, u_train_Pred_Cube, u_train_Pred_Gauss = model.predict(X_u_train)                
        u_train_Pred = np.hstack((u_train_Pred_Sin, u_train_Pred_Cube, u_train_Pred_Gauss))
        Error_u_Train = np.linalg.norm(u_train-u_train_Pred,2)/np.linalg.norm(u_train,2)   
        f.write('Training Error u: %e \n' % (Error_u_Train))     
        
        u_val_Pred_Sin, u_val_Pred_Cube, u_val_Pred_Gauss = model.predict(X_u_val)                
        u_val_Pred = np.hstack((u_val_Pred_Sin, u_val_Pred_Cube, u_val_Pred_Gauss))
        Error_u_val = np.linalg.norm(u_val-u_val_Pred,2)/np.linalg.norm(u_val,2)   
        f.write('Validation Error u: %e \n' % (Error_u_val))        

        ######################## Plots for BFGS(Pretraining) #################        
        fig = plt.figure()
        p1 = plt.plot(loss_u_history_Pretrain[1:])
        p2 = plt.plot(loss_u_val_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of BFGS(Pretraining)')  
        plt.legend((p1[0], p2[0]), ('tr', 'val'))
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS(Pretraining)')     
        plt.savefig('2.png')

        fig = plt.figure()
        plt.plot(loss_lambda_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda history of BFGS(Pretraining)')     
        plt.savefig('3.png')
        ######################## Plots for Adam #################        
        fig = plt.figure()
        p1 = plt.plot(loss_u_history_Adam[1:])
        p2 = plt.plot(loss_u_val_history_Adam[1:])
        plt.legend((p1[0], p2[0]), ('tr', 'val'))
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of Adam')  
        plt.savefig('4.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of Adam')  
        plt.savefig('5.png')

        fig = plt.figure()
        plt.plot(loss_lambda_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda history of Adam')     
        plt.savefig('6.png')
                        
        ######################## Plots for BFGS #################            
        fig = plt.figure()
        p1 = plt.plot(loss_u_history[1:])
        p2 = plt.plot(loss_u_val_history[1:])
        plt.legend((p1[0], p2[0]), ('tr', 'val'))
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of BFGS')  
        plt.savefig('7.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS')     
        plt.savefig('8.png')
                
        fig = plt.figure()
        plt.plot(loss_lambda_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda history of BFGS')     
        plt.savefig('9.png')
        ########################## Plots for STRidge #######################        
        fig = plt.figure()
        plt.plot(loss_f_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f history of STRidge')  
        plt.savefig('10.png')
        
        fig = plt.figure()
        plt.plot(loss_lambda_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda history of STRidge')
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(tol_history_STRidge[1:])
        plt.title('Tolerance History ')
        plt.savefig('12.png')
        
        fig = plt.figure()
        plt.plot(optimaltol_history[1:])
        plt.title('History of Optimal Tolerance')
        plt.savefig('13.png')     
        
        fig = plt.figure()
        for i in range(lambda_normalized_history_STRidge.shape[0]):
            plt.plot(lambda_normalized_history_STRidge[i, 1:])
        plt.title('lambda_normalized_history_STRidge')
        plt.savefig('14.png')

# =============================================================================
#       compute full-field response and Lambda if training efforts are sufficient
# =============================================================================
        # full-field response
        u_Full_Pred_Sin, u_Full_Pred_Cube, u_Full_Pred_Gauss = model.predict(X_star)  
        u_Full_Pred = np.hstack((u_Full_Pred_Sin, u_Full_Pred_Cube, u_Full_Pred_Gauss))
        u_star = np.hstack((u_star_Sin, u_star_Cube, u_star_Gauss))
        error_u = np.linalg.norm(u_star-u_Full_Pred,2)/np.linalg.norm(u_star,2)   
        f.write('Full Field Error u: %e \n' % (error_u))    
          
        # Lambda
        lambda1_value = model.sess.run(model.lambda1)
        lambda1_true = np.zeros((16,1))
        lambda1_true[5] = -1
        lambda1_true[8] = 0.01/3.1415926
        
        lambda5_error = np.abs((lambda1_true[5]-lambda1_value[5])/lambda1_true[5])*100
        lambda8_error = np.abs((lambda1_true[8]-lambda1_value[8])/lambda1_true[8])*100
        f.write('lambda5_error: %.2f%% \n' % (lambda5_error))
        f.write('lambda8_error: %.2f%% \n' % (lambda8_error))
        
        lambda_error = np.linalg.norm(lambda1_true-lambda1_value,2)/np.linalg.norm(lambda1_true,2)
        f.write('Lambda L2 Error: %e' % (lambda_error))   
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_descr)):
            if lambda1_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda1_value[i_lib,0]) + model.lib_descr[i_lib])
        disc_eq = '+'.join(disc_eq_temp)        
        f.write('The discovered equation: u_t = ' + disc_eq)
        
        f.close()
        scipy.io.savemat('DiscLam_ADO.mat', {'Lamu_True': lambda1_true, 'Lamu_Disc': lambda1_value})
        scipy.io.savemat('Lam_Plot.mat', {'lambda_history_STRidge': lambda_history_STRidge,
                                          'lambda_history_Pretrain': lambda_history_Pretrain,
                                          'ridge_u_append_counter_STRidge': ridge_u_append_counter_STRidge})

        ####################### Plot ######################################
        ## Sine
        # plot the whole domain 
        U_pred = griddata(X_star, u_Full_Pred_Sin.flatten(), (X, T), method='cubic')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)    
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Model Result: Sine')       
        plt.savefig('15.png')
        U_pred_sine = U_pred
        
        # plot the whole domain truth
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, Exact_Sine, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Ground Truth: Sine')
        plt.savefig('16.png')

        ## Cube
        # plot the whole domain 
        U_pred = griddata(X_star, u_Full_Pred_Cube.flatten(), (X, T), method='cubic')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)    
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Model Result: Cube')       
        plt.savefig('17.png')
        U_pred_cos = U_pred
        
        # plot the whole domain truth
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, Exact_Cube, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Ground Truth: Cube')
        plt.savefig('18.png')

        ## Sine
        # plot the whole domain 
        U_pred = griddata(X_star, u_Full_Pred_Gauss.flatten(), (X, T), method='cubic')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_pred, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)    
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Model Result: Gauss')       
        plt.savefig('19.png')
        U_pred_Gauss = U_pred
        
        # plot the whole domain truth
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, Exact_Gauss, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.title('Ground Truth: Gauss')
        plt.savefig('20.png')
        
        scipy.io.savemat('PredSol.mat', {'u_pred_sin': U_pred_sine, 'u_pred_cos': U_pred_cos,
                                 'u_pred_gauss': U_pred_Gauss})
                        
        ########################## Plots for Lambda ########################
        fig = plt.figure()
        plt.plot(lambda1_true)
        plt.plot(lambda1_value)
        plt.title('lambda values')
        plt.legend(['the true', 'the identified'])
        plt.savefig('21.png')