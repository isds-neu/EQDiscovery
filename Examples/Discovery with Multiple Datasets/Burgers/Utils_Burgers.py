# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the discovery of Burgers equation with multiple datasets 
# =============================================================================

import numpy as np
import tensorflow as tf # tensorflow version 1.15.0
from tqdm import tqdm
import matplotlib.pyplot as plt

class PiDL:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
    def __init__(self, layers_s, layers_i, lb, ub, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO):

        np.random.seed(1234)
        tf.set_random_seed(1234)

        self.lb = lb
        self.ub = ub
        self.ADO_iterations = ADO_iterations
        self.BFGS_epochs_Pre = BFGS_epochs_Pre
        self.Adam_epochs_ADO = Adam_epochs_ADO
        self.BFGS_epochs_ADO = BFGS_epochs_ADO
        
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

# =============================================================================
#       loss histories
# =============================================================================
        # L-BFGS-S loss histories for pretraining
        self.loss_u_history_Pretrain = np.array([0])
        self.loss_f_history_Pretrain = np.array([0])
        self.loss_lambda_history_Pretrain = np.array([0])
        self.loss_u_val_history_Pretrain = np.array([0])
        self.lambda_history_Pretrain = np.zeros((16,1))  
        self.step_Pretrain = 0
        
        # Adam loss histories for ADO
        self.loss_u_history_Adam = np.array([0])
        self.loss_f_history_Adam = np.array([0])
        self.loss_u_val_history_Adam = np.array([0])
        
        # L-BFGS-S loss histories for ADO
        self.loss_u_history_BFGS = np.array([0])
        self.loss_f_history_BFGS = np.array([0])
        self.loss_u_val_history_BFGS = np.array([0])
        self.step_BFGS = 0
        
        # STRidge loss histories for ADO
        self.loss_f_history_STRidge = np.array([0])
        self.loss_lambda_history_STRidge = np.array([0])
        self.tol_history_STRidge = np.array([0])
        self.lambda_history_STRidge = np.zeros((16, 1))
        self.ridge_append_counter_STRidge = np.array([0])
                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights_s, self.biases_s = self.initialize_NN(layers_s) # root/shared NN
        self.weights0, self.biases0 = self.initialize_NN(layers_i) # branch/individual NN 1
        self.weights1, self.biases1 = self.initialize_NN(layers_i) # branch/individual NN 2
        self.weights2, self.biases2 = self.initialize_NN(layers_i) # branch/individual NN 3
        
        # library coefficients
        self.lambda1 = tf.Variable(tf.zeros([16, 1], dtype=tf.float64))
        
        # Specify the list of trainable variables 
        var_list_ADO = self.biases0 + self.weights0 + self.biases1 + self.weights1 + self.biases2 + \
                self.weights2 + self.weights_s + self.biases_s
        var_list_Pretrain = var_list_ADO + [self.lambda1]

# =============================================================================
#       define losses
# =============================================================================
        # data losses
        self.x_tf = tf.placeholder(tf.float64)
        self.t_tf = tf.placeholder(tf.float64)
        self.u0_tf = tf.placeholder(tf.float64)
        self.u1_tf = tf.placeholder(tf.float64)
        self.u2_tf = tf.placeholder(tf.float64)
        self.u0_pred = self.predict_response(self.x_tf, self.t_tf, 0) 
        self.u1_pred = self.predict_response(self.x_tf, self.t_tf, 1)
        self.u2_pred = self.predict_response(self.x_tf, self.t_tf, 2)
        self.loss_u = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred)) + \
                    tf.reduce_mean(tf.square(self.u2_tf - self.u2_pred)) # data loss for training

        self.x_val_tf = tf.placeholder(tf.float64)
        self.t_val_tf = tf.placeholder(tf.float64)
        self.u0_val_tf = tf.placeholder(tf.float64)
        self.u1_val_tf = tf.placeholder(tf.float64)
        self.u2_val_tf = tf.placeholder(tf.float64)                
        self.u0_val_pred = self.predict_response(self.x_val_tf, self.t_val_tf, 0)
        self.u1_val_pred = self.predict_response(self.x_val_tf, self.t_val_tf, 1)
        self.u2_val_pred = self.predict_response(self.x_val_tf, self.t_val_tf, 2) 
        self.loss_u_val = tf.reduce_mean(tf.square(self.u0_val_tf - self.u0_val_pred)) + \
                tf.reduce_mean(tf.square(self.u1_val_tf - self.u1_val_pred)) + \
                    tf.reduce_mean(tf.square(self.u2_val_tf - self.u2_val_pred)) # data loss for validation

        # physics loss
        self.x_f_tf = tf.placeholder(tf.float64)
        self.t_f_tf = tf.placeholder(tf.float64)        
        self.f_pred, self.Phi_pred, self.u_t_pred = self.physics_residue(self.x_f_tf, self.t_f_tf)        
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))
        
        # L1 regularization for library coefficients
        self.loss_lambda = tf.norm(self.lambda1, ord = 1)       
        
        # total loss
        self.loss = tf.log(self.loss_u  + 0.1*self.loss_f + 1e-4*self.loss_lambda) 
                            
# =============================================================================
#       define optimizers
# =============================================================================
        # optimizer for pretraining
        self.optimizer_BFGS_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                        var_list = var_list_Pretrain,
                                                        method = 'L-BFGS-B', 
                                                       options = {'maxiter': self.BFGS_epochs_Pre,
                                                                   'maxfun': self.BFGS_epochs_Pre,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : np.finfo(float).eps})
        # optimizer for ADO
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.5,
                                                    staircase=True)
        self.optimizer_Adam_ADO = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
        self.train_op_Adam_ADO = self.optimizer_Adam_ADO.minimize(self.loss, var_list = var_list_ADO,
                                                           global_step = self.global_step)
            

        self.optimizer_BFGS_ADO = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                var_list = var_list_ADO,
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': self.BFGS_epochs_ADO,
                                                                           'maxfun': self.BFGS_epochs_ADO,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_normal(size=[layers[l], layers[l+1]]) # initialization when using tanh as activation function
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64))
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_normal(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64))

    def predict_response(self, x, t, IBC_flag):  
        u_int = self.FCNet(tf.concat([x,t],1), self.weights_s, self.biases_s, True) # use root NN
        if IBC_flag == 0:
            u = self.FCNet(u_int, self.weights0, self.biases0, False) # use branch NN 1
        elif IBC_flag == 1:
            u = self.FCNet(u_int, self.weights1, self.biases1, False) # use branch NN 2
        elif IBC_flag == 2:
            u = self.FCNet(u_int, self.weights2, self.biases2, False) # use branch NN 3
        return u

    def FCNet(self, X, weights, biases, si_flag):
        num_layers = len(weights) + 1    
        if si_flag:
            H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # input to root NN
        else:
            H = X # input to branch NN
        for l in range(num_layers-2):
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
        
                
    def physics_residue(self, x, t):
        u0 = self.predict_response(x, t, 0)
        u1 = self.predict_response(x, t, 1)
        u2 = self.predict_response(x, t, 2)
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
        Phi = tf.concat([tf.ones_like(u, optimize = False), u, u**2, u**3, u_x, u*u_x, u**2*u_x,
                              u**3*u_x, u_xx, u*u_xx, u**2*u_xx, u**3*u_xx, u_xxx, u*u_xxx, u**2*u_xxx, u**3*u_xxx], 1)            
        self.library_description = ['1',
                         'u', 'u**2', 'u**3',
                         'u_x', 'u*u_x', 'u**2*u_x', 'u**3*u_x',
                         'u_xx', 'u*u_xx', 'u**2*u_xx', 'u**3*u_xx',
                         'u_xxx', 'u*u_xxx', 'u**2*u_xxx', 'u**3*u_xxx']
        
        f = tf.matmul(Phi, self.lambda1) - u_t      
        return f, Phi, u_t

    def train(self, X, u, X_f, X_val, u_val):
        # training measurements
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.u0 = u[:, 0:1]
        self.u1 = u[:, 1:2]
        self.u2 = u[:, 2:3]
        
        # validation measurements
        self.x_val = X_val[:,0:1]
        self.t_val = X_val[:,1:2]
        self.u0_val = u_val[:, 0:1]
        self.u1_val = u_val[:, 1:2]
        self.u2_val = u_val[:, 2:3]
        
        # physics collocation points
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        
        self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t,
                        self.u0_tf: self.u0, self.u1_tf: self.u1, self.u2_tf: self.u2, 
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.t_val_tf: self.t_val,
                       self.u0_val_tf: self.u0_val, self.u1_val_tf: self.u1_val, self.u2_val_tf: self.u2_val}    
        
        print('L-BFGS-B pretraining begins')
        self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                feed_dict = self.tf_dict,
                                fetches = [self.loss_u, self.loss_f, self.loss_lambda,
                                           self.loss_u_val, self.lambda1],
                                loss_callback = self.callback_Pretrain)
        
        self.tol_best_ADO = 0
        
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            
            print('STRidge begins')
            self.callTrainSTRidge()

            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_f, loss_u_val = self.sess.run([self.loss_u, self.loss_f, self.loss_u_val], self.tf_dict)                   
                    self.loss_u_history_Adam = np.append(self.loss_u_history_Adam, loss_u)
                    self.loss_f_history_Adam = np.append(self.loss_f_history_Adam, loss_f)
                    self.loss_u_val_history_Adam = np.append(self.loss_u_val_history_Adam, loss_u_val)
                    print("Adam epoch(ADO) %s : loss_u = %10.3e ,loss_f = %10.3e , loss_u_val = %10.3e" % (it_Adam, loss_u, loss_f, loss_u_val))
            print('L-BFGS-B begins')
            self.optimizer_BFGS_ADO.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_f, self.loss_u_val],
                                    loss_callback = self.callback_ADO)
        
    def callback_Pretrain(self, loss_u, loss_f, loss_lambda, loss_u_val, lambda1):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:
                        
            self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
            self.loss_f_history_Pretrain = np.append(self.loss_f_history_Pretrain, loss_f)
            self.loss_lambda_history_Pretrain = np.append(self.loss_lambda_history_Pretrain, loss_lambda)            
            self.loss_u_val_history_Pretrain = np.append(self.loss_u_val_history_Pretrain, loss_u_val)
            self.lambda_history_Pretrain = np.append(self.lambda_history_Pretrain, lambda1, axis = 1)
            print("BFGS epoch(Pretrain) %s : loss_u = %10.3e ,loss_f = %10.3e , loss_u_val = %10.3e, loss_lambda = %10.3e" % (self.step_Pretrain, loss_u, loss_f, loss_u_val, loss_lambda))

            
    def callback_ADO(self, loss_u, loss_f, loss_u_val):
        self.step_BFGS = self.step_BFGS + 1
        if self.step_BFGS%10 == 0:                        
            self.loss_u_history_BFGS = np.append(self.loss_u_history_BFGS, loss_u)
            self.loss_f_history_BFGS = np.append(self.loss_f_history_BFGS, loss_f)            
            self.loss_u_val_history_BFGS = np.append(self.loss_u_val_history_BFGS, loss_u_val)    
            print("BFGS epoch(ADO) %s : loss_u = %10.3e ,loss_f = %10.3e , loss_u_val = %10.3e" % (self.step_BFGS, loss_u, loss_f, loss_u_val))

    def callTrainSTRidge(self):
        d_tol = 1
        maxit = 100
        l0_penalty = 10
        Phi_pred, u_t_pred = self.sess.run([self.Phi_pred, self.u_t_pred], self.tf_dict)        
        lambda2 = self.TrainSTRidge(Phi_pred, u_t_pred, d_tol, maxit, l0_penalty = l0_penalty) 
        self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype = tf.float64))

    def TrainSTRidge(self, Phi, ut, d_tol, maxit, STR_iters = 10, l0_penalty = None):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================           
        # First normalize data 
        n,d = Phi.shape
        Phi_normalized = np.zeros((n,d), dtype=np.float64)
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(Phi[:,i],2))
            Phi_normalized[:,i] = Mreg[i]*Phi[:,i]
            
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = self.tol_best_ADO + d_tol
        if l0_penalty == None: 
            l0_penalty = 5e-5*np.linalg.cond(Phi_normalized)
        
        # inherit Lambda
        lambda_best_normalized = self.sess.run(self.lambda1)/Mreg
        
        # record initial sparsity and regression accuracy, and set them as the best
        err_f = np.linalg.norm(ut - Phi_normalized.dot(lambda_best_normalized), 2)
        err_lambda = l0_penalty*np.count_nonzero(lambda_best_normalized)
        err_best = err_f + err_lambda
        tol_best = self.tol_best_ADO                
        self.loss_f_history_STRidge = np.append(self.loss_f_history_STRidge, err_f)
        self.loss_lambda_history_STRidge = np.append(self.loss_lambda_history_STRidge, err_lambda)
        self.tol_history_STRidge = np.append(self.tol_history_STRidge, tol_best)
    
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
            # Get a set of coefficients and error
            lambda1_normalized = self.STRidge(Phi_normalized, ut, STR_iters, tol, Mreg)
            err_f = np.linalg.norm(ut - Phi_normalized.dot(lambda1_normalized), 2)
            err_lambda = l0_penalty*np.count_nonzero(lambda1_normalized)
            err = err_f + err_lambda
    
            if err <= err_best:
                # update the optimal setting if the total error decreases
                err_best = err
                lambda_best_normalized = lambda1_normalized
                tol_best = tol
                tol = tol + d_tol
                
                self.loss_f_history_STRidge = np.append(self.loss_f_history_STRidge, err_f)
                self.loss_lambda_history_STRidge = np.append(self.loss_lambda_history_STRidge, err_lambda)
                self.tol_history_STRidge = np.append(self.tol_history_STRidge, tol_best)
    
            else:
                # otherwise decrease tol and try again
                tol = max([0,tol - 2*d_tol])
                d_tol = d_tol/1.618
                tol = tol + d_tol
        self.tol_best_ADO = tol_best
        return np.real(lambda_best_normalized*Mreg)     

    def STRidge(self, Phi_normalized, ut, STR_iters, tol, Mreg):     
        n,d = Phi_normalized.shape
        # Inherit lambda from previous training and normalize it.
        lambda1_normalized = self.sess.run(self.lambda1)/Mreg            
        
        # find big coefficients
        biginds = np.where(abs(lambda1_normalized) > tol)[0]
        num_relevant = d            
        
        # record lambda evolution
        ridge_append_counter = 0
        self.lambda_history_STRidge = np.append(self.lambda_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
        ridge_append_counter += 1

        # Threshold small coefficients until convergence
        for j in range(STR_iters):  
            # Figure out which items to cut out
            smallinds = np.where(abs(lambda1_normalized) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]
                
            # If nothing changes then stop
            if num_relevant == len(new_biginds): 
                break
            else: 
                num_relevant = len(new_biginds)
                
            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0: 
                    # record lambda evolution
                    self.lambda_history_STRidge = np.append(self.lambda_history_STRidge, np.multiply(Mreg, lambda1_normalized), axis = 1)
                    ridge_append_counter += 1
                    self.ridge_append_counter_STRidge = np.append(self.ridge_append_counter_STRidge, ridge_append_counter)
                    
                    return lambda1_normalized
                else: 
                    break
            biginds = new_biginds
            
            # Otherwise get a new guess
            lambda1_normalized[smallinds] = 0            
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds].T.dot(Phi_normalized[:, biginds]) + 1e-5*np.eye(len(biginds)),Phi_normalized[:, biginds].T.dot(ut))[0]
            
            # record lambda evolution
            self.lambda_history_STRidge = np.append(self.lambda_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
            ridge_append_counter += 1
            
        # Now that we have the sparsity pattern, use standard least squares to get lambda1_normalized
        if biginds != []: 
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds],ut)[0]
        
        # record lambda evolution
        self.lambda_history_STRidge = np.append(self.lambda_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
        ridge_append_counter += 1
        self.ridge_append_counter_STRidge = np.append(self.ridge_append_counter_STRidge, ridge_append_counter)

        return lambda1_normalized
    
    def visualize_training(self):
        # plot loss histories in pretraining        
        fig = plt.figure()
        plt.plot(self.loss_u_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history(Pretraining)')  
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(self.loss_f_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(Pretraining)')     
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda history(Pretraining)')  
        plt.savefig('3.png')
                
        fig = plt.figure()
        plt.plot(self.loss_u_val_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u_val history(Pretraining)')  
        plt.savefig('4.png')
        
        # plot loss histories in ADO               
        fig = plt.figure()
        plt.plot(self.loss_u_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of Adam(ADO)')  
        plt.savefig('5.png')

        fig = plt.figure()
        plt.plot(self.loss_f_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of Adam(ADO)')  
        plt.savefig('6.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_u_val_history_Adam[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u_val history of Adam(ADO)')  
        plt.savefig('7.png')
                                    
        fig = plt.figure()
        plt.plot(self.loss_u_history_BFGS[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of BFGS(ADO)')  
        plt.savefig('8.png')
        
        fig = plt.figure()
        plt.plot(self.loss_f_history_BFGS[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of BFGS(ADO)')     
        plt.savefig('9.png')
                
        fig = plt.figure()
        plt.plot(self.loss_u_val_history_BFGS[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u_val history of BFGS(ADO)')  
        plt.savefig('10.png')
                        
        fig = plt.figure()
        plt.plot(self.loss_f_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f history of STRidge')  
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda history of STRidge')
        plt.savefig('12.png')
        
        fig = plt.figure()
        plt.plot(self.tol_history_STRidge[1:])
        plt.title('Tolerance History of STRidge')
        plt.savefig('13.png')
                        
    def inference(self, X_star):        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}            
        u0 = self.sess.run(self.u0_pred, tf_dict)
        u1 = self.sess.run(self.u1_pred, tf_dict)
        u2 = self.sess.run(self.u2_pred, tf_dict)
        return u0, u1, u2
        
                    
        
