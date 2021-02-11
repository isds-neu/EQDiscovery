# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the equation discovery of cell migration
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
    def __init__(self, layers, lb, ub, pre_ADO_iterations, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO):

        np.random.seed(1234)
        tf.set_random_seed(1234)

        self.lb = lb
        self.ub = ub
        self.pre_ADO_iterations = pre_ADO_iterations
        self.Adam_epochs_Pre = Adam_epochs_Pre
        self.BFGS_epochs_Pre = BFGS_epochs_Pre
        self.ADO_iterations = ADO_iterations
        self.Adam_epochs_ADO = Adam_epochs_ADO
        
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

# =============================================================================
#       loss histories
# =============================================================================
        # loss histories for pretraining
        self.loss_u_history_Pretrain = np.array([0])
        self.loss_f_u_history_Pretrain = np.array([0])
        self.loss_lambda_u_history_Pretrain = np.array([0])
        self.lambda_u_history_Pretrain = np.zeros([8, 1])
        self.step_Pretrain = 0
        self.loss_bc_all_history_Pretrain = np.array([0])
        self.diff_coeff_u_history_Pretrain = np.array([0])
        
        # loss histories for ADO
        self.loss_u_history = np.array([0])
        self.loss_f_u_history = np.array([0])
        self.loss_bc_all_history = np.array([0])
        self.step_ADO = 0
        
        # STRidge loss histories for ADO
        self.loss_f_u_history_STRidge = np.array([0])
        self.loss_lambda_u_history_STRidge = np.array([0])
        self.tol_u_history_STRidge = np.array([0])                            
        self.lambda_u_history_STRidge = np.zeros([9, 1])
        self.ridge_u_append_counter_STRidge = np.array([0])
                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # library coefficients
        self.lambda_u_core = tf.Variable(tf.random_uniform([8, 1], minval = -1, maxval = 1, dtype=tf.float64))
        self.diff_coeff_u_core = tf.Variable(tf.random_uniform([], minval = -1, maxval = 1, dtype=tf.float64)) 
                    
        self.lambda_u_scale = 50
        self.lambda_u = self.lambda_u_scale*tf.tanh(self.lambda_u_core)
        self.diff_u_scale = 1000
        self.diff_coeff_u = self.diff_u_scale*tf.tanh(self.diff_coeff_u_core)
        
        # Specify the list of trainable variables 
        var_list_ADO = self.weights + self.biases
        var_list_Pretrain = var_list_ADO + [self.lambda_u_core] + [self.diff_coeff_u_core] 

# =============================================================================
#       define losses
# =============================================================================
        # data losses
        self.X_tf = tf.placeholder(tf.float64)
        self.U_tf = tf.placeholder(tf.float64)
        self.U_pred = self.predict_response(self.X_tf)
        self.loss_u = tf.reduce_mean(tf.square(self.U_tf - self.U_pred))
                
        self.loss_u_coeff = tf.placeholder(tf.float64)

        self.loss_U = self.loss_u_coeff*self.loss_u
            
        # physics loss
        self.x_f_tf = tf.placeholder(tf.float64)
        self.t_f_tf = tf.placeholder(tf.float64)        
        self.f_u_pred, self.Phi_pred, self.u_t, self.u_xx_pred = self.physics_residue(self.x_f_tf, self.t_f_tf)        
        self.loss_f_u = tf.reduce_mean(tf.square(self.f_u_pred))
        
        self.loss_f = self.loss_f_u

        # Neumann boundary loss
        self.x_l_tf = tf.placeholder(tf.float64)
        self.x_r_tf = tf.placeholder(tf.float64)        
        self.t_l_tf = tf.placeholder(tf.float64)
        self.t_r_tf = tf.placeholder(tf.float64)

        self.U_l = self.predict_response(tf.concat((self.x_l_tf, self.t_l_tf), 1))
        self.U_r = self.predict_response(tf.concat((self.x_r_tf, self.t_r_tf), 1))

        self.U_l_x = tf.gradients(self.U_l, self.x_l_tf)[0]
        self.U_r_x = tf.gradients(self.U_r, self.x_r_tf)[0]
        self.loss_bc_all = tf.reduce_mean(tf.square(self.U_l_x)) + tf.reduce_mean(tf.square(self.U_r_x))
        
        self.loss_bc_coeff = tf.placeholder(tf.float64)      
        
        # L1 regularization for library coefficients
        self.loss_lambda_u = tf.norm(self.lambda_u, ord=1)    
        
        # pretraining loss
        self.loss = tf.log(self.loss_U + self.loss_f + 0*self.loss_lambda_u + self.loss_bc_coeff*self.loss_bc_all)
        
        # ADO loss
        self.loss_ADO = tf.log(self.loss_U + 2e3*self.loss_f + self.loss_bc_coeff*self.loss_bc_all)    
                        
        # post-training loss
        self.nonzero_mask_lambda_u_tf = tf.placeholder(tf.float64)
        self.f_u_pred_pt, self.u_xx_pt = self.physics_residue_pt(self.x_f_tf, self.t_f_tf)
        self.loss_f_u_pt = tf.reduce_mean(tf.square(self.f_u_pred_pt))
        self.loss_pt = tf.log(self.loss_U + 2e3*self.loss_f_u_pt + self.loss_bc_coeff*self.loss_bc_all)
        
# =============================================================================
#       define optimizers
# =============================================================================
        # optimizers for pretraining
        self.global_step_Pre = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate_Pre = tf.train.exponential_decay(starter_learning_rate, self.global_step_Pre,
                                                            2000, 0.1, staircase=True)
        self.optimizer_Adam_Pre = tf.train.AdamOptimizer(learning_rate = self.learning_rate_Pre)
        self.train_op_Adam_Pretrain = self.optimizer_Adam_Pre.minimize(self.loss, var_list = var_list_Pretrain,
                                                                  global_step = self.global_step_Pre)
        
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
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 2000, 0.5,
                                                     staircase=True)

        # The default settings: learning rate = 1e-3, beta1 = 0.9, beta2 = 0.999， epsilon = 1e-8
        self.optimizer_Adam_ADO = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
        self.train_op_Adam_ADO = self.optimizer_Adam_ADO.minimize(self.loss_ADO, var_list = var_list_ADO, 
                                                          global_step = self.global_step)

        # optimizer for post-training
        self.optimizer_Adam_pt = tf.train.AdamOptimizer(learning_rate = 1e-3) 
        self.train_op_Adam_pt = self.optimizer_Adam_pt.minimize(self.loss_pt, var_list = var_list_ADO)
            
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def coeff_activation(self, x, a):
        return a*tf.sigmoid(x)
        
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

    def predict_response(self, X):  
        U = self.FCNet(X, self.weights, self.biases)
        return U

    def FCNet(self, X, weights, biases):
        num_layers = len(weights) + 1    
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        W = weights[0]
        b = biases[0]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))

        for l in range(1, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = weights[-1]
        b = biases[-1]            
        Y = tf.log(tf.exp(tf.add(tf.matmul(H, W), b)) + 1) # softplus
        return Y
                
    def physics_residue(self, x, t):
        u = self.predict_response(tf.concat((x, t), 1))

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        Phi = tf.concat((tf.ones_like(x, optimize = False), u, u**2, u**3, u_x, u_x*u, u_x*u**2, u_x*u**3), 1)      
        self.lib_descr = ['1', 'u', 'u**2', 'u**3', 'u_x', 'u_x*u', 'u_x*u**2', 'u_x*u**3']
        f_u = tf.matmul(Phi, self.lambda_u) + self.diff_coeff_u*u_xx - u_t      
                    
        return f_u, Phi, u_t, u_xx

    def physics_residue_pt(self, x, t):
        # for post-training
        u = self.predict_response(tf.concat((x, t), 1))

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        Phi = tf.concat((tf.ones_like(x, optimize = False), u, u**2, u**3, u_x, u_x*u, u_x*u**2, u_x*u**3), 1)      
        self.lib_descr = ['1', 'u', 'u**2', 'u**3', 'u_x', 'u_x*u', 'u_x*u**2', 'u_x*u**3']
        f_u = tf.matmul(Phi, self.lambda_u*self.nonzero_mask_lambda_u_tf) + self.diff_coeff_u*u_xx - u_t 
        
        return f_u, u_xx
                
    def train(self, X, U, X_f, X_l, X_r):        
        self.tf_dict = {self.X_tf: X,  self.U_tf: U, 
                            self.x_f_tf: X_f[:, 0:1], self.t_f_tf: X_f[:, 1:2],
                            self.x_l_tf: X_l[:, 0:1], self.t_l_tf: X_l[:, 1:2],
                            self.x_r_tf: X_r[:, 0:1], self.t_r_tf: X_r[:, 1:2]}    
        
        # adaptively determine loss coefficients
        self.tf_dict[self.loss_u_coeff] = 1
        self.tf_dict[self.loss_bc_coeff] = 1
        self.anneal_lam = [1, 1]
        self.anneal_alpha = 0.8
            
        print('Pre ADO')
        for it_ADO_Pre in range(self.pre_ADO_iterations): 
            print('Adam pretraining begins')
            for it_Adam in range(self.Adam_epochs_Pre):
                self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_f_u, loss_lambda_u, lambda_u, diff_coeff_u, loss_bc_all = self.sess.run([self.loss_u, self.loss_f_u, self.loss_lambda_u, self.lambda_u, self.diff_coeff_u, self.loss_bc_all], self.tf_dict)                   
                    self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
                    self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
                    self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
                    self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
                    self.diff_coeff_u_history_Pretrain = np.append(self.diff_coeff_u_history_Pretrain, diff_coeff_u)
                    self.loss_bc_all_history_Pretrain = np.append(self.loss_bc_all_history_Pretrain, loss_bc_all)
                    print("Adam epoch(Pretrain) %s : loss_u = %10.3e, loss_f_u = %10.3e, loss_lambda_u = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_f_u, loss_lambda_u, loss_bc_all))
            
            print('L-BFGS-B pretraining begins')
            self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_f_u, self.loss_lambda_u, self.lambda_u, self.diff_coeff_u, self.loss_bc_all],
                                    loss_callback = self.callback_Pretrain)
            
            # adaptively determine loss coefficients
            loss_u, loss_f_u, loss_bc_all = self.sess.run([self.loss_u, self.loss_f_u, self.loss_bc_all], self.tf_dict)
            self.anneal_lam[0] = (1 - self.anneal_alpha)*self.anneal_lam[0] + self.anneal_alpha*loss_u/loss_f_u
            self.tf_dict[self.loss_u_coeff] = self.anneal_lam[0]
            
            self.anneal_lam[1] = (1 - self.anneal_alpha)*self.anneal_lam[1] + self.anneal_alpha*loss_bc_all/loss_f_u
            self.tf_dict[self.loss_bc_coeff] = self.anneal_lam[1]
                
        self.tol_best_ADO_u = 0
        
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            print('STRidge begins')
            self.callTrainSTRidge()

            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_f_u, loss_bc_all = self.sess.run([self.loss_u, self.loss_f_u, self.loss_bc_all], self.tf_dict)                   
                    self.loss_u_history = np.append(self.loss_u_history, loss_u)
                    self.loss_f_u_history = np.append(self.loss_f_u_history, loss_f_u)
                    self.loss_bc_all_history = np.append(self.loss_bc_all_history, loss_bc_all)
                    print("Adam epoch(ADO) %s : loss_u = %10.3e, loss_f_u = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_f_u, loss_bc_all))                
        
    def callback_Pretrain(self, loss_u, loss_f_u, loss_lambda_u, lambda_u, diff_coeff_u, loss_bc_all):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:                        
            self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
            self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
            self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
            self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
            self.diff_coeff_u_history_Pretrain = np.append(self.diff_coeff_u_history_Pretrain, diff_coeff_u)
            self.loss_bc_all_history_Pretrain = np.append(self.loss_bc_all_history_Pretrain, loss_bc_all)
            print("BFGS epoch(Pretrain) %s : loss_u = %10.3e, loss_f_u = %10.3e, loss_lambda_u = %10.3e, loss_bc_all = %10.3e" % (self.step_Pretrain, loss_u, loss_f_u, loss_lambda_u, loss_bc_all))
            
    def callTrainSTRidge(self):
        d_tol = 1e-3
        maxit = 50
        l0_penalty = 1e-10
        
        Phi_pred, u_t, u_xx = self.sess.run([self.Phi_pred, self.u_t, self.u_xx_pred], self.tf_dict)
        Phi_aug = np.concatenate([u_xx, Phi_pred], 1) # augment Phi_pred w/ diffusion candidates
        
        # lambda_u
        lambda_u_aug = self.TrainSTRidge(Phi_aug, u_t, d_tol, maxit, l0_penalty = l0_penalty) 
        lambda_u_core_new = np.arctanh(lambda_u_aug[1:, :]/self.lambda_u_scale)
        self.lambda_u_core = tf.assign(self.lambda_u_core, tf.convert_to_tensor(lambda_u_core_new, dtype = tf.float64))
        diff_coeff_u_core_new = np.arctanh(lambda_u_aug[0,0]/self.diff_u_scale)
        self.diff_coeff_u_core = tf.assign(self.diff_coeff_u_core, tf.convert_to_tensor(diff_coeff_u_core_new, dtype = tf.float64))

    def TrainSTRidge(self, Phi, ut, d_tol, maxit, STR_iters = 10, l0_penalty = None):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================      
            
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        
        tol = d_tol + self.tol_best_ADO_u
        tol_best = self.tol_best_ADO_u                
            
        if l0_penalty == None: 
            l0_penalty = 1e-3*np.linalg.cond(Phi)
                    
        # inherit augmented Lambda
        diff_u = self.sess.run(self.diff_coeff_u)
        diff_u = np.reshape(diff_u, (1, 1))
        lambda_u = self.sess.run(self.lambda_u)
        lambda_best = np.concatenate([diff_u, lambda_u], axis = 0)
        
        # record initial sparsity and regression accuracy, and set them as the best
        err_f = np.mean((ut - Phi.dot(lambda_best))**2)
        err_lambda = l0_penalty*np.count_nonzero(lambda_best)
        err_best = err_f + err_lambda
        self.loss_f_u_history_STRidge = np.append(self.loss_f_u_history_STRidge, err_f)
        self.loss_lambda_u_history_STRidge = np.append(self.loss_lambda_u_history_STRidge, err_lambda)
        self.tol_u_history_STRidge = np.append(self.tol_u_history_STRidge, tol_best)
    
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
            # Get a set of coefficients and error
            lambda1 = self.STRidge(Phi, ut, STR_iters, tol)
            err_f = np.mean((ut - Phi.dot(lambda_best))**2)
            err_lambda = l0_penalty*np.count_nonzero(lambda1)
            err = err_f + err_lambda
    
            if err <= err_best:
                # update the optimal setting if the total error decreases
                err_best = err
                lambda_best = lambda1
                tol_best = tol
                tol = tol + d_tol
                
                self.loss_f_u_history_STRidge = np.append(self.loss_f_u_history_STRidge, err_f)
                self.loss_lambda_u_history_STRidge = np.append(self.loss_lambda_u_history_STRidge, err_lambda)
                self.tol_u_history_STRidge = np.append(self.tol_u_history_STRidge, tol_best)
    
            else:
                # otherwise decrease tol and try again
                tol = max([0,tol - 2*d_tol])
                d_tol = 2*d_tol / (maxit - iter)
                tol = tol + d_tol
        self.tol_best_ADO_u = tol_best
        
        return np.real(lambda_best)  

    def STRidge(self, Phi, ut, STR_iters, tol):  
        # First normalize data
        n,d = Phi.shape        
        Phi_normalized = np.zeros((n,d), dtype=np.complex64)
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(Phi[:,i],2))
            Phi_normalized[:,i] = Mreg[i]*Phi[:,i]            
                
        # Inherit augmented lambda from previous training and normalize it.
        diff_u = self.sess.run(self.diff_coeff_u)
        diff_u = np.reshape(diff_u, (1, 1))

        lambda_u = self.sess.run(self.lambda_u)
        lambda1_normalized = np.concatenate([diff_u, lambda_u], axis = 0)/Mreg
                    
        # find big coefficients
        biginds = np.where(abs(lambda1_normalized[1:]) > tol)[0] + 1 # keep diff_u term unpruned
        biginds = np.insert(biginds, obj = 0, values = 0) # keep diffu u term unpruned
            
        num_relevant = d            
        
        # record lambda evolution
        ridge_append_counter = 0
        ridge_append_counter = self.record_lambda_in_STRidge(Mreg, lambda1_normalized, ridge_append_counter, end_flag = False)

        # Threshold small coefficients until convergence
        for j in range(STR_iters):  
            # Figure out which items to cut out
            smallinds = np.where(abs(lambda1_normalized[1:]) < tol)[0] + 1 # don't threhold diffu terms 
                
            new_biginds = [i for i in range(d) if i not in smallinds]
                
            # If nothing changes then stop
            if num_relevant == len(new_biginds): 
                break
            else: 
                num_relevant = len(new_biginds)
                
            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0: 
                    ridge_append_counter = self.record_lambda_in_STRidge(Mreg, lambda1_normalized, ridge_append_counter, end_flag = True)                    
                    return lambda1_normalized*Mreg
                else: 
                    break
            biginds = new_biginds
            
            # Otherwise get a new guess
            lambda1_normalized[smallinds] = 0            
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds], ut)[0]
            
            # record lambda evolution
            ridge_append_counter = self.record_lambda_in_STRidge(Mreg, lambda1_normalized, ridge_append_counter, end_flag = False)
            
        # Now that we have the sparsity pattern, use standard least squares to get lambda1_normalized
        if biginds != []: 
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds],ut)[0]
        
        # record lambda evolution
        ridge_append_counter = self.record_lambda_in_STRidge(Mreg, lambda1_normalized, ridge_append_counter, end_flag = True)
        return lambda1_normalized*Mreg
    
    def record_lambda_in_STRidge(self, Mreg, lambda1_normalized, ridge_append_counter, end_flag):
        ridge_append_counter += 1
        self.lambda_u_history_STRidge = np.append(self.lambda_u_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
        if end_flag:
            self.ridge_u_append_counter_STRidge = np.append(self.ridge_u_append_counter_STRidge, ridge_append_counter)
        return ridge_append_counter
        
    def visualize_training(self):
# =============================================================================
#         plot loss histories in pretraining        
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_u_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history(Pretraining)')  
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(self.loss_f_u_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(Pretraining)')     
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_u_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_lambda history(Pretraining)')  
        plt.savefig('3.png')
                
        fig = plt.figure()
        plt.plot(self.loss_bc_all_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_bc_all_history_Pretrain')  
        plt.savefig('4.png')    

        fig = plt.figure()
        plt.plot(self.diff_coeff_u_history_Pretrain[1:])
        plt.xlabel('10x')
        plt.title('diff_coeff_u_history_Pretrain')  
        plt.savefig('5.png')        
        
# =============================================================================
#         plot loss histories in ADO               
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_u_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history(ADO)')  
        plt.savefig('6.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(ADO)')  
        plt.savefig('7.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_bc_all_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_bc_all_history of ADO')  
        plt.savefig('8.png')
                                    
# =============================================================================
#            plot loss histories in STRidge                   
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_f_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f_u history of STRidge')  
        plt.savefig('9.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda_u history of STRidge')
        plt.savefig('10.png')
        
        fig = plt.figure()
        plt.plot(self.tol_u_history_STRidge[1:])
        plt.title('Tolerance_u History of STRidge')
        plt.savefig('11.png')
                        
    def visualize_post_training(self):
        fig = plt.figure()
        plt.plot(self.loss_u_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history(post-training)')  
        plt.savefig('12.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(post-training)')  
        plt.savefig('13.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_bc_all_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_bc_all_history_pt')  
        plt.savefig('14.png')  
        
    def inference(self, X_star):        
        tf_dict = {self.X_tf: X_star}            
        U = self.sess.run(self.U_pred, tf_dict)
        return U
    
    def post_train(self, pt_ADO_iterations, Adam_epochs_Pt):
                
        # loss histories for post-training
        self.loss_u_history_pt = np.array([0])
        self.loss_f_u_history_pt = np.array([0])
        self.lambda_u_history_pt = np.zeros([9, 1])
        self.loss_bc_all_history_pt = np.array([0])
        
        print('post-training begins')
        for it in tqdm(range(pt_ADO_iterations)):
# =============================================================================
#              update library coefficients lambda_u and lambda_v via least squares
# =============================================================================
            print('least squares begins')
            # find non-zero values in library coefficients
            lambda_u = self.sess.run(self.lambda_u)
            diff_u = self.sess.run(self.diff_coeff_u)
            diff_u = np.reshape(diff_u, (1, 1))
            lambda_u_aug = np.concatenate([diff_u, lambda_u], axis = 0)
            nonzero_ind_u_aug = np.nonzero(lambda_u_aug)[0]            

            # form compact libraries Phi_u_compact that only have non-zero candidates
            Phi_pred, u_t, u_xx = self.sess.run([self.Phi_pred, self.u_t, self.u_xx_pred], self.tf_dict)
            Phi_aug = np.concatenate([u_xx, Phi_pred], 1) # augment Phi_pred w/ diffusion candidates            
            Phi_u_compact = Phi_aug[:, nonzero_ind_u_aug] 

            # normalize Phi_u_compact 
            Phi_u_compact_norm = np.zeros_like(Phi_u_compact)
            Mreg_u = np.zeros((Phi_u_compact.shape[1], 1))
            for it_Phi_u_compact in range(Phi_u_compact.shape[1]):
                Mreg_u[it_Phi_u_compact] = 1.0/(np.linalg.norm(Phi_u_compact[:,it_Phi_u_compact], 2))
                Phi_u_compact_norm[:,it_Phi_u_compact] = Mreg_u[it_Phi_u_compact]*Phi_u_compact[:,it_Phi_u_compact]      
                            
            # do least square to update non-zero values in lambda_u
            lambda_u_updated_compact = np.linalg.lstsq(Phi_u_compact_norm, u_t)[0]*Mreg_u
                                    
            # assign updated values to self.lambda_u and self.diff_coeff_u_core
            lambda_u_aug_updated = np.zeros_like(lambda_u_aug)
            lambda_u_aug_updated[nonzero_ind_u_aug] = lambda_u_updated_compact            
            lambda_u_core_new = np.arctanh(lambda_u_aug_updated[1:, :]/self.lambda_u_scale)
            self.lambda_u_core = tf.assign(self.lambda_u_core, tf.convert_to_tensor(lambda_u_core_new, dtype = tf.float64))
            
            diff_coeff_u_core_new = np.arctanh(lambda_u_aug_updated[0, 0]/self.diff_u_scale)
            self.diff_coeff_u_core = tf.assign(self.diff_coeff_u_core, tf.convert_to_tensor(diff_coeff_u_core_new, dtype = tf.float64))

            self.lambda_u_history_pt = np.append(self.lambda_u_history_pt, lambda_u_aug_updated, axis = 1)
# =============================================================================
#              update NN weights and bias via Adam
# =============================================================================
            # mark non-zero candidates in the library 
            nonzero_mask_lambda_u = np.zeros_like(lambda_u)
            nonzero_ind_u = nonzero_ind_u_aug[1:] - 1
            nonzero_mask_lambda_u[nonzero_ind_u, 0] = 1
            self.tf_dict[self.nonzero_mask_lambda_u_tf] = nonzero_mask_lambda_u
            
            print('Adam begins')
            for it_Adam in range(Adam_epochs_Pt):
                self.sess.run(self.train_op_Adam_pt, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_f_u, loss_bc_all = self.sess.run([self.loss_u, self.loss_f_u_pt, self.loss_bc_all], self.tf_dict)
                    self.loss_u_history_pt = np.append(self.loss_u_history_pt, loss_u)
                    self.loss_f_u_history_pt = np.append(self.loss_f_u_history_pt, loss_f_u)
                    self.loss_bc_all_history_pt = np.append(self.loss_bc_all_history_pt, loss_bc_all)
                    print("Adam epoch(Pt-ADO) %s : loss_u = %10.3e, loss_f_u = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_f_u, loss_bc_all))                

# =============================================================================
#       determine whether the post-training is sufficient
# =============================================================================
        self.visualize_post_training()
