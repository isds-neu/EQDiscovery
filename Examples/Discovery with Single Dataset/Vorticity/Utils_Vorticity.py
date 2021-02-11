# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the discovery of Navier Stokes Vorticity equation with a single dataset 
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
    def __init__(self, layers, lb, ub, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO):

        np.random.seed(1234)
        tf.set_random_seed(1234)

        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.Adam_epochs_Pre = Adam_epochs_Pre
        self.BFGS_epochs_Pre = BFGS_epochs_Pre
        self.ADO_iterations = ADO_iterations
        self.Adam_epochs_ADO = Adam_epochs_ADO
        self.BFGS_epochs_ADO = BFGS_epochs_ADO
        
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

# =============================================================================
#       loss histories
# =============================================================================
        # loss histories for pretraining
        self.loss_u_history_Pretrain = np.array([0])
        self.loss_v_history_Pretrain = np.array([0])
        self.loss_w_history_Pretrain = np.array([0])
        self.loss_f_history_Pretrain = np.array([0])
        self.loss_lambda_history_Pretrain = np.array([0])
        self.lambda_history_Pretrain = np.zeros([60, 1])
        self.step_Pretrain = 0
        self.loss_U_val_history_Pretrain = np.array([0])
        
        # loss histories for ADO
        self.loss_u_history = np.array([0])
        self.loss_v_history = np.array([0])
        self.loss_w_history = np.array([0])
        self.loss_f_history = np.array([0])
        self.loss_U_val_history = np.array([0])
        self.step_ADO = 0
        
        # STRidge loss histories for ADO
        self.loss_f_history_STRidge = np.array([0])
        self.loss_lambda_history_STRidge = np.array([0])
        self.tol_history_STRidge = np.array([0])
                            
        self.lambda_history_STRidge = np.zeros([60, 1])
        self.ridge_append_counter_STRidge = np.array([0])
                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # library coefficients
        self.lambda1 = tf.Variable(tf.zeros([60, 1], dtype=tf.float32)) 
                    
        # Specify the list of trainable variables 
        var_list_Pretrain = self.biases + self.weights + [self.lambda1]
        var_list_ADO = self.biases + self.weights

# =============================================================================
#       define losses
# =============================================================================
        # data losses
        self.x_tf = tf.placeholder(tf.float32)
        self.y_tf = tf.placeholder(tf.float32)
        self.t_tf = tf.placeholder(tf.float32)
        self.u_tf = tf.placeholder(tf.float32)
        self.v_tf = tf.placeholder(tf.float32)
        self.w_tf = tf.placeholder(tf.float32)
        self.u_pred, self.v_pred, self.w_pred = self.predict_response(self.x_tf, self.y_tf, self.t_tf)         
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        self.loss_w = tf.reduce_mean(tf.square(self.w_tf - self.w_pred))
            
        self.x_val_tf = tf.placeholder(tf.float32)
        self.y_val_tf = tf.placeholder(tf.float32)
        self.t_val_tf = tf.placeholder(tf.float32)
        self.u_val_tf = tf.placeholder(tf.float32)
        self.v_val_tf = tf.placeholder(tf.float32) 
        self.w_val_tf = tf.placeholder(tf.float32)              
        self.u_val_pred, self.v_val_pred, self.w_val_pred = self.predict_response(self.x_val_tf, self.y_val_tf, self.t_val_tf) 
        self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
        self.loss_v_val = tf.reduce_mean(tf.square(self.v_val_tf - self.v_val_pred))
        self.loss_w_val = tf.reduce_mean(tf.square(self.w_val_tf - self.w_val_pred))
        self.loss_U_val = self.loss_u_val  + self.loss_v_val + self.loss_w_val

        # physics loss
        self.x_f_tf = tf.placeholder(tf.float32)
        self.y_f_tf = tf.placeholder(tf.float32)
        self.t_f_tf = tf.placeholder(tf.float32)        
        self.f_w_pred, self.Phi, self.w_t_pred = self.physics_residue(self.x_f_tf, self.y_f_tf, self.t_f_tf,)        
        self.loss_f = tf.reduce_mean(tf.square(self.f_w_pred))
        
        # L1 regularization for library coefficients
        self.loss_lambda = 1e-10*tf.norm(self.lambda1, ord=1)
        
        # total loss
        self.loss = tf.log(self.loss_u  + self.loss_v + self.loss_w + self.loss_f + self.loss_lambda) 
                            
# =============================================================================
#       define optimizers
# =============================================================================
        # optimizers for pretraining
        self.optimizer_Adam_Pretrain = tf.train.AdamOptimizer(learning_rate = 1e-3) 
        self.train_op_Adam_Pretrain = self.optimizer_Adam_Pretrain.minimize(self.loss, var_list = var_list_Pretrain)
        
        self.optimizer_BFGS_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                        var_list = var_list_Pretrain,
                                                        method = 'L-BFGS-B', 
                                                       options = {'maxiter': self.BFGS_epochs_Pre,
                                                                   'maxfun': self.BFGS_epochs_Pre,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : np.finfo(float).eps})
        # optimizer for ADO
        self.optimizer_Adam_ADO = tf.train.AdamOptimizer(learning_rate = 1e-4) 
        self.train_op_Adam_ADO = self.optimizer_Adam_ADO.minimize(self.loss, var_list = var_list_ADO)

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
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_normal(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def predict_response(self, x, y, t):  
        Y = self.FCNet(tf.concat([x,y,t],1), self.weights, self.biases)
        u = Y[:, 0:1]
        v = Y[:, 1:2]
        w = Y[:, 2:3]
        return u, v, w

    def FCNet(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
                
    def physics_residue(self, x, y, t):
        u, v, w = self.predict_response(x,y,t)
        data = [u, v, w]            
            
        ## derivatives     
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_xy = tf.gradients(w_x, y)[0]
                               
        derivatives = [tf.ones_like(u, optimize = False)]
        derivatives.append(w_x)
        derivatives.append(w_y)
        derivatives.append(w_xx)
        derivatives.append(w_xy)
        derivatives.append(w_yy)
        
        derivatives_description = ['', 'w_{x}', 'w_{y}', 'w_{xx}', 'w_{xy}', 'w_{yy}']
        
        lib_fun, self.lib_descr = self.build_library(data, derivatives, derivatives_description,
                                                    data_description = ['u','v', 'w'])      
        w_t = tf.gradients(w, t)[0]
        Phi = tf.concat(lib_fun, 1)
        f_w = w_t - Phi@self.lambda1                    
                                    
        return f_w, Phi, w_t
            
    def build_library(self, data, derivatives, derivatives_description, data_description = None):         
        ## 2nd order polynomial terms
        lib_poly = [tf.ones_like(data[0], optimize = False)]
        lib_poly_descr = [''] # it denotes '1'
        for i in range(len(data)): # polynomial terms of univariable
            for j in range(1, 3):
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
        
    def train(self, X, u, v, w, X_f, X_val, u_val, v_val, w_val):
        # training measurements
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]       
        self.u = u
        self.v = v
        self.w = w
        # validation measurements
        self.x_val = X_val[:,0:1]
        self.y_val = X_val[:,1:2]
        self.t_val = X_val[:,2:3]
        self.u_val = u_val
        self.v_val = v_val
        self.w_val = w_val
        
        # physics collocation points
        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]
        self.t_f = X_f[:,2:3]
        
        self.tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                        self.u_tf: self.u, self.v_tf: self.v, self.w_tf: self.w,
                       self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.y_val_tf: self.y_val, self.t_val_tf: self.t_val,
                       self.u_val_tf: self.u_val, self.v_val_tf: self.v_val, self.w_val_tf: self.w_val}    
        
        print('Adam pretraining begins')
        for it_Adam in range(self.Adam_epochs_Pre):
            self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict)
            if it_Adam % 10 == 0:
                loss_u, loss_v, loss_w, loss_f, loss_lambda, loss_U_val, lambda1 = self.sess.run([self.loss_u, self.loss_v, self.loss_w, self.loss_f, self.loss_lambda, self.loss_U_val, self.lambda1], self.tf_dict)                   
                self.loss_U_val_history_Pretrain = np.append(self.loss_U_val_history_Pretrain, loss_U_val)
                self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
                self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
                self.loss_w_history_Pretrain = np.append(self.loss_w_history_Pretrain, loss_w)
                self.loss_f_history_Pretrain = np.append(self.loss_f_history_Pretrain, loss_f)
                self.loss_lambda_history_Pretrain = np.append(self.loss_lambda_history_Pretrain, loss_lambda)
                self.lambda_history_Pretrain = np.append(self.lambda_history_Pretrain, lambda1, axis = 1)
                print("Adam epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_w = %10.3e, loss_f = %10.3e, loss_U_val = %10.3e, loss_lambda = %10.3e" % (it_Adam, loss_u, loss_v, loss_w, loss_f, loss_U_val, loss_lambda))
        
        print('L-BFGS-B pretraining begins')
        self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                feed_dict = self.tf_dict,
                                fetches = [self.loss_u, self.loss_v, self.loss_w, self.loss_f, self.loss_lambda, self.loss_U_val, self.lambda1],
                                loss_callback = self.callback_Pretrain)
        self.tol_best_ADO = 0
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_w, loss_f, loss_U_val = self.sess.run([self.loss_u, self.loss_v, self.loss_w, self.loss_f, self.loss_U_val], self.tf_dict)                   
                    self.loss_U_val_history = np.append(self.loss_U_val_history, loss_U_val)
                    self.loss_u_history = np.append(self.loss_u_history, loss_u)
                    self.loss_v_history = np.append(self.loss_v_history, loss_v)
                    self.loss_w_history = np.append(self.loss_w_history, loss_w)
                    self.loss_f_history = np.append(self.loss_f_history, loss_f)
                    print("Adam epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_w = %10.3e, loss_f = %10.3e, loss_U_val = %10.3e" % (it_Adam, loss_u, loss_v, loss_w, loss_f, loss_U_val))                

            print('L-BFGS-B begins')
            self.optimizer_BFGS_ADO.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_v, self.loss_w, self.loss_f, self.loss_U_val],
                                    loss_callback = self.callback_ADO)

            print('STRidge begins')
            self.callTrainSTRidge()
        
    def callback_Pretrain(self, loss_u, loss_v, loss_w, loss_f, loss_lambda, loss_U_val, lambda1):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:
                        
            self.loss_U_val_history_Pretrain = np.append(self.loss_U_val_history_Pretrain, loss_U_val)
            self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
            self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
            self.loss_w_history_Pretrain = np.append(self.loss_w_history_Pretrain, loss_w)
            self.loss_f_history_Pretrain = np.append(self.loss_f_history_Pretrain, loss_f)
            self.loss_lambda_history_Pretrain = np.append(self.loss_lambda_history_Pretrain, loss_lambda)
            self.lambda_history_Pretrain = np.append(self.lambda_history_Pretrain, lambda1, axis = 1)
            print("BFGS epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_w = %10.3e, loss_f = %10.3e, loss_U_val = %10.3e, loss_lambda = %10.3e" % (self.step_Pretrain, loss_u, loss_v, loss_w, loss_f, loss_U_val, loss_lambda))
            
    def callback_ADO(self, loss_u, loss_v, loss_w, loss_f, loss_U_val):
        self.step_ADO = self.step_ADO + 1
        if self.step_ADO%10 == 0:                        
            self.loss_U_val_history = np.append(self.loss_U_val_history, loss_U_val)
            self.loss_u_history = np.append(self.loss_u_history, loss_u)
            self.loss_v_history = np.append(self.loss_v_history, loss_v)
            self.loss_w_history = np.append(self.loss_w_history, loss_w)
            self.loss_f_history = np.append(self.loss_f_history, loss_f)
            print("BFGS epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_w = %10.3e, loss_f = %10.3e, loss_U_val = %10.3e" % (self.step_ADO, loss_u, loss_v, loss_w, loss_f, loss_U_val))                

    def callTrainSTRidge(self):
        d_tol = 1
        maxit = 25
        Phi, w_t_pred = self.sess.run([self.Phi, self.w_t_pred], self.tf_dict)
        lambda2 = self.TrainSTRidge(Phi, w_t_pred, d_tol, maxit) 
        self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype = tf.float32))

    def TrainSTRidge(self, Phi, ut, d_tol, maxit, STR_iters = 10, l0_penalty = None):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================      
            
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol + self.tol_best_ADO
        if l0_penalty == None: 
            l0_penalty = 1e-3*np.linalg.cond(Phi)
                    
        # inherit Lambda
        lambda_best = self.sess.run(self.lambda1)
        
        # record initial sparsity and regression accuracy, and set them as the best
        err_f = np.linalg.norm(ut - Phi.dot(lambda_best), 2)
        err_lambda = l0_penalty*np.count_nonzero(lambda_best)
        err_best = err_f + err_lambda
        tol_best = self.tol_best_ADO                
        self.loss_f_history_STRidge = np.append(self.loss_f_history_STRidge, err_f)
        self.loss_lambda_history_STRidge = np.append(self.loss_lambda_history_STRidge, err_lambda)
        self.tol_history_STRidge = np.append(self.tol_history_STRidge, tol_best)
    
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
            # Get a set of coefficients and error
            lambda1 = self.STRidge(Phi, ut, STR_iters, tol)
            err_f = np.linalg.norm(ut - Phi.dot(lambda1), 2)
            err_lambda = l0_penalty*np.count_nonzero(lambda1)
            err = err_f + err_lambda
    
            if err <= err_best:
                # update the optimal setting if the total error decreases
                err_best = err
                lambda_best = lambda1
                tol_best = tol
                tol = tol + d_tol
                
                self.loss_f_history_STRidge = np.append(self.loss_f_history_STRidge, err_f)
                self.loss_lambda_history_STRidge = np.append(self.loss_lambda_history_STRidge, err_lambda)
                self.tol_history_STRidge = np.append(self.tol_history_STRidge, tol_best)
    
            else:
                # otherwise decrease tol and try again
                tol = max([0,tol - 2*d_tol])
                d_tol = 2*d_tol / (maxit - iter)
                tol = tol + d_tol
        self.tol_best_ADO = tol_best
        return np.real(lambda_best)  

    def STRidge(self, Phi, ut, STR_iters, tol):    
        # First normalize data
        n,d = Phi.shape        
        Phi_normalized = np.zeros((n,d), dtype=np.complex64)
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(Phi[:,i],2))
            Phi_normalized[:,i] = Mreg[i]*Phi[:,i]            
            
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
                    
                    return lambda1_normalized*Mreg
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

        return lambda1_normalized*Mreg
    
    def visualize_training(self):
# =============================================================================
#         plot loss histories in pretraining        
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_u_history_Pretrain[1:])
        plt.plot(self.loss_v_history_Pretrain[1:])
        plt.plot(self.loss_w_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.title('loss_u, loss_ and loss_w history(Pretraining)')  
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
        plt.plot(self.loss_U_val_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_U_val history(Pretraining)')  
        plt.savefig('4.png')
        
# =============================================================================
#         plot loss histories in ADO               
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_u_history[1:])
        plt.plot(self.loss_v_history[1:])
        plt.plot(self.loss_w_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v', 'loss_w'))
        plt.title('loss_u, loss_ and loss_w history(ADO)')  
        plt.savefig('5.png')

        fig = plt.figure()
        plt.plot(self.loss_f_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(ADO)')  
        plt.savefig('6.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_U_val_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_U_val history(ADO)')  
        plt.savefig('7.png')
                                    
# =============================================================================
#            plot loss histories in STRidge                   
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_f_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f history of STRidge')  
        plt.savefig('8.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda history of STRidge')
        plt.savefig('9.png')
        
        fig = plt.figure()
        plt.plot(self.tol_history_STRidge[1:])
        plt.title('Tolerance History of STRidge')
        plt.savefig('10.png')
                        
    def inference(self, X_star):        
        tf_dict = {self.x_tf: X_star[:,0:1], self.y_tf: X_star[:,1:2], self.t_tf: X_star[:,2:3]}
            
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        
        return u_star, v_star, w_star
        
                    
        
