# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the discovery of nonlinear Schrodinger equation with a single dataset 
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
        self.loss_r_history_Pretrain = np.array([0]) # real part
        self.loss_i_history_Pretrain = np.array([0]) # imaginary part
        self.loss_f_history_Pretrain = np.array([0])
        self.loss_lambda_history_Pretrain = np.array([0])
        self.lambda_history_Pretrain = np.zeros([40, 1])
        self.step_Pretrain = 0
        self.loss_u_val_history_Pretrain = np.array([0])
        
        # loss histories for ADO
        self.loss_r_history = np.array([0])
        self.loss_i_history = np.array([0])
        self.loss_f_history = np.array([0])
        self.loss_u_val_history = np.array([0])
        self.step_ADO = 0
        
        # STRidge loss histories for ADO
        self.loss_f_history_STRidge = np.array([0])
        self.loss_lambda_history_STRidge = np.array([0])
        self.tol_history_STRidge = np.array([0])
                            
        self.lambda_history_STRidge = np.zeros([40, 1])
        self.ridge_append_counter_STRidge = np.array([0])
                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # library coefficients
        self.lambda1 = tf.Variable(tf.zeros([40, 1], dtype=tf.float32)) 
                    
        # Specify the list of trainable variables 
        var_list_Pretrain = self.biases + self.weights + [self.lambda1]
        var_list_ADO = self.biases + self.weights

# =============================================================================
#       define losses
# =============================================================================
        # data losses
        self.x_tf = tf.placeholder(tf.float32)
        self.t_tf = tf.placeholder(tf.float32)
        self.r_tf = tf.placeholder(tf.float32)
        self.i_tf = tf.placeholder(tf.float32)
        self.r_pred, self.i_pred = self.predict_response(self.x_tf, self.t_tf) 
        self.loss_r = tf.reduce_mean(tf.square(self.r_tf - self.r_pred)) # real part data loss for training
        self.loss_i = tf.reduce_mean(tf.square(self.i_tf - self.i_pred)) # imaginary part data loss for training
            
        self.x_val_tf = tf.placeholder(tf.float32)
        self.t_val_tf = tf.placeholder(tf.float32)
        self.r_val_tf = tf.placeholder(tf.float32)
        self.i_val_tf = tf.placeholder(tf.float32)               
        self.r_val_pred, self.i_val_pred = self.predict_response(self.x_val_tf, self.t_val_tf) 
        self.loss_r_val = tf.reduce_mean(tf.square(self.r_val_tf - self.r_val_pred)) # real part data loss for validation
        self.loss_i_val = tf.reduce_mean(tf.square(self.i_val_tf - self.i_val_pred)) # imaginary part data loss for validation
        self.loss_u_val = self.loss_r_val  + self.loss_i_val

        # physics loss
        self.x_f_tf = tf.placeholder(tf.float32)
        self.t_f_tf = tf.placeholder(tf.float32)        
        self.f_pred, self.Phi, self.iu_t_pred = self.physics_residue(self.x_f_tf, self.t_f_tf)        
        self.loss_f = tf.reduce_mean(tf.square(tf.abs(self.f_pred)))
        
        # L1 regularization for library coefficients
        self.loss_lambda = 1e-5*tf.norm(self.lambda1, ord=1)
        
        # total loss
        self.loss = tf.log(self.loss_r  + self.loss_i + self.loss_f + self.loss_lambda) 
                            
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

    def predict_response(self, x, t):  
        Y = self.FCNet(tf.concat([x,t],1), self.weights, self.biases)
        r = Y[:, 0:1]
        imag = Y[:, 1:2]
        return r, imag

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
                
    def physics_residue(self, x, t):
        r, imag = self.predict_response(x,t)
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
        u_ABS = tf.complex(tf.abs(u), tf.zeros_like(r, optimize = False))
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
                
        lib_fun = [tf.ones_like(u, dtype=tf.complex64)] + deri + poly + abs_u + \
                    poly_abs + deri_poly + deri_abs + deri_poly_abs
        self.lib_descr = ['1'] + deri_descr + poly_descr + abs_u_descr + poly_abs_descr + \
                    deri_poly_descr + deri_abs_descr + deri_poly_abs_descr
                    
        Phi = tf.transpose(tf.reshape(tf.convert_to_tensor(lib_fun), [40, -1]))            
        f = iu_t - tf.matmul(Phi, tf.complex(self.lambda1, tf.zeros([40, 1], dtype=tf.float32))) 
        
        return f, Phi, iu_t        
            
    def train(self, X, r, imag, X_f, X_val, r_val, i_val):
        # training measurements
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.r = r
        self.i = imag
        
        # validation measurements
        self.x_val = X_val[:,0:1]
        self.t_val = X_val[:,1:2]
        self.r_val = r_val
        self.i_val = i_val
        
        # physics collocation points
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        
        self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.r_tf: self.r, self.i_tf: self.i, 
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.r_val_tf: self.r_val,
                       self.i_val_tf: self.i_val}    
        
        print('Adam pretraining begins')
        for it_Adam in range(self.Adam_epochs_Pre):
            self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict)
            if it_Adam % 10 == 0:
                loss_r, loss_i, loss_f, loss_lambda, loss_u_val, lambda1 = self.sess.run([self.loss_r, self.loss_i, self.loss_f, self.loss_lambda, self.loss_u_val, self.lambda1], self.tf_dict)                   
                self.loss_u_val_history_Pretrain = np.append(self.loss_u_val_history_Pretrain, loss_u_val)
                self.loss_r_history_Pretrain = np.append(self.loss_r_history_Pretrain, loss_r)
                self.loss_i_history_Pretrain = np.append(self.loss_i_history_Pretrain, loss_i)
                self.loss_f_history_Pretrain = np.append(self.loss_f_history_Pretrain, loss_f)
                self.loss_lambda_history_Pretrain = np.append(self.loss_lambda_history_Pretrain, loss_lambda)
                self.lambda_history_Pretrain = np.append(self.lambda_history_Pretrain, lambda1, axis = 1)
                print("Adam epoch(Pretrain) %s : loss_r = %10.3e, loss_i = %10.3e, loss_f = %10.3e, loss_u_val = %10.3e, loss_lambda = %10.3e" % (it_Adam, loss_r, loss_i, loss_f, loss_u_val, loss_lambda))
        
        print('L-BFGS-B pretraining begins')
        self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                feed_dict = self.tf_dict,
                                fetches = [self.loss_r, self.loss_i,
                                           self.loss_f, 
                                           self.loss_lambda, 
                                           self.loss_u_val,
                                           self.lambda1],
                                loss_callback = self.callback_Pretrain)
        
        self.tol_best_ADO = 0
        
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_r, loss_i, loss_f, loss_u_val = self.sess.run([self.loss_r, self.loss_i, self.loss_f, self.loss_u_val], self.tf_dict)                   
                    self.loss_u_val_history = np.append(self.loss_u_val_history, loss_u_val)
                    self.loss_r_history = np.append(self.loss_r_history, loss_r)
                    self.loss_i_history = np.append(self.loss_i_history, loss_i)
                    self.loss_f_history = np.append(self.loss_f_history, loss_f)
                    print("Adam epoch(ADO) %s : loss_r = %10.3e, loss_i = %10.3e, loss_f = %10.3e, loss_u_val = %10.3e" % (it_Adam, loss_r, loss_i, loss_f, loss_u_val))                

            print('L-BFGS-B begins')
            self.optimizer_BFGS_ADO.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_r, self.loss_i, self.loss_f, self.loss_u_val],
                                    loss_callback = self.callback_ADO)

            print('STRidge begins')
            self.callTrainSTRidge()
        
    def callback_Pretrain(self, loss_r, loss_i, loss_f, loss_lambda, loss_u_val, lambda1):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:
                        
            self.loss_u_val_history_Pretrain = np.append(self.loss_u_val_history_Pretrain, loss_u_val)
            self.loss_r_history_Pretrain = np.append(self.loss_r_history_Pretrain, loss_r)
            self.loss_i_history_Pretrain = np.append(self.loss_i_history_Pretrain, loss_i)
            self.loss_f_history_Pretrain = np.append(self.loss_f_history_Pretrain, loss_f)
            self.loss_lambda_history_Pretrain = np.append(self.loss_lambda_history_Pretrain, loss_lambda)
            self.lambda_history_Pretrain = np.append(self.lambda_history_Pretrain, lambda1, axis = 1)
            print("BFGS epoch(Pretrain) %s : loss_r = %10.3e, loss_i = %10.3e, loss_f = %10.3e, loss_u_val = %10.3e, loss_lambda = %10.3e" % (self.step_Pretrain, loss_r, loss_i, loss_f, loss_u_val, loss_lambda))
            
    def callback_ADO(self, loss_r, loss_i, loss_f, loss_u_val):
        self.step_ADO = self.step_ADO + 1
        if self.step_ADO%10 == 0:                        
            self.loss_u_val_history = np.append(self.loss_u_val_history, loss_u_val)
            self.loss_r_history = np.append(self.loss_r_history, loss_r)
            self.loss_i_history = np.append(self.loss_i_history, loss_i)
            self.loss_f_history = np.append(self.loss_f_history, loss_f)
            print("BFGS epoch(ADO) %s : loss_r = %10.3e, loss_i = %10.3e, loss_f = %10.3e, loss_u_val = %10.3e" % (self.step_ADO, loss_r, loss_i, loss_f, loss_u_val))                

    def callTrainSTRidge(self):
        d_tol = 0.01
        maxit = 100
        Phi, iu_t_pred = self.sess.run([self.Phi, self.iu_t_pred], self.tf_dict)
        lambda2 = self.TrainSTRidge(Phi, iu_t_pred, d_tol, maxit) 
        self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype = tf.float32))

    def TrainSTRidge(self, Phi, ut, d_tol, maxit, STR_iters = 10, l0_penalty = None):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================      
        # First normalize data
        n,d = Phi.shape        
        Phi_normalized = np.zeros((n,d), dtype=np.complex64)
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(Phi[:,i],2))
            Phi_normalized[:,i] = Mreg[i]*Phi[:,i]
            
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol + self.tol_best_ADO
        if l0_penalty == None: 
            l0_penalty = 0.05*np.linalg.cond(Phi_normalized)
                    
        # inherit Lambda
        lambda_best_normalized = self.sess.run(self.lambda1)/Mreg + 0*1j
        
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
            lambda_normalized = self.STRidge(Phi_normalized, ut, STR_iters, tol, Mreg)
            err_f = np.linalg.norm(ut - Phi_normalized.dot(lambda_normalized), 2)
            err_lambda = l0_penalty*np.count_nonzero(lambda_normalized)
            err = err_f + err_lambda
    
            if err <= err_best:
                # update the optimal setting if the total error decreases
                err_best = err
                lambda_best_normalized = lambda_normalized
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
        lambda1_normalized = self.sess.run(self.lambda1)/Mreg + 0*1j         
        
        # find big coefficients
        biginds = np.where(abs(lambda1_normalized*Mreg) > tol)[0]
        num_relevant = d            
        
        # record lambda evolution
        ridge_append_counter = 0
        self.lambda_history_STRidge = np.append(self.lambda_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
        ridge_append_counter += 1

        # Threshold small coefficients until convergence
        for j in range(STR_iters):  
            # Figure out which items to cut out
            smallinds = np.where(abs(lambda1_normalized*Mreg) < tol)[0]
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
# =============================================================================
#         plot loss histories in pretraining        
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_r_history_Pretrain[1:])
        plt.plot(self.loss_i_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history(Pretraining)')  
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
        
# =============================================================================
#         plot loss histories in ADO               
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_r_history[1:])
        plt.plot(self.loss_i_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_r', 'loss_i'))
        plt.title('loss_r and loss_i history(ADO)')  
        plt.savefig('5.png')

        fig = plt.figure()
        plt.plot(self.loss_f_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history(ADO)')  
        plt.savefig('6.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_u_val_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u_val history(ADO)')  
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
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}        
        r_star = self.sess.run(self.r_pred, tf_dict)
        i_star = self.sess.run(self.i_pred, tf_dict)
        return r_star, i_star
        
                    
        
