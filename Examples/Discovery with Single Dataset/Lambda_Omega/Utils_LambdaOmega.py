# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the discovery of lambda-omega equation with a single dataset 
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
        self.loss_f_u_history_Pretrain = np.array([0])
        self.loss_f_v_history_Pretrain = np.array([0])
        self.loss_lambda_u_history_Pretrain = np.array([0])
        self.loss_lambda_v_history_Pretrain = np.array([0])
        self.lambda_u_history_Pretrain = np.zeros([110, 1])
        self.lambda_v_history_Pretrain = np.zeros([110, 1])
        self.step_Pretrain = 0
        self.loss_U_val_history_Pretrain = np.array([0])
        
        # loss histories for ADO
        self.loss_u_history = np.array([0])
        self.loss_v_history = np.array([0])
        self.loss_f_u_history = np.array([0])
        self.loss_f_v_history = np.array([0])
        self.loss_U_val_history = np.array([0])
        self.step_ADO = 0
        
        # STRidge loss histories for ADO
        self.loss_f_u_history_STRidge = np.array([0])
        self.loss_lambda_u_history_STRidge = np.array([0])
        self.tol_u_history_STRidge = np.array([0])                            
        self.lambda_u_history_STRidge = np.zeros([110, 1])
        self.ridge_u_append_counter_STRidge = np.array([0])

        self.loss_f_v_history_STRidge = np.array([0])
        self.loss_lambda_v_history_STRidge = np.array([0])
        self.tol_v_history_STRidge = np.array([0])                            
        self.lambda_v_history_STRidge = np.zeros([110, 1])
        self.ridge_v_append_counter_STRidge = np.array([0])

                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        # library coefficients
        self.lambda_u = tf.Variable(tf.zeros([110, 1], dtype=tf.float32))
        self.lambda_v = tf.Variable(tf.zeros([110, 1], dtype=tf.float32)) 
                    
        # Specify the list of trainable variables 
        var_list_Pretrain = self.biases + self.weights + [self.lambda_u] + [self.lambda_v]
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
        self.u_pred, self.v_pred = self.predict_response(self.x_tf, self.y_tf, self.t_tf)         
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
            
        self.x_val_tf = tf.placeholder(tf.float32)
        self.y_val_tf = tf.placeholder(tf.float32)
        self.t_val_tf = tf.placeholder(tf.float32)
        self.u_val_tf = tf.placeholder(tf.float32)
        self.v_val_tf = tf.placeholder(tf.float32) 
        self.w_val_tf = tf.placeholder(tf.float32)              
        self.u_val_pred, self.v_val_pred = self.predict_response(self.x_val_tf, self.y_val_tf, self.t_val_tf) 
        self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
        self.loss_v_val = tf.reduce_mean(tf.square(self.v_val_tf - self.v_val_pred))
        self.loss_U_val = self.loss_u_val  + self.loss_v_val

        # physics loss
        self.x_f_tf = tf.placeholder(tf.float32)
        self.y_f_tf = tf.placeholder(tf.float32)
        self.t_f_tf = tf.placeholder(tf.float32)        
        self.f_u_pred, self.f_v_pred, self.Phi, self.u_t_pred, self.v_t_pred = self.physics_residue(self.x_f_tf, self.y_f_tf, self.t_f_tf)        
        self.loss_f_u = tf.reduce_mean(tf.square(self.f_u_pred))
        self.loss_f_v = tf.reduce_mean(tf.square(self.f_v_pred))

        # L1 regularization for library coefficients
        self.loss_lambda_u = tf.norm(self.lambda_u, ord=1)    
        self.loss_lambda_v = tf.norm(self.lambda_v, ord=1) 
        
        # total loss for pretraining and ADO
        self.loss = tf.log(self.loss_u  + self.loss_v + 10*self.loss_f_u + 10*self.loss_f_v + 1e-10*self.loss_lambda_u + 1e-10*self.loss_lambda_v) 
                            
        # post-training loss
        # self.nonzero_mask_lambda_u_tf = tf.placeholder(tf.float32)
        # self.nonzero_mask_lambda_v_tf = tf.placeholder(tf.float32)
        # self.f_u_pred_pt, self.f_v_pred_pt = self.physics_residue_pt(self.x_f_tf, self.y_f_tf, self.t_f_tf)
        # self.loss_f_u_pt = tf.reduce_mean(tf.square(self.f_u_pred_pt))
        # self.loss_f_v_pt = tf.reduce_mean(tf.square(self.f_v_pred_pt))
        # self.loss_pt = tf.log(self.loss_u  + self.loss_v + 10*self.loss_f_u_pt + 10*self.loss_f_v_pt)
        
# =============================================================================
#       define optimizers
# =============================================================================
        # optimizers for pretraining
        self.optimizer_Adam_Pretrain = tf.train.AdamOptimizer(learning_rate = 5e-3) 
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
        self.optimizer_Adam_ADO = tf.train.AdamOptimizer(learning_rate = 1e-3) 
        self.train_op_Adam_ADO = self.optimizer_Adam_ADO.minimize(self.loss, var_list = var_list_ADO)

        self.optimizer_BFGS_ADO = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                var_list = var_list_ADO,
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': self.BFGS_epochs_ADO,
                                                                           'maxfun': self.BFGS_epochs_ADO,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : np.finfo(float).eps})
        
        # optimizer for post-training
        # self.optimizer_Adam_pt = tf.train.AdamOptimizer(learning_rate = 1e-3) 
        # self.train_op_Adam_pt = self.optimizer_Adam_pt.minimize(self.loss_pt, var_list = var_list_ADO)

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
        return u, v

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
        u, v = self.predict_response(x,y,t)
        data = [u, v]            
            
        ## derivatives     
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]
        u_xy = tf.gradients(u_x,y)[0]
        u_t = tf.gradients(u,t)[0]
        
        v_x = tf.gradients(v,x)[0]
        v_xx = tf.gradients(v_x,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_yy = tf.gradients(v_y,y)[0]
        v_xy = tf.gradients(v_x,y)[0]
        v_t = tf.gradients(v,t)[0]
        
        derivatives = [tf.ones_like(data[0], optimize = False)]
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_y)
        derivatives.append(u_yy)
        derivatives.append(u_xy)
        derivatives.append(v_x)
        derivatives.append(v_xx)
        derivatives.append(v_y)
        derivatives.append(v_yy)
        derivatives.append(v_xy)
        
        derivatives_description = ['', 'u_{x}', 'u_{xx}', 'u_{y}','u_{yy}','u_{xy}','v_{x}', 'v_{xx}', 'v_{y}','v_{yy}','v_{xy}']
        
        lib_fun, self.lib_descr = self.build_library(data, derivatives, derivatives_description, PolyOrder = 3, 
                                                    data_description = ['u','v'])  
        Phi = tf.concat(lib_fun, 1)
        f_u = u_t - Phi@self.lambda_u           
        f_v = v_t - Phi@self.lambda_v           
                                    
        return f_u, f_v, Phi, u_t, v_t

    def physics_residue_pt(self, x, y, t):
        # for post-training
        u, v = self.predict_response(x,y,t)
        data = [u, v]            
            
        ## derivatives     
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]
        u_xy = tf.gradients(u_x,y)[0]
        u_t = tf.gradients(u,t)[0]
        
        v_x = tf.gradients(v,x)[0]
        v_xx = tf.gradients(v_x,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_yy = tf.gradients(v_y,y)[0]
        v_xy = tf.gradients(v_x,y)[0]
        v_t = tf.gradients(v,t)[0]
        
        derivatives = [tf.ones_like(data[0], optimize = False)]
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_y)
        derivatives.append(u_yy)
        derivatives.append(u_xy)
        derivatives.append(v_x)
        derivatives.append(v_xx)
        derivatives.append(v_y)
        derivatives.append(v_yy)
        derivatives.append(v_xy)
        
        derivatives_description = ['', 'u_{x}', 'u_{xx}', 'u_{y}','u_{yy}','u_{xy}','v_{x}', 'v_{xx}', 'v_{y}','v_{yy}','v_{xy}']
        
        lib_fun, self.lib_descr = self.build_library(data, derivatives, derivatives_description, PolyOrder = 3, 
                                                    data_description = ['u','v'])  
        Phi = tf.concat(lib_fun, 1)
        f_u = u_t - Phi@(self.lambda_u*self.nonzero_mask_lambda_u_tf)
        f_v = v_t - Phi@(self.lambda_v*self.nonzero_mask_lambda_v_tf)
                                    
        return f_u, f_v
            
    def build_library(self, data, derivatives, derivatives_description, PolyOrder, data_description = None):         
        ## polynomial terms
        P = PolyOrder
        lib_poly = [tf.ones_like(data[0], optimize = False)]
        lib_poly_descr = [''] # it denotes '1'
        for i in range(len(data)): # polynomial terms of univariable
            for j in range(1, P+1):
                lib_poly.append(data[i]**j)
                lib_poly_descr.append(data_description[i]+"**"+str(j))
                
        for i in range(1,P): # polynomial terms of bivariable. Assume we only have 2 variables.
            for j in range(1,P-i+1):
                lib_poly.append(data[0]**i*data[1]**j)
                lib_poly_descr.append(data_description[0]+"**"+str(i)+data_description[1]+"**"+str(j))
                
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
    
    def train(self, X, u, v, X_f, X_val, u_val, v_val):
        # training measurements
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]       
        self.u = u
        self.v = v
        # validation measurements
        self.x_val = X_val[:,0:1]
        self.y_val = X_val[:,1:2]
        self.t_val = X_val[:,2:3]
        self.u_val = u_val
        self.v_val = v_val
        
        # physics collocation points
        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]
        self.t_f = X_f[:,2:3]
        
        self.tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                        self.u_tf: self.u, self.v_tf: self.v, 
                       self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f,
                       self.x_val_tf: self.x_val, self.y_val_tf: self.y_val, self.t_val_tf: self.t_val,
                       self.u_val_tf: self.u_val, self.v_val_tf: self.v_val}    
        
        print('Adam pretraining begins')
        for it_Adam in range(self.Adam_epochs_Pre):
            self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict)
            if it_Adam % 10 == 0:
                loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, loss_U_val, lambda_u, lambda_v = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_lambda_u, self.loss_lambda_v, self.loss_U_val, self.lambda_u, self.lambda_v], self.tf_dict)                   
                self.loss_U_val_history_Pretrain = np.append(self.loss_U_val_history_Pretrain, loss_U_val)
                self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
                self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
                self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
                self.loss_f_v_history_Pretrain = np.append(self.loss_f_v_history_Pretrain, loss_f_v)
                self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
                self.loss_lambda_v_history_Pretrain = np.append(self.loss_lambda_v_history_Pretrain, loss_lambda_v)
                self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
                self.lambda_v_history_Pretrain = np.append(self.lambda_v_history_Pretrain, lambda_v, axis = 1)
                print("Adam epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_U_val = %10.3e, loss_lambda_u = %10.3e, loss_lambda_v = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val, loss_lambda_u, loss_lambda_v))
        
        print('L-BFGS-B pretraining begins')
        self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                feed_dict = self.tf_dict,
                                fetches = [self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_lambda_u, self.loss_lambda_v, self.loss_U_val, self.lambda_u, self.lambda_v],
                                loss_callback = self.callback_Pretrain)
        self.tol_best_ADO_u = 0
        self.tol_best_ADO_v = 0
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_U_val], self.tf_dict)                   
                    self.loss_U_val_history = np.append(self.loss_U_val_history, loss_U_val)
                    self.loss_u_history = np.append(self.loss_u_history, loss_u)
                    self.loss_v_history = np.append(self.loss_v_history, loss_v)
                    self.loss_f_u_history = np.append(self.loss_f_u_history, loss_f_u)
                    self.loss_f_v_history = np.append(self.loss_f_v_history, loss_f_v)
                    print("Adam epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_U_val = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val))                

            print('L-BFGS-B begins')
            self.optimizer_BFGS_ADO.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_U_val],
                                    loss_callback = self.callback_ADO)

            print('STRidge begins')
            self.callTrainSTRidge()
        
    def callback_Pretrain(self, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, loss_U_val, lambda_u, lambda_v):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:                        
            self.loss_U_val_history_Pretrain = np.append(self.loss_U_val_history_Pretrain, loss_U_val)
            self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
            self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
            self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
            self.loss_f_v_history_Pretrain = np.append(self.loss_f_v_history_Pretrain, loss_f_v)
            self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
            self.loss_lambda_v_history_Pretrain = np.append(self.loss_lambda_v_history_Pretrain, loss_lambda_v)
            self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
            self.lambda_v_history_Pretrain = np.append(self.lambda_v_history_Pretrain, lambda_v, axis = 1)
            print("BFGS epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_U_val = %10.3e, loss_lambda_u = %10.3e, loss_lambda_v = %10.3e" % (self.step_Pretrain, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val, loss_lambda_u, loss_lambda_v))
            
    def callback_ADO(self, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val):
        self.step_ADO = self.step_ADO + 1
        if self.step_ADO%10 == 0:                        
            self.loss_U_val_history = np.append(self.loss_U_val_history, loss_U_val)
            self.loss_u_history = np.append(self.loss_u_history, loss_u)
            self.loss_v_history = np.append(self.loss_v_history, loss_v)
            self.loss_f_u_history = np.append(self.loss_f_u_history, loss_f_u)
            self.loss_f_v_history = np.append(self.loss_f_v_history, loss_f_v)
            print("BFGS epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_U_val = %10.3e" % (self.step_ADO, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val))                

    def callTrainSTRidge(self):
        d_tol = 1
        maxit = 25
        
        # lambda_u
        Phi, u_t_pred, v_t_pred = self.sess.run([self.Phi, self.u_t_pred, self.v_t_pred], self.tf_dict)
        lambda_u2 = self.TrainSTRidge(Phi, u_t_pred, d_tol, maxit, uv_flag = True) 
        self.lambda_u = tf.assign(self.lambda_u, tf.convert_to_tensor(lambda_u2, dtype = tf.float32))
        
        # lambda_v
        lambda_v2 = self.TrainSTRidge(Phi, v_t_pred, d_tol, maxit, uv_flag = False) 
        self.lambda_v = tf.assign(self.lambda_v, tf.convert_to_tensor(lambda_v2, dtype = tf.float32))

    def TrainSTRidge(self, Phi, ut, d_tol, maxit, STR_iters = 10, l0_penalty = None, uv_flag = True):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================      
            
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        
        if uv_flag:
            tol = d_tol + self.tol_best_ADO_u
            tol_best = self.tol_best_ADO_u                
        else:
            tol = d_tol + self.tol_best_ADO_v
            tol_best = self.tol_best_ADO_v                
            
        if l0_penalty == None: 
            l0_penalty = 1e-3*np.linalg.cond(Phi)
                    
        # inherit Lambda
        if uv_flag:
            lambda_best = self.sess.run(self.lambda_u)
        else:
            lambda_best = self.sess.run(self.lambda_v)
        
        # record initial sparsity and regression accuracy, and set them as the best
        err_f = np.linalg.norm(ut - Phi.dot(lambda_best), 2)
        err_lambda = l0_penalty*np.count_nonzero(lambda_best)
        err_best = err_f + err_lambda
        self.loss_f_u_history_STRidge = np.append(self.loss_f_u_history_STRidge, err_f)
        self.loss_lambda_u_history_STRidge = np.append(self.loss_lambda_u_history_STRidge, err_lambda)
        self.tol_u_history_STRidge = np.append(self.tol_u_history_STRidge, tol_best)
    
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
            # Get a set of coefficients and error
            lambda1 = self.STRidge(Phi, ut, STR_iters, tol, uv_flag = uv_flag)
            err_f = np.linalg.norm(ut - Phi.dot(lambda1), 2)
            err_lambda = l0_penalty*np.count_nonzero(lambda1)
            err = err_f + err_lambda
    
            if err <= err_best:
                # update the optimal setting if the total error decreases
                err_best = err
                lambda_best = lambda1
                tol_best = tol
                tol = tol + d_tol
                
                if uv_flag:
                    self.loss_f_u_history_STRidge = np.append(self.loss_f_u_history_STRidge, err_f)
                    self.loss_lambda_u_history_STRidge = np.append(self.loss_lambda_u_history_STRidge, err_lambda)
                    self.tol_u_history_STRidge = np.append(self.tol_u_history_STRidge, tol_best)
                else: 
                    self.loss_f_v_history_STRidge = np.append(self.loss_f_v_history_STRidge, err_f)
                    self.loss_lambda_v_history_STRidge = np.append(self.loss_lambda_v_history_STRidge, err_lambda)
                    self.tol_v_history_STRidge = np.append(self.tol_v_history_STRidge, tol_best)
    
            else:
                # otherwise decrease tol and try again
                tol = max([0,tol - 2*d_tol])
                d_tol = 2*d_tol / (maxit - iter)
                tol = tol + d_tol
        if uv_flag:
            self.tol_best_ADO_u = tol_best
        else:
            self.tol_best_ADO_v = tol_best
        
        return np.real(lambda_best)  

    def STRidge(self, Phi, ut, STR_iters, tol, uv_flag):    
        # First normalize data
        n,d = Phi.shape        
        Phi_normalized = np.zeros((n,d), dtype=np.complex64)
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(Phi[:,i],2))
            Phi_normalized[:,i] = Mreg[i]*Phi[:,i]            
            
        # Inherit lambda from previous training and normalize it.
        if uv_flag:
            lambda1_normalized = self.sess.run(self.lambda_u)/Mreg
        else:
            lambda1_normalized = self.sess.run(self.lambda_v)/Mreg
        
        # find big coefficients
        biginds = np.where(abs(lambda1_normalized) > tol)[0]
        num_relevant = d            
        
        # record lambda evolution
        ridge_append_counter = 0
        ridge_append_counter = self.record_lambda_in_STRidge(uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag = False)

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
                    ridge_append_counter = self.record_lambda_in_STRidge(uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag = True)                    
                    return lambda1_normalized*Mreg
                else: 
                    break
            biginds = new_biginds
            
            # Otherwise get a new guess
            lambda1_normalized[smallinds] = 0            
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds].T.dot(Phi_normalized[:, biginds]) + 1e-5*np.eye(len(biginds)),Phi_normalized[:, biginds].T.dot(ut))[0]
            
            # record lambda evolution
            ridge_append_counter = self.record_lambda_in_STRidge(uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag = False)
            
        # Now that we have the sparsity pattern, use standard least squares to get lambda1_normalized
        if biginds != []: 
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds],ut)[0]
        
        # record lambda evolution
        ridge_append_counter = self.record_lambda_in_STRidge(uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag = True)
        return lambda1_normalized*Mreg
    
    def record_lambda_in_STRidge(self, uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag):
        ridge_append_counter += 1
        if uv_flag:
            self.lambda_u_history_STRidge = np.append(self.lambda_u_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
            if end_flag:
                self.ridge_u_append_counter_STRidge = np.append(self.ridge_u_append_counter_STRidge, ridge_append_counter)
        else:
            self.lambda_v_history_STRidge = np.append(self.lambda_v_history_STRidge, np.multiply(Mreg,lambda1_normalized), axis = 1)
            if end_flag:
                self.ridge_v_append_counter_STRidge = np.append(self.ridge_v_append_counter_STRidge, ridge_append_counter)
        return ridge_append_counter
        
    def visualize_training(self):
# =============================================================================
#         plot loss histories in pretraining        
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_u_history_Pretrain[1:])
        plt.plot(self.loss_v_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u, loss_v history(Pretraining)')  
        plt.savefig('1.png')
        
        fig = plt.figure()
        plt.plot(self.loss_f_u_history_Pretrain[1:])
        plt.plot(self.loss_f_v_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f history(Pretraining)')     
        plt.savefig('2.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_u_history_Pretrain[1:])
        plt.plot(self.loss_lambda_v_history_Pretrain[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_lambda_u', 'loss_lambda_v'))
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
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u, loss_v history(ADO)')  
        plt.savefig('5.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history[1:])
        plt.plot(self.loss_f_v_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_u', 'loss_f_v'))
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
        plt.plot(self.loss_f_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f_u history of STRidge')  
        plt.savefig('8.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda_u history of STRidge')
        plt.savefig('9.png')
        
        fig = plt.figure()
        plt.plot(self.tol_u_history_STRidge[1:])
        plt.title('Tolerance_u History of STRidge')
        plt.savefig('10.png')

        fig = plt.figure()
        plt.plot(self.loss_f_v_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f_v history of STRidge')  
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_v_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda_v history of STRidge')
        plt.savefig('12.png')
        
        fig = plt.figure()
        plt.plot(self.tol_v_history_STRidge[1:])
        plt.title('Tolerance_v History of STRidge')
        plt.savefig('13.png')
                        
    def visualize_post_training(self):
        fig = plt.figure()
        plt.plot(self.loss_u_history_pt[1:])
        plt.plot(self.loss_v_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u, loss_v history(post-training)')  
        plt.savefig('14.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history_pt[1:])
        plt.plot(self.loss_f_v_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f history(post-training)')  
        plt.savefig('15.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_U_val_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_U_val history(post-training)')  
        plt.savefig('16.png')
        
    def inference(self, X_star):        
        tf_dict = {self.x_tf: X_star[:,0:1], self.y_tf: X_star[:,1:2], self.t_tf: X_star[:,2:3]}
            
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        
        return u_star, v_star
    
    def post_train(self, pt_ADO_iterations, Adam_epochs_Pt):
                
        # loss histories for post-training
        self.loss_u_history_pt = np.array([0])
        self.loss_v_history_pt = np.array([0])
        self.loss_f_u_history_pt = np.array([0])
        self.loss_f_v_history_pt = np.array([0])
        self.lambda_u_history_pt = np.zeros([110, 1])
        self.lambda_v_history_pt = np.zeros([110, 1])
        self.loss_U_val_history_pt = np.array([0])
        
        print('post-training begins')
        for it in tqdm(range(pt_ADO_iterations)):
# =============================================================================
#              update library coefficients lambda_u and lambda_v via least squares
# =============================================================================
            print('least squares begins')
            # find non-zero values in library coefficients
            lambda_u, lambda_v = self.sess.run([self.lambda_u, self.lambda_v])
            nonzero_ind_u = np.nonzero(lambda_u)            
            nonzero_ind_v = np.nonzero(lambda_v)

            # form compact libraries Phi_u_compact & Phi_v_compact that only have non-zero candidates
            Phi, u_t, v_t = self.sess.run([self.Phi, self.u_t_pred, self.v_t_pred], self.tf_dict) 
            Phi_u_compact = Phi[:, nonzero_ind_u[0]] 
            Phi_v_compact = Phi[:, nonzero_ind_v[0]] 

            # normalize Phi_u_compact and Phi_v_compact
            Phi_u_compact_norm = np.zeros_like(Phi_u_compact)
            Mreg_u = np.zeros((Phi_u_compact.shape[1], 1))
            for it_Phi_u_compact in range(Phi_u_compact.shape[1]):
                Mreg_u[it_Phi_u_compact] = 1.0/(np.linalg.norm(Phi_u_compact[:,it_Phi_u_compact], 2))
                Phi_u_compact_norm[:,it_Phi_u_compact] = Mreg_u[it_Phi_u_compact]*Phi_u_compact[:,it_Phi_u_compact]      
                
            Phi_v_compact_norm = np.zeros_like(Phi_v_compact)
            Mreg_v = np.zeros((Phi_v_compact.shape[1], 1))
            for it_Phi_v_compact in range(Phi_v_compact.shape[1]):
                Mreg_v[it_Phi_v_compact] = 1.0/(np.linalg.norm(Phi_v_compact[:,it_Phi_v_compact], 2))
                Phi_v_compact_norm[:,it_Phi_v_compact] = Mreg_v[it_Phi_v_compact]*Phi_v_compact[:,it_Phi_v_compact]      
            
            # do least square to update non-zero values in lambda_u and lambda_v
            lambda_u_updated_compact = np.linalg.lstsq(Phi_u_compact_norm, u_t)[0]*Mreg_u
            lambda_v_updated_compact = np.linalg.lstsq(Phi_v_compact_norm, v_t)[0]*Mreg_v
                                    
            # assign updated values to self.lambda_u
            lambda_u_updated = np.zeros_like(lambda_u)
            lambda_u_updated[nonzero_ind_u] = lambda_u_updated_compact
            self.lambda_u = tf.assign(self.lambda_u, tf.convert_to_tensor(lambda_u_updated, dtype = tf.float32))

            # assign updated values to self.lambda_v
            lambda_v_updated = np.zeros_like(lambda_v)
            lambda_v_updated[nonzero_ind_v] = lambda_v_updated_compact
            self.lambda_v = tf.assign(self.lambda_v, tf.convert_to_tensor(lambda_v_updated, dtype = tf.float32))

            self.lambda_u_history_pt = np.append(self.lambda_u_history_pt, lambda_u_updated, axis = 1)
            self.lambda_v_history_pt = np.append(self.lambda_v_history_pt, lambda_v_updated, axis = 1)

# =============================================================================
#              update NN weights and bias via Adam
# =============================================================================
            # mark non-zero candidates in the library 
            nonzero_mask_lambda_u = np.zeros_like(lambda_u)
            nonzero_mask_lambda_u[nonzero_ind_u] = 1
            self.tf_dict[self.nonzero_mask_lambda_u_tf] = nonzero_mask_lambda_u
            
            nonzero_mask_lambda_v = np.zeros_like(lambda_v)
            nonzero_mask_lambda_v[nonzero_ind_v] = 1
            self.tf_dict[self.nonzero_mask_lambda_v_tf] = nonzero_mask_lambda_v

            print('Adam begins')
            for it_Adam in range(Adam_epochs_Pt):
                self.sess.run(self.train_op_Adam_pt, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u_pt, self.loss_f_v_pt, self.loss_U_val], self.tf_dict)
                    self.loss_U_val_history_pt = np.append(self.loss_U_val_history_pt, loss_U_val)
                    self.loss_u_history_pt = np.append(self.loss_u_history_pt, loss_u)
                    self.loss_v_history_pt = np.append(self.loss_v_history_pt, loss_v)
                    self.loss_f_u_history_pt = np.append(self.loss_f_u_history_pt, loss_f_u)
                    self.loss_f_v_history_pt = np.append(self.loss_f_v_history_pt, loss_f_v)
                    print("Adam epoch(Pt-ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_U_val = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_U_val))                

# =============================================================================
#       determine whether the post-training is sufficient
# =============================================================================
        self.visualize_post_training()
                    
        
