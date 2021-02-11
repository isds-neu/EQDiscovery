# =============================================================================
# Physics-informed learning of governing equations from scarce data
# Zhao Chen, Yang Liu, and Hao Sun
# 2021. Northeastern University

# Utility script for the discovery of FN equation with multiple datasets 
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
    def __init__(self, layers_s, layers_i, lb, ub, pre_ADO_iterations, Adam_epochs_Pre, BFGS_epochs_Pre, ADO_iterations, Adam_epochs_ADO, BFGS_epochs_ADO):

        np.random.seed(1234)
        tf.set_random_seed(1234)

        self.lb = lb
        self.ub = ub
        self.pre_ADO_iterations = pre_ADO_iterations
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
        self.lambda_u_history_Pretrain = np.zeros([70, 1])
        self.lambda_v_history_Pretrain = np.zeros([70, 1])
        self.step_Pretrain = 0
        self.loss_U_val_history_Pretrain = np.array([0])
        self.loss_bc_all_history_Pretrain = np.array([0])
        self.diff_coeff_u_history_Pretrain = np.array([0])
        self.diff_coeff_v_history_Pretrain = np.array([0])
        
        # loss histories for ADO
        self.loss_u_history = np.array([0])
        self.loss_v_history = np.array([0])
        self.loss_f_u_history = np.array([0])
        self.loss_f_v_history = np.array([0])
        self.loss_U_val_history = np.array([0])
        self.loss_bc_all_history = np.array([0])
        self.step_ADO = 0
        
        # STRidge loss histories for ADO
        self.loss_f_u_history_STRidge = np.array([0])
        self.loss_lambda_u_history_STRidge = np.array([0])
        self.tol_u_history_STRidge = np.array([0])                            
        self.lambda_u_history_STRidge = np.zeros([72, 1])
        self.ridge_u_append_counter_STRidge = np.array([0])

        self.loss_f_v_history_STRidge = np.array([0])
        self.loss_lambda_v_history_STRidge = np.array([0])
        self.tol_v_history_STRidge = np.array([0])                            
        self.lambda_v_history_STRidge = np.zeros([72, 1])
        self.ridge_v_append_counter_STRidge = np.array([0])

                    
# =============================================================================
#       define trainable variables
# =============================================================================
        # NN
        self.weights_s, self.biases_s = self.initialize_NN(layers_s) # root NN
        self.weights0, self.biases0 = self.initialize_NN(layers_i) # branch NN 1
        self.weights1, self.biases1 = self.initialize_NN(layers_i) # branch NN 2
        self.weights2, self.biases2 = self.initialize_NN(layers_i) # branch NN 3
        
        # library coefficients
        self.lambda_u = tf.Variable(tf.random_uniform([70, 1], minval = -1, maxval = 1, dtype=tf.float32))
        self.lambda_v = tf.Variable(tf.random_uniform([70, 1], minval = -1, maxval = 1, dtype=tf.float32))
        self.diff_coeff_u_core = tf.Variable(tf.random_uniform([], minval = -1, maxval = 1, dtype=tf.float32))
        self.diff_coeff_v_core = tf.Variable(tf.random_uniform([], minval = -1, maxval = 1, dtype=tf.float32)) 
                    
        self.diff_u_scale = 5
        self.diff_v_scale = 150
        self.diff_coeff_u = self.coeff_activation(self.diff_coeff_u_core, self.diff_u_scale)
        self.diff_coeff_v = self.coeff_activation(self.diff_coeff_v_core, self.diff_v_scale)
        
        # Specify the list of trainable variables 
        var_list_ADO = self.biases0 + self.weights0 + \
                self.biases2 + self.weights2 + \
                self.biases1 + self.weights1 + \
                self.weights_s + self.biases_s
        var_list_Pretrain = var_list_ADO + [self.lambda_u] + [self.lambda_v] + [self.diff_coeff_u_core] + [self.diff_coeff_v_core]

# =============================================================================
#       define losses
# =============================================================================
        # data losses
        self.X_tf = tf.placeholder(tf.float32)
        self.U_tf = tf.placeholder(tf.float32)
        self.U0_pred = self.predict_response(self.X_tf, 0)
        self.U1_pred = self.predict_response(self.X_tf, 1)
        self.U2_pred = self.predict_response(self.X_tf, 2)  
        self.loss_u = tf.reduce_mean(tf.square(self.U_tf[:, 0, 0] - self.U0_pred[:, 0])) + \
                tf.reduce_mean(tf.square(self.U_tf[:, 0, 1] - self.U1_pred[:, 0])) + \
                tf.reduce_mean(tf.square(self.U_tf[:, 0, 2] - self.U2_pred[:, 0]))
        self.loss_v = tf.reduce_mean(tf.square(self.U_tf[:, 1, 0] - self.U0_pred[:, 1])) + \
                tf.reduce_mean(tf.square(self.U_tf[:, 1, 1] - self.U1_pred[:, 1])) + \
                tf.reduce_mean(tf.square(self.U_tf[:, 1, 2] - self.U2_pred[:, 1]))
                
        self.loss_u_coeff = tf.placeholder(tf.float32)
        self.loss_v_coeff = tf.placeholder(tf.float32)

        self.loss_U = self.loss_u_coeff*self.loss_u + self.loss_v_coeff*self.loss_v
            
        # physics loss
        self.x_f_tf = tf.placeholder(tf.float32)
        self.y_f_tf = tf.placeholder(tf.float32)
        self.t_f_tf = tf.placeholder(tf.float32)        
        self.f_u_pred, self.f_v_pred, self.Phi_pred, self.u_t, self.v_t, self.u_xx_pred, self.u_yy_pred, self.v_xx_pred, self.v_yy_pred = self.physics_residue(self.x_f_tf, self.y_f_tf, self.t_f_tf)        
        self.loss_f_u = tf.reduce_mean(tf.square(self.f_u_pred))
        self.loss_f_v = tf.reduce_mean(tf.square(self.f_v_pred))
        
        self.loss_f_v_coeff = tf.placeholder(tf.float32)
        self.loss_f = self.loss_f_u + self.loss_f_v_coeff*self.loss_f_v

        # periodic boundary loss
        self.x_l_tf = tf.placeholder(tf.float32)
        self.x_r_tf = tf.placeholder(tf.float32)
        self.x_u_tf = tf.placeholder(tf.float32)
        self.x_b_tf = tf.placeholder(tf.float32)
        
        self.y_l_tf = tf.placeholder(tf.float32)
        self.y_r_tf = tf.placeholder(tf.float32)
        self.y_u_tf = tf.placeholder(tf.float32)
        self.y_b_tf = tf.placeholder(tf.float32)

        self.t_l_tf = tf.placeholder(tf.float32)
        self.t_r_tf = tf.placeholder(tf.float32)
        self.t_u_tf = tf.placeholder(tf.float32)
        self.t_b_tf = tf.placeholder(tf.float32)

        self.U_l_0 = self.predict_response(tf.concat((self.x_l_tf, self.y_l_tf, self.t_l_tf), 1), 0)
        self.U_r_0 = self.predict_response(tf.concat((self.x_r_tf, self.y_r_tf, self.t_r_tf), 1), 0)
        self.U_u_0 = self.predict_response(tf.concat((self.x_u_tf, self.y_u_tf, self.t_u_tf), 1), 0)
        self.U_b_0 = self.predict_response(tf.concat((self.x_b_tf, self.y_b_tf, self.t_b_tf), 1), 0)
        
        self.U_l_1 = self.predict_response(tf.concat((self.x_l_tf, self.y_l_tf, self.t_l_tf), 1), 1)
        self.U_r_1 = self.predict_response(tf.concat((self.x_r_tf, self.y_r_tf, self.t_r_tf), 1), 1)
        self.U_u_1 = self.predict_response(tf.concat((self.x_u_tf, self.y_u_tf, self.t_u_tf), 1), 1)
        self.U_b_1 = self.predict_response(tf.concat((self.x_b_tf, self.y_b_tf, self.t_b_tf), 1), 1)

        self.U_l_2 = self.predict_response(tf.concat((self.x_l_tf, self.y_l_tf, self.t_l_tf), 1), 2)
        self.U_r_2 = self.predict_response(tf.concat((self.x_r_tf, self.y_r_tf, self.t_r_tf), 1), 2)
        self.U_u_2 = self.predict_response(tf.concat((self.x_u_tf, self.y_u_tf, self.t_u_tf), 1), 2)
        self.U_b_2 = self.predict_response(tf.concat((self.x_b_tf, self.y_b_tf, self.t_b_tf), 1), 2)
        
        self.loss_bc = tf.reduce_mean(tf.square(self.U_l_0 - self.U_r_0)) + tf.reduce_mean(tf.square(self.U_u_0 - self.U_b_0)) + \
            tf.reduce_mean(tf.square(self.U_l_1 - self.U_r_1)) + tf.reduce_mean(tf.square(self.U_u_1 - self.U_b_1)) + \
            tf.reduce_mean(tf.square(self.U_l_2 - self.U_r_2)) + tf.reduce_mean(tf.square(self.U_u_2 - self.U_b_2))
        
        # loss for measurements on boundary condition
        self.X_bc_meas_tf = tf.placeholder(tf.float32)
        self.U_bc_meas_tf = tf.placeholder(tf.float32)

        self.U0_bc_pred = self.predict_response(self.X_bc_meas_tf, 0)
        self.U1_bc_pred = self.predict_response(self.X_bc_meas_tf, 1)
        self.U2_bc_pred = self.predict_response(self.X_bc_meas_tf, 2)

        self.loss_bc_meas = tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 0, 0] - self.U0_bc_pred[:, 0])) + \
            tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 1, 0] - self.U0_bc_pred[:, 1])) + \
            tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 0, 1] - self.U1_bc_pred[:, 0])) + \
            tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 1, 1] - self.U1_bc_pred[:, 1])) + \
            tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 0, 2] - self.U2_bc_pred[:, 0])) + \
            tf.reduce_mean(tf.square(self.U_bc_meas_tf[:, 1, 2] - self.U2_bc_pred[:, 1]))
            
        self.loss_bc_all = self.loss_bc + self.loss_bc_meas
            
        # L1 regularization for library coefficients
        self.loss_lambda_u = tf.norm(self.lambda_u, ord=1)    
        self.loss_lambda_v = tf.norm(self.lambda_v, ord=1) 
        
        # pretraining loss
        self.loss = tf.log(self.loss_U + self.loss_f + 1e-5*self.loss_lambda_u + 1e-8*self.loss_lambda_v + self.loss_bc_all)
        
        # ADO loss
        self.loss_ADO = tf.log(self.loss_U + self.loss_f + self.loss_bc_all)
                            
        # post-training loss
        self.nonzero_mask_lambda_u_tf = tf.placeholder(tf.float32)
        self.nonzero_mask_lambda_v_tf = tf.placeholder(tf.float32)
        self.f_u_pred_pt, self.f_v_pred_pt, self.u_xx_pt, self.u_yy_pt, self.v_xx_pt, self.v_yy_pt = self.physics_residue_pt(self.x_f_tf, self.y_f_tf, self.t_f_tf)
        self.loss_f_u_pt = tf.reduce_mean(tf.square(self.f_u_pred_pt))
        self.loss_f_v_pt = tf.reduce_mean(tf.square(self.f_v_pred_pt))
        self.loss_pt = tf.log(self.loss_U + 10*self.loss_f_u_pt + self.loss_f_v_coeff*self.loss_f_v_pt)
        
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
        starter_learning_rate = 5e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 10000, 0.8,
                                                     staircase=True)

        # The default settings: learning rate = 1e-3, beta1 = 0.9, beta2 = 0.999， epsilon = 1e-8
        self.optimizer_Adam_ADO = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
        self.train_op_Adam_ADO = self.optimizer_Adam_ADO.minimize(self.loss_ADO, var_list = var_list_ADO, 
                                                          global_step = self.global_step)

        self.optimizer_BFGS_ADO = tf.contrib.opt.ScipyOptimizerInterface(self.loss_ADO, var_list = var_list_ADO,
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': self.BFGS_epochs_ADO,
                                                                           'maxfun': self.BFGS_epochs_ADO,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : np.finfo(float).eps})        
        # optimizer for post-training
        self.global_step_Pt = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-4
        self.learning_rate_Pt = tf.train.exponential_decay(starter_learning_rate, self.global_step_Pt, 10000, 0.8,
                                                     staircase=True)
        self.optimizer_Adam_pt = tf.train.AdamOptimizer(learning_rate = self.learning_rate_Pt) 
        self.train_op_Adam_pt = self.optimizer_Adam_pt.minimize(self.loss_pt, var_list = var_list_ADO, 
                                                          global_step = self.global_step_Pt)
            
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
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_normal(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def predict_response(self, X, IBC_flag):  
        U_int = self.FCNet(X, self.weights_s, self.biases_s, True) # use root NN
        
        if IBC_flag == 0:
            U = self.FCNet(U_int, self.weights0, self.biases0, False) # use branch NN 1
        elif IBC_flag == 1:
            U = self.FCNet(U_int, self.weights1, self.biases1, False) # use branch NN 2
        elif IBC_flag == 2:
            U = self.FCNet(U_int, self.weights2, self.biases2, False) # use branch NN 3
        return U

    def FCNet(self, X, weights, biases, si_flag):
        num_layers = len(weights) + 1    
        if si_flag: # input to root NN
            H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
            W = weights[0]
            b = biases[0]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        else: # input to branch NN
            H = X
            W = weights[0]
            b = biases[0]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        for l in range(1, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        if si_flag: # output from root NN
            Y = tf.tanh(tf.add(tf.matmul(H, W), b))
        else: # output from branch NN
            Y = tf.add(tf.matmul(H, W), b)
        return Y
                
    def physics_residue(self, x, y, t):
        U0 = self.predict_response(tf.concat((x, y, t), 1), 0)
        U1 = self.predict_response(tf.concat((x, y, t), 1), 1)
        U2 = self.predict_response(tf.concat((x, y, t), 1), 2)

        u0 = U0[:, 0:1]
        v0 = U0[:, 1:2]
        u0_t = tf.gradients(u0, t)[0]
        u0_x = tf.gradients(u0, x)[0]
        u0_xx = tf.gradients(u0_x, x)[0]
        u0_xy = tf.gradients(u0_x, y)[0]
        u0_y = tf.gradients(u0, y)[0]
        u0_yy = tf.gradients(u0_y, y)[0]
        v0_t = tf.gradients(v0, t)[0]
        v0_x = tf.gradients(v0, x)[0]
        v0_xx = tf.gradients(v0_x, x)[0]
        v0_xy = tf.gradients(v0_x, y)[0]
        v0_y = tf.gradients(v0, y)[0]
        v0_yy = tf.gradients(v0_y, y)[0]

        u1 = U1[:, 0:1]
        v1 = U1[:, 1:2]
        u1_t = tf.gradients(u1, t)[0]
        u1_x = tf.gradients(u1, x)[0]
        u1_xx = tf.gradients(u1_x, x)[0]
        u1_xy = tf.gradients(u1_x, y)[0]
        u1_y = tf.gradients(u1, y)[0]
        u1_yy = tf.gradients(u1_y, y)[0]
        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_xy = tf.gradients(v1_x, y)[0]
        v1_xx = tf.gradients(v1_x, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v1_yy = tf.gradients(v1_y, y)[0]

        u2 = U2[:, 0:1]
        v2 = U2[:, 1:2]
        u2_t = tf.gradients(u2, t)[0]
        u2_x = tf.gradients(u2, x)[0]
        u2_xx = tf.gradients(u2_x, x)[0]
        u2_xy = tf.gradients(u2_x, y)[0]
        u2_y = tf.gradients(u2, y)[0]
        u2_yy = tf.gradients(u2_y, y)[0]
        v2_t = tf.gradients(v2, t)[0]
        v2_x = tf.gradients(v2, x)[0]
        v2_xx = tf.gradients(v2_x, x)[0]
        v2_xy = tf.gradients(v2_x, y)[0]
        v2_y = tf.gradients(v2, y)[0]
        v2_yy = tf.gradients(v2_y, y)[0]

        # 3 ICs
        u_t = tf.concat((u0_t, u1_t, u2_t), 0)
        v_t = tf.concat((v0_t, v1_t, v2_t), 0)
        
        u_x = tf.concat((u0_x, u1_x, u2_x), 0)
        u_xx = tf.concat((u0_xx, u1_xx, u2_xx), 0)
        u_y = tf.concat((u0_y, u1_y, u2_y), 0)
        u_yy = tf.concat((u0_yy, u1_yy, u2_yy), 0)
        u_xy = tf.concat((u0_xy, u1_xy, u2_xy), 0)
        
        v_x = tf.concat((v0_x, v1_x, v2_x), 0)
        v_xx = tf.concat((v0_xx, v1_xx, v2_xx), 0)
        v_xy = tf.concat((v0_xy, v1_xy, v2_xy), 0)
        v_y = tf.concat((v0_y, v1_y, v2_y), 0)
        v_yy = tf.concat((v0_yy, v1_yy, v2_yy), 0)

        data = [tf.concat((u0, u1, u2), 0), tf.concat((v0, v1, v2), 0)]
        
        # build a polynomial&derivative library
        derivatives = [tf.ones_like(data[0], optimize = False, dtype = tf.float32)]
        derivatives.append(u_x)
        derivatives.append(u_y)
        derivatives.append(u_xy)
        derivatives.append(v_x)
        derivatives.append(v_y)
        derivatives.append(v_xy)

        derivatives_description = ['', 'u_{x}', 'u_{y}', 'u_{xy}','v_{x}', 'v_{y}','v_{xy}']
        
        Phi, self.lib_descr = self.build_PolyDeriLibrary(data, derivatives, derivatives_description, PolyOrder = 3, 
                                                data_description = ['u','v'])      
        
        f_u = tf.matmul(Phi, self.lambda_u) + self.diff_coeff_u*(u_xx + u_yy) - u_t      
        f_v = tf.matmul(Phi, self.lambda_v) + self.diff_coeff_v*(v_xx + v_yy) - v_t
                
        return f_u, f_v, Phi, u_t, v_t, u_xx, u_yy, v_xx, v_yy

    def physics_residue_pt(self, x, y, t):
        # for post-training
        U0 = self.predict_response(tf.concat((x, y, t), 1), 0)
        U1 = self.predict_response(tf.concat((x, y, t), 1), 1)
        U2 = self.predict_response(tf.concat((x, y, t), 1), 2)

        u0 = U0[:, 0:1]
        v0 = U0[:, 1:2]
        u0_t = tf.gradients(u0, t)[0]
        u0_x = tf.gradients(u0, x)[0]
        u0_xx = tf.gradients(u0_x, x)[0]
        u0_xy = tf.gradients(u0_x, y)[0]
        u0_y = tf.gradients(u0, y)[0]
        u0_yy = tf.gradients(u0_y, y)[0]
        v0_t = tf.gradients(v0, t)[0]
        v0_x = tf.gradients(v0, x)[0]
        v0_xx = tf.gradients(v0_x, x)[0]
        v0_xy = tf.gradients(v0_x, y)[0]
        v0_y = tf.gradients(v0, y)[0]
        v0_yy = tf.gradients(v0_y, y)[0]

        u1 = U1[:, 0:1]
        v1 = U1[:, 1:2]
        u1_t = tf.gradients(u1, t)[0]
        u1_x = tf.gradients(u1, x)[0]
        u1_xx = tf.gradients(u1_x, x)[0]
        u1_xy = tf.gradients(u1_x, y)[0]
        u1_y = tf.gradients(u1, y)[0]
        u1_yy = tf.gradients(u1_y, y)[0]
        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_xy = tf.gradients(v1_x, y)[0]
        v1_xx = tf.gradients(v1_x, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v1_yy = tf.gradients(v1_y, y)[0]

        u2 = U2[:, 0:1]
        v2 = U2[:, 1:2]
        u2_t = tf.gradients(u2, t)[0]
        u2_x = tf.gradients(u2, x)[0]
        u2_xx = tf.gradients(u2_x, x)[0]
        u2_xy = tf.gradients(u2_x, y)[0]
        u2_y = tf.gradients(u2, y)[0]
        u2_yy = tf.gradients(u2_y, y)[0]
        v2_t = tf.gradients(v2, t)[0]
        v2_x = tf.gradients(v2, x)[0]
        v2_xx = tf.gradients(v2_x, x)[0]
        v2_xy = tf.gradients(v2_x, y)[0]
        v2_y = tf.gradients(v2, y)[0]
        v2_yy = tf.gradients(v2_y, y)[0]

        # 3 ICs
        u_t = tf.concat((u0_t, u1_t, u2_t), 0)
        v_t = tf.concat((v0_t, v1_t, v2_t), 0)
        
        u_x = tf.concat((u0_x, u1_x, u2_x), 0)
        u_xx = tf.concat((u0_xx, u1_xx, u2_xx), 0)
        u_y = tf.concat((u0_y, u1_y, u2_y), 0)
        u_yy = tf.concat((u0_yy, u1_yy, u2_yy), 0)
        u_xy = tf.concat((u0_xy, u1_xy, u2_xy), 0)
        
        v_x = tf.concat((v0_x, v1_x, v2_x), 0)
        v_xx = tf.concat((v0_xx, v1_xx, v2_xx), 0)
        v_xy = tf.concat((v0_xy, v1_xy, v2_xy), 0)
        v_y = tf.concat((v0_y, v1_y, v2_y), 0)
        v_yy = tf.concat((v0_yy, v1_yy, v2_yy), 0)

        data = [tf.concat((u0, u1, u2), 0), tf.concat((v0, v1, v2), 0)]
        
        # build a polynomial&derivative library
        derivatives = [tf.ones_like(data[0], optimize = False, dtype = tf.float32)]
        derivatives.append(u_x)
        derivatives.append(u_y)
        derivatives.append(u_xy)
        derivatives.append(v_x)
        derivatives.append(v_y)
        derivatives.append(v_xy)

        derivatives_description = ['', 'u_{x}', 'u_{y}', 'u_{xy}','v_{x}', 'v_{y}','v_{xy}']
        
        Phi, self.lib_descr = self.build_PolyDeriLibrary(data, derivatives, derivatives_description, PolyOrder = 3, 
                                                data_description = ['u','v'])
        f_u = tf.matmul(Phi, self.lambda_u*self.nonzero_mask_lambda_u_tf) + self.diff_coeff_u*(u_xx + u_yy) - u_t      
        f_v = tf.matmul(Phi, self.lambda_v*self.nonzero_mask_lambda_v_tf) + self.diff_coeff_v*(v_xx + v_yy) - v_t
                                            
        return f_u, f_v, u_xx, u_yy, v_xx, v_yy
            
    def build_PolyDeriLibrary(self, data, lib_deri, lib_deri_descr, PolyOrder, data_description = None):         
        ## polynomial terms
        P = PolyOrder
        lib_poly = [tf.ones_like(data[0], optimize = False, dtype = tf.float32)]
        lib_poly_descr = [''] # it denotes '1'
        for i in range(len(data)): # polynomial terms of univariable
            for j in range(1, P+1):
                lib_poly.append((data[i])**j)
                lib_poly_descr.append(data_description[i]+"**"+str(j))
                
        for i in range(1,P): # polynomial terms of bivariable. Assume we only have 2 variables.
            for j in range(1,P-i+1):
                lib_poly.append((data[0])**i*(data[1])**j)
                lib_poly_descr.append(data_description[0]+"**"+str(i)+data_description[1]+"**"+str(j))
                
        ## derivative terms            
        ## Multiplication of derivatives and polynomials (including the multiplication with '1')
        lib_poly_deri = []
        lib_poly_deri_descr = []
        for i in range(len(lib_poly)):
            for j in range(len(lib_deri)):
                lib_poly_deri.append(lib_poly[i]*lib_deri[j])
                lib_poly_deri_descr.append(lib_poly_descr[i]+lib_deri_descr[j])

        return tf.concat(lib_poly_deri, axis = 1), lib_poly_deri_descr
    
    def train(self, X, U, X_f, X_l, X_r, X_u, X_b, X_bc_meas, U_bc_meas):        
        self.tf_dict = {self.X_tf: X,  self.U_tf: U,
                        self.x_f_tf: X_f[:, 0:1], self.y_f_tf: X_f[:, 1:2], self.t_f_tf: X_f[:, 2:3],
                        self.x_l_tf: X_l[:, :1], self.x_r_tf: X_r[:, :1], self.x_u_tf: X_u[:, :1], self.x_b_tf: X_b[:, :1],
                        self.y_l_tf: X_l[:, 1:2], self.y_r_tf: X_r[:, 1:2], self.y_u_tf: X_u[:, 1:2], self.y_b_tf: X_b[:, 1:2],
                        self.t_l_tf: X_l[:, 2:3], self.t_r_tf: X_r[:, 2:3], self.t_u_tf: X_u[:, 2:3], self.t_b_tf: X_b[:, 2:3],
                        self.X_bc_meas_tf: X_bc_meas, self.U_bc_meas_tf: U_bc_meas}    
        
        # adaptively determine loss coefficients
        self.tf_dict[self.loss_u_coeff] = 1
        self.tf_dict[self.loss_v_coeff] = 1
        self.tf_dict[self.loss_f_v_coeff] = 1
        self.anneal_lam = [1, 1, 1]
        self.anneal_alpha = 0.8
            
        print('Pre ADO')
        for it_ADO_Pre in range(self.pre_ADO_iterations): 
            print('Adam pretraining begins')
            for it_Adam in range(self.Adam_epochs_Pre):
                self.sess.run(self.train_op_Adam_Pretrain, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, lambda_u, lambda_v, loss_bc_all, diff_coeff_u, diff_coeff_v = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_lambda_u, self.loss_lambda_v, self.lambda_u, self.lambda_v, self.loss_bc_all, self.diff_coeff_u, self.diff_coeff_v], self.tf_dict)                   
                    self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
                    self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
                    self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
                    self.loss_f_v_history_Pretrain = np.append(self.loss_f_v_history_Pretrain, loss_f_v)
                    self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
                    self.loss_lambda_v_history_Pretrain = np.append(self.loss_lambda_v_history_Pretrain, loss_lambda_v)
                    self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
                    self.lambda_v_history_Pretrain = np.append(self.lambda_v_history_Pretrain, lambda_v, axis = 1)
                    self.loss_bc_all_history_Pretrain = np.append(self.loss_bc_all_history_Pretrain, loss_bc_all)
                    self.diff_coeff_u_history_Pretrain = np.append(self.diff_coeff_u_history_Pretrain, diff_coeff_u)
                    self.diff_coeff_v_history_Pretrain = np.append(self.diff_coeff_v_history_Pretrain, diff_coeff_v)
                    print("Adam epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_lambda_u = %10.3e, loss_lambda_v = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, loss_bc_all))
            
            print('L-BFGS-B pretraining begins')
            self.optimizer_BFGS_Pretrain.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_lambda_u, self.loss_lambda_v, self.lambda_u, self.lambda_v, self.loss_bc_all, self.diff_coeff_u, self.diff_coeff_v],
                                    loss_callback = self.callback_Pretrain)
            
            # adaptively determine loss coefficients
            loss_u, loss_v, loss_f_u, loss_f_v = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v], self.tf_dict)
            self.anneal_lam[0] = (1 - self.anneal_alpha)*self.anneal_lam[0] + self.anneal_alpha*loss_u/loss_f_u
            self.anneal_lam[1] = (1 - self.anneal_alpha)*self.anneal_lam[1] + self.anneal_alpha*loss_v/loss_f_u
            self.anneal_lam[2] = (1 - self.anneal_alpha)*self.anneal_lam[2] + self.anneal_alpha*loss_f_v/loss_f_u
            self.tf_dict[self.loss_u_coeff] = self.anneal_lam[0]
            self.tf_dict[self.loss_v_coeff] = self.anneal_lam[1]
            self.tf_dict[self.loss_f_v_coeff] = self.anneal_lam[2]
                
        self.tol_best_ADO_u = 0
        self.tol_best_ADO_v = 0
        
        print('ADO begins')
        for it in tqdm(range(self.ADO_iterations)):
            print('STRidge begins')
            self.callTrainSTRidge()

            print('Adam begins')
            for it_Adam in range(self.Adam_epochs_ADO):
                self.sess.run(self.train_op_Adam_ADO, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_bc_all], self.tf_dict)                   
                    self.loss_u_history = np.append(self.loss_u_history, loss_u)
                    self.loss_v_history = np.append(self.loss_v_history, loss_v)
                    self.loss_f_u_history = np.append(self.loss_f_u_history, loss_f_u)
                    self.loss_f_v_history = np.append(self.loss_f_v_history, loss_f_v)
                    self.loss_bc_all_history = np.append(self.loss_bc_all_history, loss_bc_all)
                    print("Adam epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all))                

            print('L-BFGS-B begins')
            self.optimizer_BFGS_ADO.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v, self.loss_bc_all],
                                    loss_callback = self.callback_ADO)

        
    def callback_Pretrain(self, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, lambda_u, lambda_v, loss_bc_all, diff_coeff_u, diff_coeff_v):
        self.step_Pretrain += 1
        if self.step_Pretrain % 10 == 0:                        
            self.loss_u_history_Pretrain = np.append(self.loss_u_history_Pretrain, loss_u)
            self.loss_v_history_Pretrain = np.append(self.loss_v_history_Pretrain, loss_v)
            self.loss_f_u_history_Pretrain = np.append(self.loss_f_u_history_Pretrain, loss_f_u)
            self.loss_f_v_history_Pretrain = np.append(self.loss_f_v_history_Pretrain, loss_f_v)
            self.loss_lambda_u_history_Pretrain = np.append(self.loss_lambda_u_history_Pretrain, loss_lambda_u)
            self.loss_lambda_v_history_Pretrain = np.append(self.loss_lambda_v_history_Pretrain, loss_lambda_v)
            self.lambda_u_history_Pretrain = np.append(self.lambda_u_history_Pretrain, lambda_u, axis = 1)
            self.lambda_v_history_Pretrain = np.append(self.lambda_v_history_Pretrain, lambda_v, axis = 1)
            self.loss_bc_all_history_Pretrain = np.append(self.loss_bc_all_history_Pretrain, loss_bc_all)
            self.diff_coeff_u_history_Pretrain = np.append(self.diff_coeff_u_history_Pretrain, diff_coeff_u)
            self.diff_coeff_v_history_Pretrain = np.append(self.diff_coeff_v_history_Pretrain, diff_coeff_v)
            print("BFGS epoch(Pretrain) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_lambda_u = %10.3e, loss_lambda_v = %10.3e, loss_bc_all = %10.3e" % (self.step_Pretrain, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v, loss_bc_all))
            
    def callback_ADO(self, loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all):
        self.step_ADO = self.step_ADO + 1
        if self.step_ADO%10 == 0:                        
            self.loss_u_history = np.append(self.loss_u_history, loss_u)
            self.loss_v_history = np.append(self.loss_v_history, loss_v)
            self.loss_f_u_history = np.append(self.loss_f_u_history, loss_f_u)
            self.loss_f_v_history = np.append(self.loss_f_v_history, loss_f_v)
            self.loss_bc_all_history = np.append(self.loss_bc_all_history, loss_bc_all)
            print("BFGS epoch(ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_bc_all = %10.3e" % (self.step_ADO, loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all))                

    def callTrainSTRidge(self):
        d_tol = 1
        maxit = 50
        l0_penalty = 5e-4
        
        Phi_pred, u_t, v_t, u_xx, u_yy, v_xx, v_yy = self.sess.run([self.Phi_pred, self.u_t, self.v_t, self.u_xx_pred, self.u_yy_pred, self.v_xx_pred, self.v_yy_pred], self.tf_dict)
        Phi_aug = np.concatenate([u_xx + u_yy, v_xx + v_yy, Phi_pred], 1) # augment Phi_pred w/ diffusion candidates
        
        # lambda_u
        lambda_u_aug = self.TrainSTRidge(Phi_aug, u_t, d_tol, maxit, uv_flag = True, l0_penalty = l0_penalty) 
        self.lambda_u = tf.assign(self.lambda_u, tf.convert_to_tensor(lambda_u_aug[2:], dtype = tf.float32))
        diff_coeff_u_core_new = -np.log(self.diff_u_scale/lambda_u_aug[0,0] - 1)
        self.diff_coeff_u_core = tf.assign(self.diff_coeff_u_core, tf.convert_to_tensor(diff_coeff_u_core_new, dtype = tf.float32))

        # lambda_v
        lambda_v_aug = self.TrainSTRidge(Phi_aug, v_t, d_tol, maxit, uv_flag = False, l0_penalty = l0_penalty) 
        self.lambda_v = tf.assign(self.lambda_v, tf.convert_to_tensor(lambda_v_aug[2:], dtype = tf.float32))
        diff_coeff_v_core_new = -np.log(self.diff_v_scale/lambda_v_aug[1,0] - 1)
        self.diff_coeff_v_core = tf.assign(self.diff_coeff_v_core, tf.convert_to_tensor(diff_coeff_v_core_new, dtype = tf.float32))

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
                    
        # inherit augmented Lambda
        diff_u, diff_v = self.sess.run([self.diff_coeff_u, self.diff_coeff_v])
        diff_u = np.reshape(diff_u, (1, 1))
        diff_v = np.reshape(diff_v, (1, 1))
        if uv_flag:
            lambda_u = self.sess.run(self.lambda_u)
            lambda_best = np.concatenate([diff_u, diff_v, lambda_u], axis = 0)
        else:
            lambda_v = self.sess.run(self.lambda_v)
            lambda_best = np.concatenate([diff_u, diff_v, lambda_v], axis = 0)
        
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
            lambda1 = self.STRidge(Phi, ut, STR_iters, tol, uv_flag = uv_flag)
            err_f = np.mean((ut - Phi.dot(lambda_best))**2)
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
                
        # Inherit augmented lambda from previous training and normalize it.
        diff_u, diff_v = self.sess.run([self.diff_coeff_u, self.diff_coeff_v])
        diff_u = np.reshape(diff_u, (1, 1))
        diff_v = np.reshape(diff_v, (1, 1))

        if uv_flag:
            lambda_u = self.sess.run(self.lambda_u)
            lambda1_normalized = np.concatenate([diff_u, diff_v, lambda_u], axis = 0)/Mreg
            self.ConstScaleFac = 1e-5
        else:
            lambda_v = self.sess.run(self.lambda_v)
            lambda1_normalized = np.concatenate([diff_u, diff_v, lambda_v], axis = 0)/Mreg
            self.ConstScaleFac = 5e-4
        
        # downscale the constant term in the normalized library
        Phi_normalized[:, 2] = np.ones_like(Phi_normalized[:, 2])*self.ConstScaleFac
        Mreg[2] = self.ConstScaleFac
            
        # find big coefficients
        biginds = np.where(abs(lambda1_normalized[2:]) > tol)[0] + 2 # keep diff_u term unpruned
        if uv_flag:
            biginds = np.insert(biginds, obj = 0, values = 0) # keep diffu u term unpruned
        else:
            biginds = np.insert(biginds, obj = 0, values = 1) # keep diffu v term unpruned
            
        num_relevant = d            
        
        # record lambda evolution
        ridge_append_counter = 0
        ridge_append_counter = self.record_lambda_in_STRidge(uv_flag, Mreg, lambda1_normalized, ridge_append_counter, end_flag = False)

        # Threshold small coefficients until convergence
        for j in range(STR_iters):  
            # Figure out which items to cut out
            smallinds = np.where(abs(lambda1_normalized[2:]) < tol)[0] + 2 # don't threhold diffu terms 
            if uv_flag:
                smallinds = np.insert(smallinds, obj = 0, values = 1) # prune diffu v term
            else:
                smallinds = np.insert(smallinds, obj = 0, values = 0) # prune diffu u term
                
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
            lambda1_normalized[biginds] = np.linalg.lstsq(Phi_normalized[:, biginds].T.dot(Phi_normalized[:, biginds]) + 1e-6*np.eye(len(biginds)),Phi_normalized[:, biginds].T.dot(ut))[0]
            
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
        
        fig = plt.figure()
        plt.plot(self.diff_coeff_v_history_Pretrain[1:])
        plt.xlabel('10x')
        plt.title('diff_coeff_v_history_Pretrain')  
        plt.savefig('6.png')        

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
        plt.savefig('7.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history[1:])
        plt.plot(self.loss_f_v_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f history(ADO)')  
        plt.savefig('8.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_bc_all_history[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_bc_all_history of ADO')  
        plt.savefig('9.png')
                                    
# =============================================================================
#            plot loss histories in STRidge                   
# =============================================================================
        fig = plt.figure()
        plt.plot(self.loss_f_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f_u history of STRidge')  
        plt.savefig('10.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_u_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda_u history of STRidge')
        plt.savefig('11.png')
        
        fig = plt.figure()
        plt.plot(self.tol_u_history_STRidge[1:])
        plt.title('Tolerance_u History of STRidge')
        plt.savefig('12.png')

        fig = plt.figure()
        plt.plot(self.loss_f_v_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_f_v history of STRidge')  
        plt.savefig('13.png')
        
        fig = plt.figure()
        plt.plot(self.loss_lambda_v_history_STRidge[1:])
        plt.yscale('log')       
        plt.title('loss_lambda_v history of STRidge')
        plt.savefig('14.png')
        
        fig = plt.figure()
        plt.plot(self.tol_v_history_STRidge[1:])
        plt.title('Tolerance_v History of STRidge')
        plt.savefig('15.png')
                        
    def visualize_post_training(self):
        fig = plt.figure()
        plt.plot(self.loss_u_history_pt[1:])
        plt.plot(self.loss_v_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_u', 'loss_v'))
        plt.title('loss_u, loss_v history(post-training)')  
        plt.savefig('16.png')

        fig = plt.figure()
        plt.plot(self.loss_f_u_history_pt[1:])
        plt.plot(self.loss_f_v_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.legend(('loss_f_u', 'loss_f_v'))
        plt.title('loss_f history(post-training)')  
        plt.savefig('17.png')
                                
        fig = plt.figure()
        plt.plot(self.loss_bc_all_history_pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_bc_all_history_pt')  
        plt.savefig('18.png')  
        
    def inference(self, X_star):        
        tf_dict = {self.X_tf: X_star}            
        U0 = self.sess.run(self.U0_pred, tf_dict)
        U1 = self.sess.run(self.U1_pred, tf_dict)
        U2 = self.sess.run(self.U2_pred, tf_dict)
        return np.stack((U0, U1, U2), axis = -1) 
    
    def post_train(self, pt_ADO_iterations, Adam_epochs_Pt):
                
        # loss histories for post-training
        self.loss_u_history_pt = np.array([0])
        self.loss_v_history_pt = np.array([0])
        self.loss_f_u_history_pt = np.array([0])
        self.loss_f_v_history_pt = np.array([0])
        self.lambda_u_history_pt = np.zeros([72, 1])
        self.lambda_v_history_pt = np.zeros([72, 1])
        self.loss_bc_all_history_pt = np.array([0])
        
        print('post-training begins')
        for it in tqdm(range(pt_ADO_iterations)):
# =============================================================================
#              update library coefficients lambda_u and lambda_v via least squares
# =============================================================================
            print('least squares begins')
            # find non-zero values in library coefficients
            lambda_u, lambda_v = self.sess.run([self.lambda_u, self.lambda_v])
            diff_u, diff_v = self.sess.run([self.diff_coeff_u, self.diff_coeff_v])
            diff_u = np.reshape(diff_u, (1, 1))
            diff_v = np.reshape(diff_v, (1, 1))
            lambda_u_aug = np.concatenate([diff_u, diff_v, lambda_u], axis = 0)
            lambda_v_aug = np.concatenate([diff_u, diff_v, lambda_v], axis = 0)
            nonzero_ind_u_aug = np.nonzero(lambda_u_aug)            
            nonzero_ind_v_aug = np.nonzero(lambda_v_aug)

            # form compact libraries Phi_u_compact & Phi_v_compact that only have non-zero candidates
            Phi_pred, u_t, v_t, u_xx, u_yy, v_xx, v_yy = self.sess.run([self.Phi_pred, self.u_t, self.v_t, self.u_xx_pred, self.u_yy_pred, self.v_xx_pred, self.v_yy_pred], self.tf_dict)
            Phi_aug = np.concatenate([u_xx + u_yy, v_xx + v_yy, Phi_pred], 1) # augment Phi_pred w/ diffusion candidates            
            Phi_u_compact = Phi_aug[:, nonzero_ind_u_aug[0]] 
            Phi_v_compact = Phi_aug[:, nonzero_ind_v_aug[0]] 

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
                                    
            # assign updated values to self.lambda_u and self.diff_coeff_u_core
            lambda_u_aug_updated = np.zeros_like(lambda_u_aug)
            lambda_u_aug_updated[nonzero_ind_u_aug] = np.reshape(lambda_u_updated_compact, (-1, 1))            
            self.lambda_u = tf.assign(self.lambda_u, tf.convert_to_tensor(lambda_u_aug_updated[2:, :], dtype = tf.float32))
            diff_coeff_u_core_new = -np.log(self.diff_u_scale/lambda_u_aug_updated[0,0] - 1)
            self.diff_coeff_u_core = tf.assign(self.diff_coeff_u_core, tf.convert_to_tensor(diff_coeff_u_core_new, dtype = tf.float32))

            
            # assign updated values to self.lambda_v and self.diff_coeff_v_core
            lambda_v_aug_updated = np.zeros_like(lambda_v_aug)
            lambda_v_aug_updated[nonzero_ind_v_aug] = np.reshape(lambda_v_updated_compact, (-1, 1))
            self.lambda_v = tf.assign(self.lambda_v, tf.convert_to_tensor(lambda_v_aug_updated[2:, :], dtype = tf.float32))
            diff_coeff_v_core_new = -np.log(self.diff_v_scale/lambda_v_aug_updated[1,0] - 1)
            self.diff_coeff_v_core = tf.assign(self.diff_coeff_v_core, tf.convert_to_tensor(diff_coeff_v_core_new, dtype = tf.float32))
            
            self.lambda_u_history_pt = np.append(self.lambda_u_history_pt, lambda_u_aug_updated, axis = 1)
            self.lambda_v_history_pt = np.append(self.lambda_v_history_pt, lambda_v_aug_updated, axis = 1)
# =============================================================================
#              update NN weights and bias via Adam
# =============================================================================
            # mark non-zero candidates in the library 
            nonzero_mask_lambda_u = np.zeros_like(lambda_u)
            nonzero_ind_u = nonzero_ind_u_aug[0][1:] - 2
            nonzero_mask_lambda_u[nonzero_ind_u, nonzero_ind_u_aug[1]] = 1
            self.tf_dict[self.nonzero_mask_lambda_u_tf] = nonzero_mask_lambda_u
            
            nonzero_mask_lambda_v = np.zeros_like(lambda_v)
            nonzero_ind_v = nonzero_ind_v_aug[0][1:] - 2
            nonzero_mask_lambda_v[nonzero_ind_v, nonzero_ind_v_aug[1]] = 1
            self.tf_dict[self.nonzero_mask_lambda_v_tf] = nonzero_mask_lambda_v

            print('Adam begins')
            for it_Adam in range(Adam_epochs_Pt):
                self.sess.run(self.train_op_Adam_pt, self.tf_dict)
                if it_Adam % 10 == 0:
                    loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all = self.sess.run([self.loss_u, self.loss_v, self.loss_f_u_pt, self.loss_f_v_pt, self.loss_bc_all], self.tf_dict)
                    self.loss_u_history_pt = np.append(self.loss_u_history_pt, loss_u)
                    self.loss_v_history_pt = np.append(self.loss_v_history_pt, loss_v)
                    self.loss_f_u_history_pt = np.append(self.loss_f_u_history_pt, loss_f_u)
                    self.loss_f_v_history_pt = np.append(self.loss_f_v_history_pt, loss_f_v)
                    self.loss_bc_all_history_pt = np.append(self.loss_bc_all_history_pt, loss_bc_all)
                    print("Adam epoch(Pt-ADO) %s : loss_u = %10.3e, loss_v = %10.3e, loss_f_u = %10.3e, loss_f_v = %10.3e, loss_bc_all = %10.3e" % (it_Adam, loss_u, loss_v, loss_f_u, loss_f_v, loss_bc_all))                

# =============================================================================
#       determine whether the post-training is sufficient
# =============================================================================
        self.visualize_post_training()

def DownsampleMeas(idx_x, idx_t, xx, yy, tt, Exact_u_IC1, Exact_v_IC1, Exact_u_IC2 = 0, Exact_v_IC2 = 0, Exact_u_IC3 = 0, Exact_v_IC3 = 0, XU_flag = False):
    xx0 = xx[:, :, idx_t]
    xx1 = xx0[idx_x, :, :]
    xx2 = xx1[:, idx_x, :]
    
    yy0 = yy[:, :, idx_t]
    yy1 = yy0[idx_x, :, :]
    yy2 = yy1[:, idx_x, :]
    
    tt0 = tt[:, :, idx_t]
    tt1 = tt0[idx_x, :, :]
    tt2 = tt1[:, idx_x, :]    
    
    X_U_meas = np.hstack((xx2.flatten()[:,None], yy2.flatten()[:,None], tt2.flatten()[:,None]))  

    # IC1
    Exact_u_IC1_0 = Exact_u_IC1[:, :, idx_t]
    Exact_u_IC1_1 = Exact_u_IC1_0[idx_x, :, :]
    Exact_u_IC1_2 = Exact_u_IC1_1[:, idx_x, :]
    Exact_v_IC1_0 = Exact_v_IC1[:, :, idx_t]
    Exact_v_IC1_1 = Exact_v_IC1_0[idx_x, :, :]
    Exact_v_IC1_2 = Exact_v_IC1_1[:, idx_x, :]
    
    U_meas_IC1 = np.hstack((Exact_u_IC1_2.flatten()[:,None], Exact_v_IC1_2.flatten()[:,None]))  
    
    # IC2
    if type(Exact_u_IC2) != int:
        Exact_u_IC2_0 = Exact_u_IC2[:, :, idx_t]
        Exact_u_IC2_1 = Exact_u_IC2_0[idx_x, :, :]
        Exact_u_IC2_2 = Exact_u_IC2_1[:, idx_x, :]
        Exact_v_IC2_0 = Exact_v_IC2[:, :, idx_t]
        Exact_v_IC2_1 = Exact_v_IC2_0[idx_x, :, :]
        Exact_v_IC2_2 = Exact_v_IC2_1[:, idx_x, :]
        
        U_meas_IC2 = np.hstack((Exact_u_IC2_2.flatten()[:,None], Exact_v_IC2_2.flatten()[:,None]))  

    # IC3
    if type(Exact_u_IC3) != int:
        Exact_u_IC3_0 = Exact_u_IC3[:, :, idx_t]
        Exact_u_IC3_1 = Exact_u_IC3_0[idx_x, :, :]
        Exact_u_IC3_2 = Exact_u_IC3_1[:, idx_x, :]
        Exact_v_IC3_0 = Exact_v_IC3[:, :, idx_t]
        Exact_v_IC3_1 = Exact_v_IC3_0[idx_x, :, :]
        Exact_v_IC3_2 = Exact_v_IC3_1[:, idx_x, :]
    
        U_meas_IC3 = np.hstack((Exact_u_IC3_2.flatten()[:,None], Exact_v_IC3_2.flatten()[:,None]))  

    if type(Exact_u_IC2) != int and type(Exact_u_IC3) != int:
        U_meas = np.stack((U_meas_IC1, U_meas_IC2, U_meas_IC3), axis = -1)
    elif type(Exact_u_IC2) != int and type(Exact_u_IC3) == int:
        U_meas = np.stack((U_meas_IC1, U_meas_IC2), axis = -1)
    elif type(Exact_u_IC2) == int and type(Exact_u_IC3) == int:
        U_meas = U_meas_IC1

    if XU_flag:
        return X_U_meas, U_meas
    elif type(Exact_u_IC2) != int and type(Exact_u_IC3) != int:
        return xx2, yy2, tt2, Exact_u_IC1_2, Exact_v_IC1_2, Exact_u_IC2_2, Exact_v_IC2_2, Exact_u_IC3_2, Exact_v_IC3_2
    elif type(Exact_u_IC2) != int and type(Exact_u_IC3) == int:
        return xx2, yy2, tt2, Exact_u_IC1_2, Exact_v_IC1_2, Exact_u_IC2_2, Exact_v_IC2_2
    elif type(Exact_u_IC2) == int and type(Exact_u_IC3) == int:
        return xx2, yy2, tt2, Exact_u_IC1_2, Exact_v_IC1_2                    
    
