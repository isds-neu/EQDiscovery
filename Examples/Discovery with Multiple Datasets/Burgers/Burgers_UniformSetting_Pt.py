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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # limite all operations on this GPU
with tf.device('/device:GPU:0'): # '/device:GPU:0' is GPU1 in task manager. Likewise, '/device:GPU:1' is GPU0.
                    
    # Loss histories for pretraining
    loss_u_history_Pt = np.array([0])
    loss_f_history_Pt = np.array([0])
    loss_u_val_history_Pt = np.array([0])
    step_Pt = 0
    lambda_history_Pt = np.zeros((2, 1)) 
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, X_val, u_val, X_f, layers_s, layers_i, lb, ub, Lamu_init):
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
            self.lambda2 = tf.Variable(Lamu_init, dtype=tf.float64, name = 'lambda')
            
            # Specify the list of trainable variables            
            self.var_list_Pt = self.biases0 + self.weights0 + self.biases1 + self.weights1 + self.biases2 + \
                self.weights2 + self.weights_s + self.biases_s + [self.lambda2]
                
            self.var_list_1 = self.biases0 + self.weights0 + self.biases1 + self.weights1 + self.biases2 + \
                self.weights2 + self.weights_s + self.biases_s            
                
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
			
            self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
            self.loss_u = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred)) + \
                    tf.reduce_mean(tf.square(self.u2_tf - self.u2_pred))
            self.loss_f = tf.reduce_mean(tf.square(self.f_pred))
                                    
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
            self.optimizer_Pt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = self.var_list_Pt,
                                                                    method = 'L-BFGS-B', 
                                                                   options = {'maxiter': 10000, #10000
                                                                               'maxfun': 10000, #10000
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol' : 1.0 * np.finfo(float).eps})
                             
            # self.global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = 1e-3
            # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.5,
            #                                             staircase=True)

            # The default settings: learning rate = 1e-3, beta1 = 0.9, beta2 = 0.999， epsilon = 1e-8
            # self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate) 
            # self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = self.var_list_1,
            #                                                    global_step = self.global_step)

            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = 1e-3) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = self.var_list_Pt)

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
                       
        def net_f(self, x, t):
            u0 = self.net_u(x, t, 0)
            u1 = self.net_u(x, t, 1)
            u2 = self.net_u(x, t, 2)
            u0_t = tf.gradients(u0, t)[0]
            u0_x = tf.gradients(u0, x)[0]
            u0_xx = tf.gradients(u0_x, x)[0]
 
            u1_t = tf.gradients(u1, t)[0]
            u1_x = tf.gradients(u1, x)[0]
            u1_xx = tf.gradients(u1_x, x)[0]

            u2_t = tf.gradients(u2, t)[0]
            u2_x = tf.gradients(u2, x)[0]
            u2_xx = tf.gradients(u2_x, x)[0]

            u = tf.concat((u0, u1, u2), 0)
            u_t = tf.concat((u0_t, u1_t, u2_t), 0)
            u_x = tf.concat((u0_x, u1_x, u2_x), 0)
            u_xx = tf.concat((u0_xx, u1_xx, u2_xx), 0)
            Phi = tf.concat([u*u_x, u_xx], 1)            
            self.lib_descr = ['u*u_x', 'u_xx']
            f = tf.matmul(Phi, self.lambda2) - u_t      
            return f
        
        def callback_Pt(self, loss_u, loss_f, loss_u_val, lamu):
            global step_Pt
            if step_Pt%10 == 0:
                global loss_u_history_Pt
                global loss_f_history_Pt
                global loss_u_val_history_Pt
                global lambda_history_Pt
                
                loss_u_history_Pt = np.append(loss_u_history_Pt, loss_u)
                loss_f_history_Pt = np.append(loss_f_history_Pt, loss_f)
                loss_u_val_history_Pt = np.append(loss_u_val_history_Pt, loss_u_val)
                lambda_history_Pt = np.append(lambda_history_Pt, lamu, axis = 1)
                
            step_Pt = step_Pt + 1

        def train(self): 
            saver_ADO = tf.train.Saver(self.var_list_1)  
            saved_path = saver_ADO.restore(self.sess, './saved_variable_ADO')

            # Loop of Adam optimization
            print('Adam begins')                                            
            for it_Adam in tqdm(range(10000)): # 10000                    
                self.sess.run(self.train_op_Adam, self.tf_dict)                    
                # Print
                if it_Adam % 10 == 0:
                    loss_u, loss_f, loss_u_val = self.sess.run([self.loss_u, self.loss_f, self.loss_u_val], 
                                                               self.tf_dict)
                    lamu = self.sess.run(self.lambda2)
                    
                    global loss_u_history_Pt
                    global loss_f_history_Pt
                    global loss_u_val_history_Pt
                    global lambda_history_Pt
                    
                    loss_u_history_Pt = np.append(loss_u_history_Pt, loss_u)
                    loss_f_history_Pt = np.append(loss_f_history_Pt, loss_f)
                    loss_u_val_history_Pt = np.append(loss_u_val_history_Pt, loss_u_val)
                    lambda_history_Pt = np.append(lambda_history_Pt, lamu, axis = 1)

            print('L-BFGS-B pt begins')
            self.optimizer_Pt.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss_u, self.loss_f, self.loss_u_val, self.lambda2],
                                    loss_callback = self.callback_Pt)
                            
            saver_Pt = tf.train.Saver(self.var_list_Pt)          
            saved_path = saver_Pt.save(self.sess, './saved_variable_Pt')
        
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
            
            u0 = self.sess.run(self.u0_pred, tf_dict)
            u1 = self.sess.run(self.u1_pred, tf_dict)
            u2 = self.sess.run(self.u2_pred, tf_dict)
            return u0, u1, u2
                                            
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
        # inherit eq coeffs(non-zeros) from previous training
        eq_coeff_data = scipy.io.loadmat('DiscLam_ADO.mat')
        
        Lamu_init = eq_coeff_data['Lamu_Disc']
        Lamu_init = np.reshape(Lamu_init[np.nonzero(Lamu_init)], (-1, 1))

        model = PhysicsInformedNN(X_u_train, u_train, X_u_val, u_val, X_f_train, layers_s, layers_i, lb, ub, Lamu_init)
        model.train()
        
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
        p1 = plt.plot(loss_u_history_Pt[1:])
        p2 = plt.plot(loss_u_val_history_Pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_u history of pt')  
        plt.legend((p1[0], p2[0]), ('tr', 'val'))
        plt.savefig('1_pt.png')
        
        fig = plt.figure()
        plt.plot(loss_f_history_Pt[1:])
        plt.yscale('log')       
        plt.xlabel('10x')
        plt.title('loss_f history of pt')     
        plt.savefig('2_pt.png')

        fig = plt.figure()
        for i in range(lambda_history_Pt.shape[0]):
            plt.plot(lambda_history_Pt[i, 1:])
        plt.title('lambda_history_Pt')
        plt.savefig('3_Pt.png')

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
        lambda1_value = model.sess.run(model.lambda2)
        lambda1_true = np.zeros((2,1))
        lambda1_true[0] = -1
        lambda1_true[1] = 0.01/3.1415926
                
        lambda_error = np.linalg.norm(lambda1_true-lambda1_value,2)/np.linalg.norm(lambda1_true,2)
        f.write('Lambda L2 Error: %e' % (lambda_error))   
        
        disc_eq_temp = []
        for i_lib in range(len(model.lib_descr)):
            if lambda1_value[i_lib] != 0:
                disc_eq_temp.append(str(lambda1_value[i_lib,0]) + model.lib_descr[i_lib])
        disc_eq = '+'.join(disc_eq_temp)        
        print('The discovered equation: u_t = ' + disc_eq)
        
        f.close()
        scipy.io.savemat('DiscLam.mat', {'Lam_True': lambda1_true, 'Lam_Disc': lambda1_value,
                                         'lambda_history_Pt': lambda_history_Pt[:, 1:]})

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
        plt.savefig('4_Pt.png')
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
        plt.savefig('5_Pt.png')

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
        plt.savefig('6_Pt.png')
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
        plt.savefig('7_Pt.png')

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
        plt.savefig('8_Pt.png')
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
        plt.savefig('9_Pt.png')
        
        scipy.io.savemat('PredSol.mat', {'u_pred_sin': U_pred_sine, 'u_pred_cos': U_pred_cos,
                                 'u_pred_gauss': U_pred_Gauss})
                        
        ########################## Plots for Lambda ########################
        fig = plt.figure()
        plt.plot(lambda1_true)
        plt.plot(lambda1_value)
        plt.title('lambda values')
        plt.legend(['the true', 'the identified'])
        plt.savefig('10_Pt.png')