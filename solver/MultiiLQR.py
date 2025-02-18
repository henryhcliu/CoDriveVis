import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
from numba import njit, jit
import numba
#from loguru import logger
import os 
import time

class MultiiLQRWrapper(object):
    def __init__(self, dynamic_model, obj_fun, real_obj_fun, lb, ub):
        self.dynamic_model = dynamic_model
        self.obj_fun = obj_fun
        self.real_obj_fun = real_obj_fun
        self.n = dynamic_model.n
        self.m = dynamic_model.m
        self.T = dynamic_model.T
        self.trajectory = self.dynamic_model.eval_traj()
        self.F_matrix = self.dynamic_model.eval_grad(self.trajectory)
        self.init_obj = self.real_obj_fun.eval_obj_fun(self.trajectory)
        self.obj_fun_value_last = self.init_obj
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        self.lower_bound = lb
        self.upper_bound = ub

    def get_traj(self):
        return self.trajectory.copy()

    def update_F_matrix(self, F_matrix):
        self.F_matrix = F_matrix
    
    def _none_line_search(self):
        trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, 1)
        obj_fun_value_current = self.real_obj_fun.eval_obj_fun(trajectory_current)
        return trajectory_current, obj_fun_value_current

    def _vanilla_line_search(self, gamma, maximum_line_search):
        alpha = 1.
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        for k in range(maximum_line_search): # Line Search if the z value is greater than zero
            trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, alpha)
            obj_fun_value_current = self.real_obj_fun.eval_obj_fun(trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self.obj_fun_value_last
            alpha = alpha * gamma
            if obj_fun_value_delta<0:
                return trajectory_current, obj_fun_value_current
        return self.trajectory, self.obj_fun_value_last
    
    def _exact_line_search(self, gamma, maximum_line_search):
        alpha = 1.0
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        all_trajectory = []
        cost_list = []
        for k in range(maximum_line_search): # Line Search if the z value is greater than zero
            trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, self.lower_bound, self.upper_bound, alpha)
            obj_fun_value_current = self.real_obj_fun.eval_obj_fun(trajectory_current)
            all_trajectory.append(trajectory_current)
            cost_list.append(obj_fun_value_current)
            alpha = alpha * gamma

        cost_list = np.array(cost_list)
        index = np.argmin(cost_list)
        return all_trajectory[index], cost_list[index]
    
    def _vanilla_stopping_criterion(self, obj_fun_value_current, stopping_criterion):
        obj_fun_value_delta = obj_fun_value_current - self.obj_fun_value_last
        if (abs(obj_fun_value_delta)<stopping_criterion):
            return True
        return False

    def evaluate(self, ):
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.F_matrix = self.dynamic_model.eval_grad(self.trajectory)
    
    def forward_pass_no_eval(self, gamma=0.9, max_line_search=8, stopping_criterion = 1e-6):
        self.trajectory, obj_fun_value_current = self._exact_line_search(gamma, max_line_search)
        is_stop = self._vanilla_stopping_criterion(obj_fun_value_current, stopping_criterion)
        self.obj_fun_value_last = obj_fun_value_current
        return obj_fun_value_current, is_stop

    def forward_pass(self, gamma=0.45, max_line_search=8, stopping_criterion = 1e-6):
        #self.trajectory, obj_fun_value_current = self._none_line_search()
        #self.trajectory, obj_fun_value_current = self._vanilla_line_search(gamma, max_line_search)
        self.trajectory, obj_fun_value_current = self._exact_line_search(gamma, max_line_search)
        is_stop = self._vanilla_stopping_criterion(obj_fun_value_current, stopping_criterion)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.F_matrix = self.dynamic_model.eval_grad(self.trajectory)
        self.obj_fun_value_last = obj_fun_value_current
        return obj_fun_value_current, is_stop

    def backward_pass(self):
        self.K_matrix, self.k_vector = self.backward_pass_static(self.m, self.n, self.T, self.C_matrix, self.c_vector, self.F_matrix)
        return self.K_matrix, self.k_vector

    @staticmethod
    @njit
    def backward_pass_static(m, n, T, C_matrix, c_vector, F_matrix):
        V_matrix = np.zeros((n, n))
        v_vector = np.zeros(n)
        K_matrix_list = np.zeros((T, m, n))
        k_vector_list = np.zeros((T, m))
        for i in range(T-1,-1,-1):
            Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
            q_vector = c_vector[i] + F_matrix[i].T@v_vector
            Q_uu = Q_matrix[n:n+m,n:n+m].copy()
            Q_ux = Q_matrix[n:n+m,0:n].copy()
            q_u = q_vector[n:n+m].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu,Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu,q_u)

            V_matrix = Q_matrix[:n,:n] + Q_ux.T@K_matrix_list[i]+\
                            K_matrix_list[i].T@Q_ux+\
                            K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            
            v_vector = q_vector[:n] + Q_ux.T@k_vector_list[i] +\
                            K_matrix_list[i].T@q_u + \
                            K_matrix_list[i].T@Q_uu@k_vector_list[i]

        return K_matrix_list, k_vector_list
    
    def get_obj_fun_value(self):
        return self.obj_fun_value_last

    def clear_obj_fun_value_last(self):
        self.obj_fun_value_last = self.init_obj
    
    def solve(self, max_iter = 100, is_check_stop = True, maximum_line_search = 10):
        for i in range(max_iter):
            self.backward_pass()
            obj, isStop = self.forward_pass()
            if isStop and is_check_stop:
                break


class LQRWrapper(object):
    def __init__(self,n,m,T,init_state):
        self.n = n
        self.m = m
        self.m_n = self.n + self.m
        self.T = T
        self.C_matrix = np.zeros((T, self.m_n, self.m_n),dtype=np.float64)
        self.c_vector = np.zeros((T, self.m_n),dtype=np.float64)
        self.F_matrix = np.zeros((T, self.n, self.m_n),dtype=np.float64)
        self.init_state = init_state

    def update_C(self, C_matrix, c_vector):
        self.C_matrix = C_matrix
        self.c_vector = c_vector

    def update_F(self, F_matrix):
        self.F_matrix = F_matrix

    def backward_pass(self):
        self.K_matrix, self.k_vector = self.backward_pass_static(self.m,self.n,self.T,self.C_matrix,self.c_vector,self.F_matrix)
        return self.K_matrix, self.k_vector

    def forward_pass(self):
        self.trajectory = self.forward_pass_static(self.m,self.n,self.T,self.K_matrix,self.k_vector,self.F_matrix,self.init_state)
        return self.trajectory
    
    def get_traj(self):
        return self.trajectory.copy()

    @staticmethod
    @njit
    def forward_pass_static(m,n,T,K_matrix,k_vector,F_matrix,init_state):
        trajectory = np.zeros((T,m+n))
        trajectory[0] = init_state
        for tau in range(T-1):
            x = trajectory[tau,:n]
            u = K_matrix[tau]@x+k_vector[tau]
            x_u = np.concatenate((x,u))
            trajectory[tau,n:] = u
            trajectory[tau+1, :n] = F_matrix[tau,]@x_u
        return trajectory

    @staticmethod
    @njit
    def backward_pass_static(m,n,T,C_matrix,c_vector,F_matrix):
        V_matrix = np.zeros((n, n))
        v_vector = np.zeros(n)
        K_matrix_list = np.zeros((T, m, n))
        k_vector_list = np.zeros((T, m))
        for i in range(T-1,-1,-1):
            Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
            q_vector = c_vector[i] + F_matrix[i].T@v_vector
            Q_uu = Q_matrix[n:n+m,n:n+m].copy()
            Q_ux = Q_matrix[n:n+m,0:n].copy()
            q_u = q_vector[n:n+m].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu+1e-10*np.eye(m,dtype=np.float64),Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu+1e-10*np.eye(m,dtype=np.float64),q_u)

            V_matrix = Q_matrix[:n,:n] + Q_ux.T@K_matrix_list[i]+\
                            K_matrix_list[i].T@Q_ux+\
                            K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            
            v_vector = q_vector[:n] + Q_ux.T@k_vector_list[i] +\
                            K_matrix_list[i].T@q_u + \
                            K_matrix_list[i].T@Q_uu@k_vector_list[i]

        return K_matrix_list, k_vector_list
    
