import numpy as np
import sympy as sp
from sympy import Array, Matrix
import scipy as sci
import time as tm
from scipy import io
from numba import njit, jit
import numba
import os
from datetime import datetime
#from loguru import logger

class MultiDynamicModelWrapper(object):
    def __init__(self, dynamic_model_function, x_u_var, init_state, init_inputs, T):
        self.init_state = init_state
        self.init_inputs = init_inputs
        self.n = int(init_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = T
        self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var],dynamic_model_function,'math')) # dynamic_model_function is a sympy array, create a function to evaluate it using lambdify
        grad_dynamic_model_function = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var)) # partial derivative of dynamic_model_function w.r.t. x_u_var
        self.grad_dynamic_model_lamdify = njit(sp.lambdify([x_u_var], grad_dynamic_model_function, "math")) # create a function to evaluate it using lambdify


    # function to evaluate the trajectory by given initial state and input trajectory
    def eval_traj(self, init_state=None, input_traj=None):
        if init_state is None:
            init_state = self.init_state
        if input_traj is None:
            input_traj = self.init_inputs
        return self._eval_traj_static(self.dynamic_model_lamdify, init_state, input_traj, self.m, self.n)

    def update_traj(self, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha):
        return self._update_traj_static(self.dynamic_model_lamdify, self.m, self.n, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha)

    def eval_grad(self, traj):
        return self._eval_grad_static(self.grad_dynamic_model_lamdify, traj)

    @staticmethod
    @njit
    def _eval_traj_static(dynamic_model_lamdify, init_state, input_traj, m, n):
        T = int(input_traj.shape[0])
        trajectory = np.zeros((T,m+n))
        trajectory[0] = np.concatenate((init_state,input_traj[0]))
        for tau in range(T-1):
            trajectory[tau+1, :n] = np.asarray(dynamic_model_lamdify(trajectory[tau,:]), dtype=np.float64)
            trajectory[tau+1, n:] = input_traj[tau+1]
        return trajectory

    @staticmethod
    @njit
    def _update_traj_static(dynamic_model_lamdify, m, n, old_traj, K_matrix_all, k_vector_all, lb, ub, alpha):
        T = int(K_matrix_all.shape[0])
        new_trajectory = np.zeros((T,m+n))
        new_trajectory[0] = old_traj[0]
        for tau in range(T-1):
            delta_x = new_trajectory[tau, :n] - old_traj[tau, :n]
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            input_u = old_traj[tau, n:] + delta_u
            new_trajectory[tau, n:] = np.clip(input_u,lb,ub)
            new_trajectory[tau+1, :n] = np.asarray(dynamic_model_lamdify(new_trajectory[tau,:]),dtype=np.float64)
        return new_trajectory

    @staticmethod
    @njit
    def _eval_grad_static(grad_dynamic_model_lamdify, trajectory):
        T = int(trajectory.shape[0])
        F_matrix_initial = grad_dynamic_model_lamdify(trajectory[0,:])
        F_matrix_all = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_all[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T):
            F_matrix_all[tau] = np.asarray(grad_dynamic_model_lamdify(trajectory[tau,:]), dtype = np.float64)
        return F_matrix_all

def simple():
    x_u = Array(sp.symarray('x_u',2))
    return Array([x_u[0]+x_u[1]]), x_u

def multiVehicle(num_of_vehicle, wheel_base, h_constant = 0.1):
    x_u = Array(sp.symarray('x_u',(num_of_vehicle,6))) # x,y,theta,v,w,acc

    prod = lambda x,y:Array(Matrix(x).multiply_elementwise(Matrix(y)))[:,0] # elementwise product
    sq = lambda x:x.applyfunc(lambda y:y**2)
    sqrt = lambda x:x.applyfunc(sp.sqrt)
    cos = lambda x:x.applyfunc(sp.cos)
    sin = lambda x:x.applyfunc(sp.sin)
    asin = lambda x:x.applyfunc(sp.asin)
    def add(x,n):
        return x.applyfunc(lambda y:y+n) # y+n is faster than y+sp.Matrix([n])

    h_constant = 0.1 # Time interval: 0.1s
    d_constanT = wheel_base # vehicle wheelbase: 2.7m
    h_d_constanT = h_constant/d_constanT

    b_function=add(h_constant*prod(x_u[:,3],cos(x_u[:,4])),d_constanT)\
        - sqrt(add(-(h_constant**2)*prod(sq(x_u[:,3]),sq(sin(x_u[:,4]))),d_constanT**2))

    system = sp.transpose(Array([
            x_u[:,0] + prod(b_function,cos(x_u[:,2])),
            x_u[:,1] + prod(b_function,sin(x_u[:,2])),
            x_u[:,2] + asin(h_d_constanT*prod(x_u[:,3],sin(x_u[:,4]))),
            x_u[:,3] + h_constant*x_u[:,5],
        ])).reshape(4*num_of_vehicle) #

    x = np.array(x_u[:,0:4].reshape(4*num_of_vehicle))
    u = np.array(x_u[:,4:6].reshape(2*num_of_vehicle))
    x_u = Array(np.concatenate((x,u)))

    return system, x_u

def TeslaModel3(num_of_vehicle, h_constant = 0.1):
    x_u = Array(sp.symarray('x_u',(num_of_vehicle,6)))

    prod = lambda x,y:Array(Matrix(x).multiply_elementwise(Matrix(y)))[:,0]
    sq = lambda x:x.applyfunc(lambda y:y**2)
    sqrt = lambda x:x.applyfunc(sp.sqrt)
    cos = lambda x:x.applyfunc(sp.cos)
    sin = lambda x:x.applyfunc(sp.sin)
    asin = lambda x:x.applyfunc(sp.asin)
    def add(x,n):
        return x.applyfunc(lambda y:y+n)

    h_constant = 0.1
    d_constanT = 3.0
    h_d_constanT = h_constant/d_constanT

    b_function=add(h_constant*prod(x_u[:,3],cos(x_u[:,4])),d_constanT)\
        - sqrt(add(-(h_constant**2)*prod(sq(x_u[:,3]),sq(sin(x_u[:,4]))),d_constanT**2))

    system = sp.transpose(Array([
            x_u[:,0] + prod(b_function,cos(x_u[:,2])),
            x_u[:,1] + prod(b_function,sin(x_u[:,2])),
            x_u[:,2] + asin(h_d_constanT*prod(x_u[:,3],sin(x_u[:,4]))),
            x_u[:,3] + h_constant*x_u[:,5],
        ])).reshape(4*num_of_vehicle)

    x = np.array(x_u[:,0:4].reshape(4*num_of_vehicle))
    u = np.array(x_u[:,4:6].reshape(2*num_of_vehicle))
    x_u = Array(np.concatenate((x,u)))

    return system, x_u

def multiVehicle_Varified(num_of_vehicle, h_constant = 0.1, d_constanT = 2.70):
    x_u = Array(sp.symarray('x_u',(num_of_vehicle,6)))

    prod = lambda x,y:Array(Matrix(x).multiply_elementwise(Matrix(y)))[:,0]
    sq = lambda x:x.applyfunc(lambda y:y**2)
    sqrt = lambda x:x.applyfunc(sp.sqrt)
    cos = lambda x:x.applyfunc(sp.cos)
    sin = lambda x:x.applyfunc(sp.sin)
    asin = lambda x:x.applyfunc(sp.asin)
    def add(x,n):
        return x.applyfunc(lambda y:y+n)

    h_d_constanT = h_constant/d_constanT

    b_function=add(h_constant*prod(x_u[:,3],cos(x_u[:,4])),d_constanT)\
        - sqrt(add(-(h_constant**2)*prod(sq(x_u[:,3]),sq(sin(x_u[:,4]))),d_constanT**2))

    system = sp.transpose(Array([
            x_u[:,0] + prod(b_function,cos(x_u[:,2])),
            x_u[:,1] + prod(b_function,sin(x_u[:,2])),
            x_u[:,2] + asin(h_d_constanT*prod(x_u[:,3],sin(x_u[:,4]))),
            x_u[:,3] + h_constant*x_u[:,5],
        ])).reshape(4*num_of_vehicle)

    x = np.array(x_u[:,0:4].reshape(4*num_of_vehicle))
    u = np.array(x_u[:,4:6].reshape(2*num_of_vehicle))
    x_u = Array(np.concatenate((x,u)))

    return system, x_u
