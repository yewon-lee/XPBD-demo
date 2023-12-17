import numpy as np
import math
from renderer import Render

class Pendulum:

    def __init__(self):

        # Pendulum simulation setup
        masses = np.array([0.5, 0.5, 0.5])
        n_masses = masses.shape[0]
        lengths = np.array([1, 1, 1])
        thetas = [math.pi/8, math.pi/6, math.pi/4]
        p = {'x': np.array([0, 0, 0]), 
             'y': np.array([0, 0, 0])}
        p_prev = {'x': np.array([0, 0, 0]), 
                  'y': np.array([0, 0, 0])}
        v = {'x': np.array([0, 0, 0]), 
             'y': np.array([0, 0, 0])}
        g = -9.81

        # Initialize mass positions
        for i in range(n_masses):
            if i==0:
                prev_mass_p = {'x': 0, 'y': 0}
            else:
                prev_mass_p = {'x': p['x'][i-1], 'y': p['y'][i-1]}
            p['x'][i] = prev_mass_p['x'] + lengths[i] * math.cos(thetas[i])
            p['y'][i] = prev_mass_p['x'] + lengths[i] * math.sin(thetas[i])

        # XPBD parameter setup
        xpbd_iterations = 10
        T = 5
        sim_dt = 0.1
        alpha = 0.5 * self.ones(self.n_masses) / (sim_dt^2)


    def compute_constraints(self, x, y):
        c = np.zeros(self.n_masses)
        for i in range(self.n_masses):
            if i==0:
                c[i] = np.sqrt((x[i]-0)^2 + (y[i]-0)^2)
            else:
                c[i] = np.sqrt((x[i]-x[i-1])^2 + (y[i]-y[i-1])^2)
        return c
    
    def compute_grad_constraints(self):
        return np.ones(self.n_masses)
    
    def compute_lambda_step(self, c, lmda):
        return (- c - self.alpha * lmda) / (np.sum(self.masses) + self.alpha)
    
    def compute_x_step(self, grad_c, lmda_step):
        return lmda_step * 


    def xpbd_step(self):

        # initialize solve & multiplier
        vx = self.v['x'] 
        vy = self.v['y'] + dt * self.g
        x_pred = self.p['x'] + dt * vx
        y_pred = self.p['y'] + dt * vy
        lambda_pred = np.zeros(self.n_masses)

        dt = self.sim_dt / self.xpbd_iterations

        for i in range(self.xpbd_iterations):
            continue

        # project constraints
        
            
