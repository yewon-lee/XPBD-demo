import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class BouncingBall:
    '''
    2D bouncing ball-in-a-box example:

    There are N number of balls which have some initial starting position and velocity
    The balls are enclosed in a bounded box and can move around anywhere inside this box
    '''

    def __init__(self):

        # Ball-in-a-box with gravity: simulation setup
        self.radii = np.array([0.2 , 0.4])
        self.masses = (self.radii ** 2) * 100
        self.n_balls = self.radii.shape[0]
        self.p = np.array([[0.5, 2.5], 
                           [2.5, 2.5]])
        self.p_prev = np.zeros((2, self.n_balls))
        self.v = np.array([[30, -30],
                           [30, 30]])
        self.g = 0 #-9.81
        self.skin_thickness = 0.02 # artificial skin thickness to allow for earlier collision detection

        # Number of constraints per ball
        self.n_constraints = 5 # XXX: considering contact with walls and between 2 balls

        # Box configuration
        self.w = 3
        self.h = 3

        # XPBD parameter setup
        self.xpbd_iterations = 1000
        self.T = 5
        self.sim_dt = 0.01
        self.alpha = math.pow(10, -5) / math.pow(self.sim_dt, 2)

        # Set up Matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_title('Bouncing Balls in a Box', fontsize=14)

        # Set aspect ratio to 'equal'
        self.ax.set_aspect('equal') 

        # Set the background color to black
        self.ax.set_facecolor('black')

        # Create colors for the balls
        brighter_pastel_colors = ['#FFA1B5', '#FFE872', '#77FFEA', '#8AC9FF', '#FFA573']
        self.colors = brighter_pastel_colors[:self.n_balls]

        # Create circles with a white border
        self.circles = [
            plt.Circle(
                (self.p[0, i], self.p[1, i]),
                self.radii[i],
                fill=True,
                color=self.colors[i],
                edgecolor='white',  # Add a white border
                linewidth=1          # Set the width of the border
            ) for i in range(self.n_balls)]

        # Add circles to the plot
        [self.ax.add_patch(circle) for circle in self.circles]

        # Create quiver arrows for velocity representation
        self.quiver_arrows = [self.ax.quiver(0, 0, 0, 0, color='black', scale_units='xy', scale=10) for i in range(self.n_balls)]

        # Whether or not to render arrow
        self.arrow = False


    def compute_constraints(self, x, y):
        # Inputs: 
        #       x and y are (n_balls, 2) arrays
        # Outputs:
        #       returns constraints c, which are (n_balls, n_constraints) arrays 
        #      

        c = np.zeros((self.n_balls, self.n_constraints))
        for i in range(self.n_balls):
            c[i, 0] = x[i] - self.radii[i]
            c[i, 1] = y[i] - self.radii[i]
            c[i, 2] = self.w - (x[i] + self.radii[i])
            c[i, 3] = self.h - (y[i] + self.radii[i])
        c[0, 4] = (x[0] - x[1])** 2 + (y[0] - y[1])**2 - (self.radii[0] + self.radii[1] + self.skin_thickness)**2 # XXX: we assume 2 balls for the ball-ball collision constraint
        c[1, 4] = (x[0] - x[1])** 2 + (y[0] - y[1])**2 - (self.radii[0] + self.radii[1] + self.skin_thickness)**2 # XXX: we assume 2 balls for the ball-ball collision constraint
        return c
    
    def compute_grad_constraints(self, x, y):
        # Inputs:
        #       x: (n_balls, 2) arrays
        #       y: (n_balls, 2) arrays
        #       c: (n_balls, n_constraints) constraint arrays 
        # Output: 
        #       returns gradient of constraints wrt position
        #       array of size (n_balls, n_constraints, 2)
        
        grad_c = np.zeros((self.n_balls, self.n_constraints, 2))
        grad_c[:, 0, :] = np.array([1, 0])
        grad_c[:, 1, :] = np.array([0, 1])
        grad_c[:, 2, :] = np.array([-1, 0])
        grad_c[:, 3, :] = np.array([0, -1])
        grad_c[0, 4, :] = np.array([2*(x[0]-x[1]), 2*(y[0]-y[1])])
        grad_c[1, 4, :] = np.array([2*(x[1]-x[0]), 2*(y[1]-y[0])])

        return grad_c


    def compute_lambda_step(self, c, lmda, grad_c, ball_id):
        grad_c_norm = np.sqrt(grad_c[ball_id, 4, 0]**2 + grad_c[ball_id, 4, 1]**2)
        denom = (np.sum(self.masses) * (self.n_constraints-1) + grad_c_norm * self.masses[ball_id] + self.alpha)
        return (- c - self.alpha * lmda) / denom
    

    def compute_pos_step(self, lmda_step, grad_c, ball_id):
        return lmda_step * grad_c * (1/self.masses[ball_id]) 


    def xpbd_step(self):

        # initialize solve & multiplier
        vx = self.v[0,:] 
        vy = self.v[1,:] + self.sim_dt * self.g
        p_pred = self.p + self.sim_dt * np.vstack([vx, vy])
        lmda_pred = np.zeros((self.n_balls, self.n_constraints)) # dimensions are same as total number of constraints

        for i in range(self.xpbd_iterations):

            # compute constraints
            c = self.compute_constraints(p_pred[0, :], p_pred[1, :]) # array of size (n_balls, 4)
            grad_c = self.compute_grad_constraints(p_pred[0, :], p_pred[1, :]) # array of size (n_balls, 4, 2)

            for j in range(self.n_balls):
                for k in range(self.n_constraints):
                    if c[j, k] < 0:
                        # update lambda
                        lmda_step = self.compute_lambda_step(c[j, k], lmda_pred[j, k], grad_c, j)
                        lmda_pred[j, k] += lmda_step

                        # update position
                        p_pred[:, j] += self.compute_pos_step(lmda_step, grad_c[j, k, :], j)

        self.v = (p_pred - self.p) / self.sim_dt
        self.p = p_pred


    def render_arrows(self):

        # Normalize the velocity vector
        for i in range(self.n_balls):
            v_norm = np.linalg.norm(self.v[:, i])
            normalized_v = self.v[:, i] / v_norm if v_norm != 0 else np.zeros(2)

            self.quiver_arrows[i].set_offsets(self.p[:, i])
            self.quiver_arrows[i].set_UVC(normalized_v[0], normalized_v[1])

    def render(self):

        for i in range(self.n_balls):
            self.circles[i].center = (self.p[0, i], self.p[1, i])

        if self.arrow:
            self.render_arrows()

        # Pause for a short time to create a real-time effect
        plt.pause(0.01)

            
    def simulate_step(self, frame):
        print(self.p)
        self.xpbd_step()
        self.render()


# Instantiate the bouncing ball class
bounce = BouncingBall()

# Set up the animation
num_frames = int(bounce.T/bounce.sim_dt)
ani = FuncAnimation(bounce.fig, bounce.simulate_step, frames=num_frames, interval=50, blit=False)


# Display the animation
plt.show()