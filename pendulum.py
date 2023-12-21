import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Pendulum:
    '''
    2D pendulum simulation:

    Pendulum with different masses has some initial starting condition (zero initial velocity, non zero position)
    The effect of gravity acting on the pendulum is simulated
    '''

    def __init__(self):

        # Pendulum simulation setup
        self.masses = np.array([1.0, 0.5, 0.3, 2.0])
        self.n_masses = self.masses.shape[0]
        self.lengths = np.array([1, 1, 1, 1])
        thetas = [math.pi/8, math.pi/8, math.pi/8, math.pi/8]
        self.p = np.zeros((2, self.n_masses))
        self.p_prev = np.zeros((2, self.n_masses))
        self.v = np.zeros((2, self.n_masses))
        self.g = -9.81

        # Initialize mass positions
        for i in range(self.n_masses):
            if i == 0:
                prev_mass_p = {'x': 0, 'y': 0}
            else:
                prev_mass_p = {'x': self.p[0, i - 1], 'y': self.p[1, i - 1]}
            self.p[0, i] = prev_mass_p['x'] - self.lengths[i] * math.cos(thetas[i])
            self.p[1, i] = prev_mass_p['y'] - self.lengths[i] * math.sin(thetas[i])


        # XPBD parameter setup
        self.xpbd_iterations = 1000
        self.T = 5
        self.sim_dt = 0.01
        self.alpha = math.pow(10, -9) * np.ones(self.n_masses) / math.pow(self.sim_dt, 2)

        # Set up Matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 4)
        self.ax.set_title('2D Pendulum', fontsize=14)

        # Set aspect ratio to 'equal'
        self.ax.set_aspect('equal') 

        # Set the background color to black
        self.ax.set_facecolor('black')

        # Create lines representing the pendulum
        self.lines = [self.ax.plot([], [], '-', lw=2)[0] for _ in range(self.n_masses)]
        self.markers = [self.ax.scatter([], [], s=300 * np.sqrt(self.masses[i]), color='white', marker='o') for i in range(self.n_masses)]


    def compute_constraints(self, x, y):
        c = np.zeros(self.n_masses)
        for i in range(self.n_masses):
            if i==0:
                c[i] = (np.sqrt(math.pow(x[i]-0, 2) + math.pow(y[i]-0, 2)) - self.lengths[i])**2
            else:
                c[i] = (np.sqrt(math.pow(x[i]-x[i-1], 2) + math.pow(y[i]-y[i-1], 2)) - self.lengths[i])**2
        return c
    
    def compute_grad_constraints(self, x, y):
        # Inputs:
        #       x: numpy array of size (num_masses)
        #       y: numpy array of size (num_masses)
        #       c: constrats; numpy array of size (num_masses)
        # Output: returns gradient of constraint wrt position
        #         array of size (2, num_constraints)
        grad_c = np.zeros((2, self.n_masses))
        for i in range(self.n_masses):
            if i==0:
                norm = np.sqrt(math.pow(x[i]-0, 2) + math.pow(y[i]-0, 2))
                grad_c[0, i] = ((x[i]-0) / norm) * 2 * (norm - self.lengths[i])
                grad_c[1, i] = ((y[i]-0) / norm) * 2 * (norm - self.lengths[i])
            else:
                norm = np.sqrt(math.pow(x[i]-x[i-1], 2) + math.pow(y[i]-y[i-1], 2))
                grad_c[0, i] = ((x[i] - x[i-1]) / norm) * 2 * (norm - self.lengths[i])
                grad_c[1, i] = ((y[i] - y[i-1]) / norm) * 2 * (norm - self.lengths[i])
        return grad_c
    
    def compute_lambda_step(self, c, lmda):
        # Inputs:
        #   c: numpy of constraints evaluated with current position; dimension is # of constraints
        #   lmda: numpy of lambdas; dimensions is # of constraints
        # Outputs:
        #   return lambda step, which is an array where shape is number of constraints

        return (- c - self.alpha * lmda) / (np.sum(self.masses) + self.alpha)
    
    def compute_pos_step(self, lmda_step, p_pred):
        return lmda_step * (1/self.masses) * self.compute_grad_constraints(p_pred[0, :], p_pred[1, :])


    def xpbd_step(self):

        # initialize solve & multiplier
        vx = self.v[0,:] 
        vy = self.v[1,:] + self.sim_dt * self.g
        p_pred = self.p + self.sim_dt * np.vstack([vx, vy])
        lmda_pred = np.zeros(self.n_masses) # dimensions are same as number of constraints

        for i in range(self.xpbd_iterations):
            # compute constraints
            c = self.compute_constraints(p_pred[0, :], p_pred[1, :])
            # update lambda
            lmda_step = self.compute_lambda_step(c, lmda_pred)
            lmda_pred += lmda_step
            # update position
            p_pred += self.compute_pos_step(lmda_step, p_pred)

        self.v = (p_pred - self.p) / self.sim_dt
        self.p = p_pred


    def render(self):
        # Update the positions and colors of the masses and the lines connecting them
        for i in range(self.n_masses):
            # Set the data for lines with a lower zorder
            self.lines[i].set_data(self.p[0, i], self.p[1, i])
            self.lines[i].set_zorder(1)  # Lower zorder for lines

            # Set the data and color for markers with a higher zorder
            self.markers[i].set_offsets([self.p[0, i], self.p[1, i]])
            line_color = self.lines[i].get_color()
            self.markers[i].set_facecolor(line_color)
            self.markers[i].set_zorder(2)  # Higher zorder for markers

        # Update the positions of the lines connecting the masses
        for i in range(self.n_masses - 1):
            line_x = [self.p[0, i], self.p[0, i + 1]]
            line_y = [self.p[1, i], self.p[1, i + 1]]
            self.lines[i + 1].set_data(line_x, line_y)

        # Update the position of the line connecting the first mass to the origin
        self.lines[0].set_data([0, self.p[0, 0]], [0, self.p[1, 0]])

        # Pause for a short time to create a real-time effect
        plt.pause(0.01)


    def simulate_step(self, frame):
        self.xpbd_step()
        self.render()


# Instantiate the Pendulum class
pendulum = Pendulum()

# Set up the animation
num_frames = int(pendulum.T/pendulum.sim_dt)
ani = FuncAnimation(pendulum.fig, pendulum.simulate_step, frames=num_frames, interval=50, blit=False)

# Display the animation
plt.show()