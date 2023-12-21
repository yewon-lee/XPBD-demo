import matplotlib.pyplot as plt
import numpy as np

lengths = [0.2, 0.2, 0.2]
masses = [1.0, 0.5, 0.3]
angles = [0.5 * np.pi, np.pi, np.pi]

class Pendulum:
    def __init__(self, masses, lengths, angles):
        self.masses = [0.0]
        self.lengths = [0.0]
        self.pos = [{'x': 0.0, 'y': 0.0}]
        self.prevPos = [{'x': 0.0, 'y': 0.0}]
        self.vel = [{'x': 0.0, 'y': 0.0}]
        x, y = 0.0, 0.0
        for i in range(len(masses)):
            self.masses.append(masses[i])
            self.lengths.append(lengths[i])
            x += lengths[i] * np.sin(angles[i])
            y += lengths[i] * -np.cos(angles[i])
            self.pos.append({'x': x, 'y': y})
            self.prevPos.append({'x': x, 'y': y})
            self.vel.append({'x': 0, 'y': 0})

    def simulate(self, dt, gravity):
        for i in range(1, len(self.masses)):
            self.vel[i]['y'] += dt * gravity
            self.prevPos[i]['x'] = self.pos[i]['x']
            self.prevPos[i]['y'] = self.pos[i]['y']
            self.pos[i]['x'] += self.vel[i]['x'] * dt
            self.pos[i]['y'] += self.vel[i]['y'] * dt

        for i in range(1, len(self.masses)):
            dx = self.pos[i]['x'] - self.pos[i - 1]['x']
            dy = self.pos[i]['y'] - self.pos[i - 1]['y']
            d = np.sqrt(dx * dx + dy * dy)
            w0 = 1.0 / self.masses[i - 1] if self.masses[i - 1] > 0.0 else 0.0
            w1 = 1.0 / self.masses[i] if self.masses[i] > 0.0 else 0.0
            corr = (self.lengths[i] - d) / d / (w0 + w1)
            self.pos[i - 1]['x'] -= w0 * corr * dx
            self.pos[i - 1]['y'] -= w0 * corr * dy
            self.pos[i]['x'] += w1 * corr * dx
            self.pos[i]['y'] += w1 * corr * dy

        for i in range(1, len(self.masses)):
            self.vel[i]['x'] = (self.pos[i]['x'] - self.prevPos[i]['x']) / dt
            self.vel[i]['y'] = (self.pos[i]['y'] - self.prevPos[i]['y']) / dt

    def draw(self):
        plt.figure(figsize=(8, 6))
        for i in range(1, len(self.masses)):
            plt.plot([self.pos[i - 1]['x'], self.pos[i]['x']], [self.pos[i - 1]['y'], self.pos[i]['y']], color="#303030", linewidth=10)
            r = 0.05 * np.sqrt(self.masses[i])
            circle = plt.Circle((self.pos[i]['x'], self.pos[i]['y']), r, color="#FF0000")
            plt.gca().add_patch(circle)

        plt.axis('equal')
        plt.axis('off')
        plt.show()

gravity = -10.0
dt = 0.01
num_sub_steps = 100
pendulum = Pendulum(masses, lengths, angles)

for step in range(num_sub_steps):
    pendulum.simulate(dt / num_sub_steps, gravity)

pendulum.draw()
