import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO

from simulator.env import DroneEnv

class Visualiser3D:
    def __init__(self, positions, targets):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.positions = positions
        self.targets = targets
        ani = FuncAnimation(self.fig, self.update, frames=len(positions), interval=10)
        plt.show()

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.plot3D(self.positions[:frame, 0], self.positions[:frame, 1], self.positions[:frame, 2], 'gray')
        
        self.ax.scatter(self.positions[frame, 0], self.positions[frame, 1], self.positions[frame, 2], color='blue', s=50, label="Drone Position")
        
        self.ax.scatter(self.targets[frame, 0], self.targets[frame, 1], self.targets[frame, 2], color='red', s=100, marker='X', label="Target Position")
        
        self.ax.legend()

""" Common Issues with Reinforcement Learning
- Rollout stats will not appear if env never terminates
"""

env = DroneEnv()
model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=1000000)
#model.save("test_model")

model = PPO.load("simulator/models/ppo_drone_model")

obs, _ = env.reset()
positions = [np.array(env.pos)]
targets = [np.array(env.target)]

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    positions.append(np.array(env.pos))
    targets.append(np.array(env.target))
    if terminated or truncated:
        break

positions = np.array(positions)
targets = np.array(targets)

vis = Visualiser3D(positions, targets)
