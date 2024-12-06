import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from simulator.env import DroneEnv


class Visualiser3D:
    def __init__(self, positions, yaws, targets):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.positions = positions
        self.yaws = yaws
        self.targets = targets

        ani = FuncAnimation(self.fig, self.update, frames=len(positions), interval=10)
        plt.show()

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.plot3D(
            self.positions[:frame, 0],
            self.positions[:frame, 1],
            self.positions[:frame, 2],
            "gray",
        )

        self.ax.scatter(
            self.positions[frame, 0],
            self.positions[frame, 1],
            self.positions[frame, 2],
            color="blue",
            s=50,
            label="Drone Position",
        )

        self.ax.scatter(
            self.targets[frame, 0],
            self.targets[frame, 1],
            self.targets[frame, 2],
            color="red",
            s=100,
            marker="X",
            label="Target Position",
        )

        yaw_angle = self.yaws[frame]
        dx = np.cos(yaw_angle)
        dy = np.sin(yaw_angle)
        dz = 0
        self.ax.quiver(
            self.positions[frame, 0],
            self.positions[frame, 1],
            self.positions[frame, 2],
            -dx*3, -dy*3, -dz*3,
            length=1.0,
            color='black',
            arrow_length_ratio=0.2
        )

        self.ax.legend()


""" 
Common Issues with Reinforcement Learning
- Rollout stats will not appear if env never terminates
"""
if __name__ == "__main__":
    cmd = argparse.ArgumentParser()

    cmd.add_argument(
        "-t",
        "--train",
        help="Enable model training",
        dest="is_training",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    cmd.add_argument(
        "-n",
        "--n-targets",
        help="Total number of targets in drone's flight path (-1 for inf)",
        dest="n",
        type=int,
        default=5,
    )

    args = cmd.parse_args()
    print(args)

    env = DroneEnv()
    model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="simulator/models/tmp",
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    if args.is_training:
        model.learn(total_timesteps=10000000, callback=checkpoint_callback)
        model.save("test_model")
    else:
        model = PPO.load("simulator/models/tmp/model_1500000_steps")

    obs, _ = env.reset(n_targets=args.n)

    positions = [np.array(env.pos)]
    yaws = [np.array(env.yaw)]
    targets = [np.array(env.target)]

    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        positions.append(np.array(env.pos))
        yaws.append(np.array(env.yaw))
        targets.append(np.array(env.target))

        if terminated or truncated:
            break

    positions = np.array(positions)
    yaws = np.array(yaws)
    targets = np.array(targets)

    vis = Visualiser3D(positions, yaws, targets)
