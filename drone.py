# type: ignore
import argparse

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulator.env import Drone

mpl.rcParams["axes3d.mouserotationstyle"] = "azel"


class Visualiser3D:
    def __init__(
        self,
        positions,
        move_targets,
        look_targets,
        yaws,
    ):
        self.fig = plt.figure("Drone Simulation Tool for Warwick AI")
        self.ax = self.fig.add_subplot(projection="3d")

        self.positions = positions
        self.move_targets = move_targets
        self.look_targets = look_targets
        self.yaws = yaws

        _ani = FuncAnimation(self.fig, self.update, frames=len(positions), interval=10)
        plt.show()

    def update(self, frame):
        self.ax.clear()

        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.plot(
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
            self.move_targets[frame, 0],
            self.move_targets[frame, 1],
            self.move_targets[frame, 2],
            color="red",
            s=100,
            marker="X",
            label="Move Target Position",
        )

        self.ax.scatter(
            self.look_targets[frame, 0],
            self.look_targets[frame, 1],
            self.look_targets[frame, 2],
            color="green",
            s=100,
            marker="X",
            label="Look Target Position",
        )

        yaw_angle = self.yaws[frame]
        dx = np.cos(yaw_angle)
        dy = np.sin(yaw_angle)
        dz = 0

        self.ax.quiver(
            self.positions[frame, 0],
            self.positions[frame, 1],
            self.positions[frame, 2],
            -dx * 3,
            -dy * 3,
            -dz * 3,
            length=1.0,
            color="green",
            arrow_length_ratio=0.2,
        )

        self.ax.legend(prop={"size": 7}, markerscale=0.6)


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

    env = Drone(num_envs=1)
    model = torch.load("experiments/drone-e91e02cc/model_000850.pt")

    # obs, _ = env.reset(n_targets=args.n)
    obs, _ = env.reset()
    positions = [np.array(env.pos)]
    move_targets = [np.array(env.move_target)]
    look_targets = [np.array(env.look_target)]
    yaws = [np.array(env.yaw)]

    while True:
        obs = torch.tensor([obs], dtype=torch.float32)
        action, _, _, _ = model(obs)
        action = np.array([1000000,1000000,0,0])
        obs, reward, terminated, truncated, info = env.step(np.array(action).flatten())
        positions.append(np.array(env.pos))
        move_targets.append(np.array(env.move_target))
        look_targets.append(np.array(env.look_target))
        yaws.append(np.array(env.yaw))
        if terminated or truncated:
            break

    for p in positions:
        print(p)

    positions = np.array(positions)
    print(positions)
    move_targets = np.array(move_targets)
    look_targets = np.array(look_targets)
    yaws = np.array(yaws)

    vis = Visualiser3D(
        positions,
        move_targets,
        look_targets,
        yaws,
    )
