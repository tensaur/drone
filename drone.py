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
        rolls,
        pitches,
        yaws,
    ):
        self.fig = plt.figure("Drone Simulation Tool for Warwick AI")
        self.ax = self.fig.add_subplot(projection="3d")

        self.positions = positions
        self.move_targets = move_targets
        self.look_targets = look_targets
        self.rolls = rolls
        self.pitches = pitches
        self.yaws = yaws

        _ani = FuncAnimation(self.fig, self.update, frames=len(positions), interval=1)
        plt.show()

    def update(self, frame):
        self.ax.clear()

        # set axes limits & labels
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # draw trajectory so far
        self.ax.plot(
            self.positions[:frame, 0],
            self.positions[:frame, 1],
            self.positions[:frame, 2],
            color="gray",
        )

        # scatter current pos & targets
        pos = self.positions[frame]
        self.ax.scatter(*pos, color="blue", s=50, label="Drone Position")
        self.ax.scatter(*self.move_targets[frame], color="red",   s=100, marker="X", label="Move Target")
        self.ax.scatter(*self.look_targets[frame], color="green", s=100, marker="X", label="Look Target")

        # extract Euler angles
        φ = self.rolls[frame]
        θ = self.pitches[frame]
        ψ = self.yaws[frame]

        cφ, sφ = np.cos(φ), np.sin(φ)
        cθ, sθ = np.cos(θ), np.sin(θ)
        cψ, sψ = np.cos(ψ), np.sin(ψ)

        # body→world rotation matrix
        R = np.array([
            [ cθ*cψ,             sφ*sθ*cψ - cφ*sψ,   cφ*sθ*cψ + sφ*sψ ],
            [ cθ*sψ,             sφ*sθ*sψ + cφ*cψ,   cφ*sθ*sψ - sφ*cψ ],
            [ -sθ,               sφ*cθ,               cφ*cθ           ]
        ])

        # body axes in world frame
        x_body = R @ np.array([1.0, 0.0, 0.0])
        y_body = R @ np.array([0.0, 1.0, 0.0])
        z_body = R @ np.array([0.0, 0.0, 1.0])

        length = 3.0  # scale arrows
        origin = pos

        # draw orthonormal axes
        self.ax.quiver(*origin, *(-x_body * length), color="blue",  arrow_length_ratio=0.2, normalize=False)
        self.ax.quiver(*origin, *(-y_body * length), color="red",   arrow_length_ratio=0.2, normalize=False)
        self.ax.quiver(*origin, *(-z_body * length), color="green", arrow_length_ratio=0.2, normalize=False)

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
    model = torch.load("experiments/drone-97f6570d/model_001150.pt")

    # obs, _ = env.reset(n_targets=args.n)
    obs, _ = env.reset()
    positions = [np.array(env.pos)]
    move_targets = [np.array(env.move_target)]
    look_targets = [np.array(env.look_target)]
    rolls = [np.array(env.roll)]
    pitches = [np.array(env.pitch)]
    yaws = [np.array(env.yaw)]

    while True:
        obs = torch.tensor([obs], dtype=torch.float32)
        print("obs:")
        print(obs)
        action, _, _, _ = model(obs)
        obs, reward, terminated, truncated, info = env.step(np.array(action).flatten())
        positions.append(np.array(env.pos))
        move_targets.append(np.array(env.move_target))
        look_targets.append(np.array(env.look_target))
        rolls.append(np.array(env.roll))
        pitches.append(np.array(env.pitch))
        yaws.append(np.array(env.yaw))
        if terminated or truncated:
            break

    for p in positions:
        print(p)

    positions = np.array(positions)
    print(positions)
    move_targets = np.array(move_targets)
    look_targets = np.array(look_targets)
    rolls = np.array(rolls)
    pitches = np.array(pitches)
    yaws = np.array(yaws)

    vis = Visualiser3D(
        positions,
        move_targets,
        look_targets,
        rolls,
        pitches,
        yaws,
    )
