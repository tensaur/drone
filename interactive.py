import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulator.ppo import Agent
from simulator.env import DroneEnv

import torch

from room_mesher import mesh

mpl.rcParams['keymap.save'] = []
mpl.rcParams['keymap.pan'] = []
mpl.rcParams['keymap.back'] = []
mpl.rcParams['keymap.forward'] = []
mpl.rcParams["axes3d.mouserotationstyle"] = "azel"


class Visualiser3D:
    def __init__(self, colliders):

        self.env = DroneEnv()
        self.obs, _ = self.env.reset(n_targets=1, colliders=colliders)

        self.model = Agent(None)
        self.model.load_state_dict(torch.load("end.pth"))

        self.paused = False

        self.fig = plt.figure("Drone Simulation Tool for Warwick AI")
        self.ax = self.fig.add_subplot(projection="3d")
        _ani = FuncAnimation(self.fig, self.update, frames=1000, interval=10)

        cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def on_key(self, event):
        step_size = 2  # Define how much to move the target
        if event.key == 'up':
            self.env.move_target[1] += step_size
        elif event.key == 'down':
            self.env.move_target[1] -= step_size
        elif event.key == 'left':
            self.env.move_target[0] -= step_size
        elif event.key == 'right':
            self.env.move_target[0] += step_size
        elif event.key == 'w':
            self.env.move_target[2] += step_size
        elif event.key == 's':
            self.env.move_target[2] -= step_size
        elif event.key == 'p':
            self.paused = not self.paused

    def update(self, frame):
        if not self.paused:
            action, _, _, _ = self.model.get_action_and_value(
                torch.tensor([self.obs], dtype=torch.float32)
            )
            self.obs, reward, terminated, truncated, info = self.env.step(np.array(action).flatten())

        self.draw()
    
    def draw(self):
        self.ax.clear()

        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        drone_pos = self.env.pos
        self.ax.scatter(
            drone_pos[0],
            drone_pos[1],
            drone_pos[2],
            color="blue",
            s=50,
            label="Drone Position",
        )

        move_target = self.env.move_target
        self.ax.scatter(
            move_target[0],
            move_target[1],
            move_target[2],
            color="red",
            s=100,
            marker="X",
            label="Move Target Position",
        )

        look_target = self.env.look_target
        self.ax.scatter(
            look_target[0],
            look_target[1],
            look_target[2],
            color="green",
            s=100,
            marker="X",
            label="Look Target Position",
        )

        yaw_angle = self.env.yaw
        dx = np.cos(yaw_angle)
        dy = np.sin(yaw_angle)
        dz = 0

        self.ax.quiver(
            drone_pos[0],
            drone_pos[1],
            drone_pos[2],
            -dx * 3,
            -dy * 3,
            -dz * 3,
            length=1.0,
            color="green",
            arrow_length_ratio=0.2,
        )

        near_collision = self.env.near_collision
        self.ax.quiver(
            drone_pos[0],
            drone_pos[1],
            drone_pos[2],
            near_collision[0] - drone_pos[0],
            near_collision[1] - drone_pos[1],
            near_collision[2] - drone_pos[2],
            length=1.0,
            color="black",
            linestyles="dotted",
            linewidths=0.8,
            arrow_length_ratio=0,
        )

        self.ax.scatter(
            near_collision[0],
            near_collision[1],
            near_collision[2],
            color="black",
            s=100,
            marker="X",
            label="Closest Collision Point",
        )

        poly3d = Poly3DCollection(self.env.colliders)
        poly3d.set_zsort("average")  # Important for depth ordering
        poly3d.set_facecolor("gray")
        poly3d.set_edgecolor("black")
        poly3d.set_alpha(0.3)  # Slight transparency to see behind

        self.ax.add_collection3d(poly3d)

        self.ax.legend(prop={"size": 7}, markerscale=0.6)


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
    cmd.add_argument(
        "-r",
        "--room-path",
        help="Path to .ply file containing room point cloud",
        dest="r",
        type=str,
        default=None,
    )

    args = cmd.parse_args()
    print(args)

    vis = Visualiser3D(colliders=mesh(args.r) if args.r else None)