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

mpl.rcParams["axes3d.mouserotationstyle"] = "azel"


class Visualiser3D:
    def __init__(
        self,
        positions,
        move_targets,
        look_targets,
        yaws,
        near_collisions,
        colliders,
        rays,
    ):
        self.fig = plt.figure("Drone Simulation Tool for Warwick AI")
        self.ax = self.fig.add_subplot(projection="3d")
        #self.ax.invert_xaxis()

        # Optional: adjust the initial view
        elev, azim, roll = 0, 0, 0
        self.ax.view_init(elev, azim, roll)

        # Store references
        self.positions = positions
        self.move_targets = move_targets
        self.look_targets = look_targets
        self.yaws = yaws
        self.near_collisions = near_collisions
        self.colliders = colliders
        self.rays = rays

        # Create animation
        _ani = FuncAnimation(self.fig, self.update, frames=len(positions), interval=10)
        plt.show()

    def update(self, frame):
        self.ax.clear()
        #self.ax.invert_xaxis()

        # Re-label to reflect swapped axes
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")  # swapped
        self.ax.set_zlabel("Y")  # swapped

        # Limits (optional; reorder or keep them as is)
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])  # This is now 'Z'
        self.ax.set_zlim([-10, 10])  # This is now 'Y'

        # Plot drone path
        self.ax.plot(
            self.positions[:frame, 0],
            self.positions[:frame, 1],
            self.positions[:frame, 2],
            "gray",
        )

        # Drone position
        self.ax.scatter(
            self.positions[frame, 0],
            self.positions[frame, 1],
            self.positions[frame, 2],
            color="blue",
            s=50,
            label="Drone Position",
        )

        # Move target
        self.ax.scatter(
            self.move_targets[frame, 0],
            self.move_targets[frame, 1],
            self.move_targets[frame, 2],
            color="red",
            s=100,
            marker="X",
            label="Move Target Position",
        )

        # Look target
        self.ax.scatter(
            self.look_targets[frame, 0],
            self.look_targets[frame, 1],
            self.look_targets[frame, 2],
            color="green",
            s=100,
            marker="X",
            label="Look Target Position",
        )

        # Orientation arrow
        yaw_angle = self.yaws[frame]
        dx = np.cos(yaw_angle)
        dy = np.sin(yaw_angle)
        dz = 0  # For a pure yaw, z=0 offset

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

        # Near-collision vector
        self.ax.quiver(
            self.positions[frame, 0],
            self.positions[frame, 1],
            self.positions[frame, 2],
            self.near_collisions[frame, 0] - self.positions[frame, 0],
            self.near_collisions[frame, 1] - self.positions[frame, 1],
            self.near_collisions[frame, 2] - self.positions[frame, 2],
            length=1.0,
            color="black",
            linestyles="dotted",
            linewidths=0.8,
            arrow_length_ratio=0,
        )

        # Closest collision point
        self.ax.scatter(
            self.near_collisions[frame, 0],
            self.near_collisions[frame, 1],
            self.near_collisions[frame, 2],
            color="black",
            s=100,
            marker="X",
            label="Closest Collision Point",
        )

        # Rays
        for r in self.rays:
            self.ax.quiver(
                self.positions[frame, 0],
                self.positions[frame, 1],
                self.positions[frame, 2],
                r[0],
                r[1],
                r[2],
                length=1.0,
                color="blue",
                arrow_length_ratio=0.2,
            )

        # Colliders
        poly3d = Poly3DCollection(self.colliders)
        poly3d.set_zsort("average")  # Important for depth ordering
        poly3d.set_facecolor("gray")
        poly3d.set_edgecolor("black")
        poly3d.set_alpha(0.3)

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

    # Create environment, model
    env = DroneEnv()
    model = Agent(None)
    model.load_state_dict(torch.load("end.pth"))

    # Reset env and grab initial arrays
    obs, _ = env.reset(n_targets=args.n, colliders=mesh(args.r) if args.r else None)
    positions = [np.array(env.pos)]
    move_targets = [np.array(env.move_target)]
    look_targets = [np.array(env.look_target)]
    yaws = [np.array(env.yaw)]
    near_collisions = [np.array(env.near_collision)]

    # Run simulation loop
    while True:
        action, _, _, _ = model.get_action_and_value(
            torch.tensor([obs], dtype=torch.float32)
        )
        obs, reward, terminated, truncated, info = env.step(np.array(action).flatten())

        positions.append(np.array(env.pos))
        move_targets.append(np.array(env.move_target))
        look_targets.append(np.array(env.look_target))
        yaws.append(np.array(env.yaw))
        near_collisions.append(np.array(env.near_collision))

        if terminated or truncated:
            break

    # Convert to NumPy arrays
    positions = np.array(positions)
    move_targets = np.array(move_targets)
    look_targets = np.array(look_targets)
    yaws = np.array(yaws)
    near_collisions = np.array(near_collisions)
    colliders = np.array(env.colliders)  # shape (n, 4, 3)

    # ------------------------------------------------
    # SWAP Y <-> Z in all relevant arrays here:
    # positions, move_targets, look_targets, near_collisions, and colliders
    # ------------------------------------------------
    positions[..., [1, 2]] = positions[..., [2, 1]]
    move_targets[..., [1, 2]] = move_targets[..., [2, 1]]
    look_targets[..., [1, 2]] = look_targets[..., [2, 1]]
    near_collisions[..., [1, 2]] = near_collisions[..., [2, 1]]
    colliders[..., [1, 2]] = colliders[..., [2, 1]]

    # Pass the swapped arrays to the visualiser
    vis = Visualiser3D(
        positions=positions,
        move_targets=move_targets,
        look_targets=look_targets,
        yaws=yaws,
        near_collisions=near_collisions,
        colliders=colliders,
        rays=env.rays,  # optional: if you want to swap Y/Z in rays, do the same
    )
