import numpy as np
import gymnasium as gym
from gymnasium import spaces


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8 + 6,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, n_targets=50, seed=None, options=None):
        self.n_targets = n_targets
        self.moves_left = 1500

        self.pos = np.random.randint(-10, 11, 3)
        self.yaw = 0

        self.move_target = np.random.randint(-10, 11, 3)
        self.look_target = np.random.randint(-10, 11, 3)

        self.dist_slice = []
        self.dt = 1

        self.near_collision = np.array([0, 0, 0])

        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        self.rays = np.unique(
            np.array(
                [
                    [
                        np.array([np.cos(angle), np.sin(angle), 0]).round(decimals=6)
                        for angle in angles
                    ],
                    [
                        np.array([0, np.cos(angle), np.sin(angle)]).round(decimals=6)
                        for angle in angles
                    ],
                    [
                        np.array([np.cos(angle), 0, np.sin(angle)]).round(decimals=6)
                        for angle in angles
                    ],
                ]
            ).reshape(-1, 3),
            axis=0,
        )
        self.prod_totals = np.zeros(len(self.rays))

        self.colliders = [
            [
                np.array([-10, -10, -10]),
                np.array([10, -10, -10]),
                np.array([10, -10, 10]),
                np.array([-10, -10, 10]),
            ],
            [
                np.array([10, -10, -10]),
                np.array([10, 10, -10]),
                np.array([10, 10, 10]),
                np.array([10, -10, 10]),
            ],
            [
                np.array([10, 10, -10]),
                np.array([10, 10, 10]),
                np.array([-10, 10, 10]),
                np.array([-10, 10, -10]),
            ],
            [
                np.array([-10, 10, -10]),
                np.array([-10, 10, 10]),
                np.array([-10, -10, 10]),
                np.array([-10, -10, -10]),
            ],
            [
                np.array([-10, -10, 10]),
                np.array([10, -10, 10]),
                np.array([10, 10, 10]),
                np.array([-10, 10, 10]),
            ],
            [
                np.array([-10, -10, -10]),
                np.array([10, -10, -10]),
                np.array([10, 10, -10]),
                np.array([-10, 10, -10]),
            ],
            [
                np.array([0, 0, -5]),
                np.array([0, 10, -5]),
                np.array([0, 10, 10]),
                np.array([0, 0, 10]),
            ],
            [
                np.array([0, 0, -5]),
                np.array([0, -10, -5]),
                np.array([0, -10, -10]),
                np.array([0, 0, -10]),
            ],
        ]

        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        rotation_matrix = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ]
        )

        vel = dot(rotation_matrix, action) * 0.1
        next_pos = np.clip(self.pos + vel, -10, 10)

        if np.linalg.norm(next_pos - self.move_target) < 1:
            reward += 1.0
            self.move_target = np.random.randint(-10, 11, 3)
            self.n_targets -= 1
            if self.n_targets == 0:
                pass
                # terminated = True

        # distance, closest point, collider index
        closest_collider = (np.inf, None, None)
        self.prod_totals = np.zeros(len(self.rays))

        for idx, collider in enumerate(self.colliders):
            # Get normal vector of plane
            collider_norm = np.cross(
                collider[1] - collider[0],
                collider[2] - collider[0],
            )
            # and normalise this to a unit vector
            collider_norm = collider_norm / np.linalg.norm(collider_norm)

            # full plane is ax+by+cz+d=0. a,b,c are components of norm
            d = -dot(collider[0], collider_norm)

            # Get shortest distance from drone to point on full plane
            dist_to_plane = np.abs(
                collider_norm[0] * next_pos[0]
                + collider_norm[1] * next_pos[1]
                + collider_norm[2] * next_pos[2]
                + d
            ) / np.sqrt(np.sum(collider_norm**2))

            # Get the closest point on the full plane
            # The line connecting the drone and the closest point on
            # the full plane is r=drone_pos+mu*plane_normal
            mu = -(d + dot(collider_norm, next_pos)) / np.sum(collider_norm**2)
            # then, the position on the full plane is:
            point_on_plane = next_pos + mu * collider_norm

            # Calculate vectors along sides of collider and from corner to point on plane
            p = next_pos - collider[0]
            edges = [
                collider[1] - collider[0],
                collider[3] - collider[0],
            ]

            # Check if this point is already on the collider
            if (0 < dot(p, edges[0]) < dot(edges[0], edges[0])) and (
                0 < dot(p, edges[1]) < dot(edges[1], edges[1])
            ):
                closest_point = point_on_plane
                closest_dist = dist_to_plane
            else:
                corner_dists = {
                    tuple(collider[i]): np.linalg.norm(collider[i] - point_on_plane)
                    for i in range(0, 4)
                }

                # Closest side of the collider to the drone
                closest_corners = list(
                    map(
                        np.asarray,
                        sorted(corner_dists, key=lambda k: corner_dists[k])[0:2],
                    )
                )

                # Vector along the closest edge of the collider
                closest_edge = closest_corners[1] - closest_corners[0]

                # Find the closest point to the drone on this collider edge
                # Can calculate this point using the equation for a general point on
                # the edge of the collider: r=corner+omega*closest_edge
                # Hence, rearrange h = |r-p| where h is the distance in the plane
                #
                # TODO: (sam) explain the maths for this part better
                omega = (-dot(closest_edge, closest_corners[0] - point_on_plane)) / dot(
                    closest_edge, closest_edge
                )

                if omega < 0 or omega > 1:
                    omega = 0

                closest_point = closest_corners[0] + omega * closest_edge
                closest_dist = np.linalg.norm(closest_point - next_pos)

            if closest_dist < closest_collider[0]:
                closest_collider = (closest_dist, closest_point, idx)

            for r, ray in enumerate(self.rays):
                dir_unit = (closest_point - next_pos) / np.linalg.norm(
                    closest_point - next_pos
                )

                projection = np.clip(dot(ray, dir_unit), 0, 1)
                if (projection > self.prod_totals[r]) and closest_dist < 1:
                    self.prod_totals[r] = projection * (1 - closest_dist)

        self.near_collision = closest_collider[1]
        self.dist_slice.append(closest_collider[0])

        if len(self.dist_slice) - 1 == self.dt:
            self.dist_slice.pop(0)

        col_rad = 0.55

        if np.min(self.dist_slice) <= col_rad:
            reward -= 0.25
        elif col_rad < np.min(self.dist_slice) < col_rad + 0.2:
            reward -= 0.1 + ((np.min(self.dist_slice) - col_rad) / 2)

        self.moves_left -= 1
        if self.moves_left == 0:
            truncated = True

        # reward -= 0.1

        self.pos = next_pos

        # randomise yaw and look target -> learns to move towards look target regardless of yaw
        self.yaw = np.random.rand() * 2 * np.pi  # (self.yaw + 0.1) % (2 * np.pi)
        # self.look_target = np.clip(self.look_target + (np.random.randn(3) / 5), -10, 10)
        # self.yaw = np.arctan2(
        #     self.pos[1] - self.look_target[1], self.pos[0] - self.look_target[0]
        # )

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        return np.concatenate(
            (
                self.move_target / 10,
                self.pos / 10,
                [np.sin(self.yaw)],
                [np.cos(self.yaw)],
                self.prod_totals,
            ),
            dtype=np.float32,
        )

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = DroneEnv()
    env.reset()
    print(env.get_observation())
