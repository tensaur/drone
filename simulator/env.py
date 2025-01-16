import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, n_targets=5, seed=None, options=None):
        self.n_targets = n_targets
        self.moves_left = 500

        self.pos = np.random.randint(-10, 11, 3)
        self.yaw = 0

        self.move_target = np.random.randint(-10, 11, 3)
        self.look_target = np.random.randint(-10, 11, 3)

        self.dist_slice = []
        self.dt = 10

        self.near_collision = np.array([0, 0, 0])

        self.colliders = [
            [
                np.array([0, 0, 0]),
                np.array([0, 10, 0]),
                np.array([0, 10, 10]),
                np.array([0, 0, 10]),
            ]
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

        vel = np.dot(rotation_matrix, action) * 0.1
        next_pos = np.clip(self.pos + vel, -10, 10)

        if np.linalg.norm(next_pos - self.move_target) < 2:
            reward += 1.0
            self.move_target = np.random.randint(-10, 11, 3)
            self.n_targets -= 1
            if self.n_targets == 0:
                terminated = True

        if np.linalg.norm(next_pos - self.move_target) < np.linalg.norm(
            self.pos - self.move_target
        ):
            reward += 0.1

        # Calculate the bounding box (min and max values for each axis)
        p_min = np.min(self.colliders[0], axis=0)
        p_max = np.max(self.colliders[0], axis=0)

        # Get normal vector of plane
        collider_norm = np.cross(
            self.colliders[0][1] - self.colliders[0][0],
            self.colliders[0][2] - self.colliders[0][0],
        )
        # and normalise this to a unit vector
        collider_norm = collider_norm / np.linalg.norm(collider_norm)

        # full plane is ax+by+cz+d=0. a,b,c are components of norm
        d = -np.dot(self.colliders[0][0], collider_norm)

        # Get shortest distance from drone to point on full plane
        dist_to_plane = np.abs(
            collider_norm[0] * self.pos[0]
            + collider_norm[1] * self.pos[1]
            + collider_norm[2] * self.pos[2]
            + d
        ) / np.sqrt(np.sum(collider_norm**2))

        # Get the closest point on the full plane
        # The line connecting the drone and the closest point on
        # the full plane is r=drone_pos+mu*plane_normal
        mu = -(d + np.dot(collider_norm, self.pos)) / np.sum(collider_norm**2)
        # then, the position on the full plane is:
        point_on_plane = self.pos + mu * collider_norm

        # Calculate vectors along sides of collider and from corner to point on plane
        p = self.pos - self.colliders[0][0]
        edges = [
            self.colliders[0][1] - self.colliders[0][0],
            self.colliders[0][3] - self.colliders[0][0],
        ]

        # Check if this point is already on the collider
        if (0 < np.dot(p, edges[0]) < np.dot(edges[0], edges[0])) and (
            0 < np.dot(p, edges[1]) < np.dot(edges[1], edges[1])
        ):
            closest_point, closest_dist = point_on_plane, dist_to_plane
        else:
            corner_dists = {
                tuple(self.colliders[0][i]): np.linalg.norm(
                    self.colliders[0][i] - point_on_plane
                )
                for i in range(0, 4)
            }

            # Closest side of the collider to the drone
            closest_corners = list(
                map(
                    np.asarray, sorted(corner_dists, key=lambda k: corner_dists[k])[0:2]
                )
            )

            # Vector along the closest edge of the collider
            closest_edge = closest_corners[1] - closest_corners[0]

            # The height of the triangle joining the two closest corners of the collider
            # and the closest point on the full plane to the drone. Uses h=2A/b where b is
            # the base of the triangle which is the edge of the collider object, and A is the area.
            # The area is calculated using A=0.5*ab*sin(theta)
            a = closest_corners[0] - point_on_plane
            b = closest_corners[1] - point_on_plane
            theta = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            A = 0.5 * np.linalg.norm(a) * np.linalg.norm(b) * np.sin(theta)
            h = (2 * A) / np.linalg.norm(closest_edge)

            # Find the closest point to the drone on this collider edge, using h
            # Can calculate this point using the equation for a general point on
            # the edge of the collider: r=corner+omega*closest_edge
            # Hence, rearrange h = |r-p|
            # TODO: (sam) explain maths for this part better
            omega = (
                -np.dot(closest_edge, closest_corners[0] - point_on_plane)
            ) / np.dot(closest_edge, closest_edge)

            if omega < 0 or omega > 1:
                omega = 0

            closest_point = closest_corners[0] + omega * closest_edge
            closest_dist = np.linalg.norm(closest_point - self.pos)

            # Verify answer is correct
            # np.testing.assert_almost_equal(
            #     closest_dist, np.sqrt(h**2 + dist_to_plane**2), decimal=10, verbose=True
            # )

        self.near_collision = closest_point

        # Check if the drone is inside the bounding box
        if (
            p_min[0] - 1 <= next_pos[0] <= p_max[0] + 1
            and p_min[1] <= next_pos[1] <= p_max[1]
            and p_min[2] <= next_pos[2] <= p_max[2]
        ):
            reward -= 1000
            truncated = True

        self.moves_left -= 1
        if self.moves_left == 0:
            truncated = True

        reward -= 0.2

        self.pos = next_pos

        # randomise yaw and look target -> learns to move towards look target regardless of yaw
        # self.yaw = np.random.rand() * 2 * np.pi  # (self.yaw + 0.1) % (2 * np.pi)
        # self.look_target = np.random.randint(-10, 11, 3)
        self.look_target = np.clip(self.look_target + (np.random.randn(3) / 5), -10, 10)
        self.yaw = np.arctan2(
            self.pos[1] - self.look_target[1], self.pos[0] - self.look_target[0]
        )

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        return np.concatenate(
            (
                self.move_target / 10,
                self.pos / 10,
                [np.sin(self.yaw)],
                [np.cos(self.yaw)],
            )
        )

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = DroneEnv()
    env.reset()
    print(env.get_observation())
