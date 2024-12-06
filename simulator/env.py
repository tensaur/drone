import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(12,), dtype=np.float32
        )

    def reset(self, n_targets=5, seed=None, options=None):
        self.n_targets = n_targets
        self.moves_left = 1000

        self.vel = np.array([0, 0, 0])
        self.pos = np.random.randint(-10, 11, 3)
        self.yaw_vel = 0
        self.yaw = np.random.random()

        self.target = np.random.randint(-10, 11, 3)
        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        next_yaw_vel = self.yaw_vel + action[3] * 0.1
        next_yaw = self.yaw + next_yaw_vel
        next_yaw = next_yaw % (2 * np.pi)

        c = np.cos(next_yaw)
        s = np.sin(next_yaw)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        local_vel_increment = action[:3] * 0.1
        global_vel_increment = np.dot(R,  local_vel_increment)

        next_vel = self.vel + global_vel_increment
        next_pos = self.pos + next_vel

        to_target = next_pos - self.target
        to_target /= np.linalg.norm(to_target)

        yaw_vector = np.array([np.cos(next_yaw), np.sin(next_yaw), 0])
        yaw_vector /= np.linalg.norm(yaw_vector)

        angle_rew = np.dot(to_target, yaw_vector)
        #reward += angle_rew * 0.1

        if np.linalg.norm(self.pos - self.target) > np.linalg.norm(next_pos - self.target):
            reward += 0.2

        if np.linalg.norm(next_pos - self.target) < 1:
            reward += 1.0
            self.target = np.random.randint(-10, 11, 3)
            self.n_targets -= 1
            if self.n_targets == 0:
                terminated = True

        self.vel = next_vel
        self.pos = next_pos
        self.yaw_vel = next_yaw_vel
        self.yaw = next_yaw

        self.moves_left -= 1
        if self.moves_left == 0:
            truncated = True

        reward -= 0.1

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        return np.concatenate((self.pos / 10, self.vel, [np.sin(self.yaw)], [np.cos(self.yaw)], [self.yaw_vel], self.target / 10))

    def render(self):
        pass

    def close(self):
        pass
