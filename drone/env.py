import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3,3,3])


    def reset(self, n_targets=5, seed=None, options=None):
        self.n_targets = n_targets
        self.moves_left = 500

        self.pos = np.random.randint(-10, 11, 3)
        self.yaw = 0

        self.target = np.random.randint(-10, 11, 3)
        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])

        move_vector = np.dot(rotation_matrix, (action[:3] - 1) / 10)
        next_pos = np.clip(self.pos + move_vector, -10, 10)

        if np.linalg.norm(next_pos - self.target) < 1:
            reward += 1.0
            self.target = np.random.randint(-10, 11, 3)
            self.n_targets -= 1
            if self.n_targets == 0:
                terminated = True
        
        if np.linalg.norm(next_pos - self.target) < np.linalg.norm(self.pos - self.target):
            reward += 0.1

        self.moves_left -= 1
        if self.moves_left == 0:
            truncated = True

        reward -= 0.2

        self.pos = next_pos

        # randomise yaw -> learns to move towards target regardless of yaw 
        self.yaw += 0.1 #np.random.rand() * 2 * np.pi #(self.yaw + 0.1) % (2 * np.pi)

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        return np.concatenate((self.target / 10, self.pos / 10, [np.sin(self.yaw)], [np.cos(self.yaw)]))

    def render(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = DroneEnv()
    env.reset()
    print(env.get_observation())