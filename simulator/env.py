import numpy as np
import gymnasium as gym
from gymnasium import spaces

DRONE_MASS_KG = 0.08

actions = {
    0: np.array([1, 0, 0]),
    1: np.array([-1, 0, 0]),
    2: np.array([0, 1, 0]),
    3: np.array([0, -1, 0]),
    4: np.array([0, 0, 1]),
    5: np.array([0, 0, -1]),
}


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

    def reset(self, n_targets=5, dt=0.001, seed=None, options=None):
        self.n_targets = n_targets
        self.moves_left = 1000
        self.vel = np.array([0, 0, 0])
        self.dt = dt
        self.accel = np.array([0, 0, 0])
        self.pos = np.random.randint(-10, 11, 3)
        self.target = np.random.randint(-10, 11, 3)
        observation = self.get_observation()
        return observation, {}

    def update(self, action, n=1000):
        vel = self.vel
        pos = self.pos

        for i in range(n):
            # Calculate next acceleration by applying force to drone
            force = action * 0.001
            next_accel = force / DRONE_MASS_KG

            # Calculate next velocity with v=u+at, then update pos with s=0.5t(u+v)
            next_vel = vel + next_accel * self.dt
            pos = pos + 0.5 * self.dt * (vel + next_vel)
            vel = next_vel

        return vel, pos

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        next_vel, next_pos = self.update(action)

        if np.linalg.norm(self.pos - self.target) > np.linalg.norm(next_pos - self.target):
            reward += 0.05

        if np.linalg.norm(next_pos - self.target) < 1:
            reward += 1.0
            self.target = np.random.randint(-10, 11, 3)
            self.n_targets -= 1
            if self.n_targets == 0:
                terminated = True

        self.pos = next_pos
        self.vel = next_vel

        self.moves_left -= 1
        if self.moves_left == 0:
            truncated = True

        reward -= 0.1

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        return np.concatenate((self.pos / 10, self.vel, self.target / 10))

    def render(self):
        pass

    def close(self):
        pass
