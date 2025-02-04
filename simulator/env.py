import numpy as np
import gymnasium
import pufferlib
from cy_env import CyDrone


class Drone(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        report_interval=1,
        buf=None,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(8,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        self.num_agents = num_envs

        self.report_interval = report_interval
        self.human_action = None
        self.tick = 0

        super().__init__(buf)

        self.c_envs = CyDrone(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs,
        )

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log["episode_length"] > 0:
                info.append(log)

        self.tick += 1
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        pass

    def close(self):
        self.c_envs.close()


def test_performance(timeout=10, atn_cache=1024):
    env = Drone(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")


if __name__ == "__main__":
    test_performance()
