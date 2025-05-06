# type: ignore
import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean import torch


try:
    from simulator.cy_env import CyDrone
    from simulator.policy import DronePolicy
except ModuleNotFoundError:
    from cy_env import CyDrone
    from policy import DronePolicy


torch.DronePolicy = DronePolicy


def env_creator(_):
    return Drone


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
            shape=(17,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
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
        self.export_vars()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.export_vars()

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

    def export_vars(self):
        # Make properties available which are needed for the visualisation
        if self.num_agents == 1:
            self.pos = self.c_envs.pos
            self.move_target = self.c_envs.move_target
            self.look_target = self.c_envs.look_target
            self.roll = self.c_envs.roll
            self.pitch = self.c_envs.pitch
            self.yaw = self.c_envs.yaw

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
