from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import numpy as np
from time import perf_counter

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


# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        self.pos = np.array([0,0,0], dtype=np.float64)
        self.vel = np.array([0,0,0], dtype=np.float64)



        self.env = DroneEnv()
        self.obs, _ = self.env.reset(n_targets=1, colliders=mesh("flat.ply"))

        self.model = Agent(None)
        self.model.load_state_dict(torch.load("end.pth"))

        self.paused = False

        self.fig = plt.figure("Drone Simulation Tool for Warwick AI")
        self.ax = self.fig.add_subplot(projection="3d")
        _ani = FuncAnimation(self.fig, self.update, frames=1000, interval=10)

        cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()
    
    def connect(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)
    
    # Call it always before finishing. To deallocate resources.
    def disconnect(self):
        self.tello.end()

    def update(self):

        t = perf_counter()
        new_t = perf_counter()
        dt = new_t - t

        ax = self.tello.get_acceleration_x()
        ay = self.tello.get_acceleration_y()
        az = self.tello.get_acceleration_z()

        dx = self.tello.get_speed_x()
        dy = self.tello.get_speed_y()
        dz = self.tello.get_speed_z()

        print("vels: ", dx, dy, dz)
        print("accs: ", ax, ay, az)

        print("dt: ", dt)


        self.vel += np.array([ax, ay, az], dtype=np.float64) * dt
        print("integrated vels: ", self.vel)

        t = new_t

        for event in pygame.event.get():
            if event.type == pygame.USEREVENT + 1:
                self.update()
            elif event.type == pygame.QUIT:
                should_stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    should_stop = True
                else:
                    self.keydown(event.key)
            elif event.type == pygame.KEYUP:
                self.keyup(event.key)

        time.sleep(1 / FPS)


    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        # Update routine. Send velocities to Tello.
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()