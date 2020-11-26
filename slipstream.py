import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from math import sin, cos, pi, sqrt

class Slipstream:
    def __init__(self, input_size):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_x = input_size[0]
        self.screen_n_y = input_size[1]
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5
        self.full_gas = 16

        # variables
        self.reset()

    def rad2deg(self, val):
        return val/pi*180

    def deg2step(self, ang):
        dx = cos(ang)
        if dx < -0.5:
            dx = -1
        elif -0.5 <= dx and dx < 0.5:
            dx = 0
        else:
            dx = 1
        dy = sin(ang)
        if dy < -0.5:
            dy = -1
        elif -0.5 <= dy and dy < 0.5:
            dy = 0
        else:
            dy = 1
        return int(dx), int(dy)

    def update(self, action):
        """
        action:
            0: move right
            1: go straight
            2: move left
        """
        # update player position
        if action == self.enable_actions[0]: # move right
            self.p_dire = self.p_dire - pi/4
        elif action == self.enable_actions[1]: # go straight
            self.p_dire = self.p_dire
        elif action == self.enable_actions[2]: # go left
            self.p_dire = self.p_dire + pi/4
        else:
            # do nothing
            pass
        dx, dy = self.deg2step(self.p_dire)
        self.player_x += dx
        self.player_y += dy
        self.player_x = max(self.player_x, 1)
        self.player_x = min(self.player_x, self.screen_n_x - 2)
        self.player_y = max(self.player_y, 1)
        self.player_y = min(self.player_y, self.screen_n_y - 2)
        self.p_head_x, self.p_head_y = self.ang2head(self.p_dire, self.player_x, self.player_y)
        #print("action = %d, player_x = %d, player_y = %d, p_dire = %f, p_head_x = %d, p_head_y = %d"%(action, self.player_x, self.player_y, self.rad2deg(self.p_dire), self.p_head_x, self.p_head_y))

        # update competitor position
        a = random.randint(0, 2)
        if a == self.enable_actions[0]: # move right
            self.c_dire = self.c_dire - pi/4
        elif a == self.enable_actions[1]: # go straight
            self.c_dire = self.c_dire
        elif a == self.enable_actions[2]: # go left
            self.c_dire = self.c_dire + pi/4
        else:
            # do nothing
            pass
        dx, dy = self.deg2step(self.c_dire)
        self.comptr_x += dx
        self.comptr_y += dy
        self.comptr_x = max(self.comptr_x, 1)
        self.comptr_x = min(self.comptr_x, self.screen_n_x - 2)
        self.comptr_y = max(self.comptr_y, 1)
        self.comptr_y = min(self.comptr_y, self.screen_n_y - 2)
        self.c_head_x, self.c_head_y = self.ang2head(self.c_dire, self.comptr_x, self.comptr_y)

        # energy consumption
        if self.player_x != self.p_prev_x or self.player_y != self.p_prev_y:
            self.achiv += 1
            comsumption = self.energy_consumption(self.player_x, self.player_y, self.p_dire, self.comptr_x, self.comptr_y, self.c_dire)
        else:
            comsumption = 1.0
        self.energy -= comsumption

        # evaluation
        self.reward = 0
        self.terminal = False
        if self.energy < 0:
            self.terminal = True
            self.reward = self.achiv - self.full_gas
            print("Terminated. reward = %f"%self.reward)
        else:
            self.p_prev_x = self.player_x
            self.p_prev_y = self.player_y

    def energy_consumption(self, player_x, player_y, p_dire, comptr_x, comptr_y, c_dire):
        consumption = 0
        dist = sqrt((player_x - comptr_x)*(player_x - comptr_x) + (player_y - comptr_y)*(player_y - comptr_y))
        inn = cos(p_dire)*cos(c_dire) + sin(p_dire)*sin(c_dire)
        vec_x = player_x - comptr_x
        vec_y = player_y - comptr_y
        norm = sqrt(vec_x*vec_x + vec_y*vec_y)
        if norm != 0: 
            vec_x /= norm
            vec_y /= norm
            inn2 = cos(c_dire)*vec_x + sin(c_dire)*vec_y
        else:
            inn2 = -1.0
        if inn > 0.0 and inn2 < 0.0:
            if dist != 0.0:
                consumption = 1.0 - 1.0/dist #*inn
            else:
                consumption = 1.0 - 1.0 #*inn
        else:
            consumption = 1.0
        print("consumption = %f, dist = %f, inn = %f, inn2 = %f"%(consumption, dist, inn, inn2))
        return consumption


    def draw(self, debug=False):
        # reset screen
        self.screen = np.zeros((self.screen_n_y, self.screen_n_x))

        # draw player
        self.screen[(self.screen_n_y - 1) - self.player_y, self.player_x] = 1
        self.screen[(self.screen_n_y - 1) - self.p_head_y, self.p_head_x] = 0.9

        # draw ball
        self.screen[(self.screen_n_y - 1) - self.comptr_y, self.comptr_x] = 0.5
        self.screen[(self.screen_n_y - 1) - self.c_head_y, self.c_head_x] = 0.4

        if debug == True:
            plt.imshow(self.screen, cmap="jet")
            plt.axis("off")
            plt.show()

    def observe(self, debug=False):
        self.draw(debug)
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def ang2head(self, angle, x, y):
        angle_ = copy.copy(angle)
        angle_ = self.rad2deg(angle_ % (2*pi))
        head_x = 0
        head_y = 0
        if (0 <= angle_ and angle_ < 22.5) or (337.5 <= angle_ and angle_ < 360): # 0
            head_x = x + 1
            head_y = y
        elif 22.5 <= angle_ and angle_ < 67.5: # 1
            head_x = x + 1
            head_y = y + 1
        elif 67.5 <= angle_ and angle_ < 112.5: # 2
            head_x = x
            head_y = y + 1
        elif 112.5 <= angle_ and angle_ < 157.5: # 3
            head_x = x - 1
            head_y = y + 1
        elif 157.5 <= angle_ and angle_ < 202.5: # 4
            head_x = x - 1
            head_y = y
        elif 202.5 <= angle_ and angle_ < 247.7: # 5
            head_x = x - 1
            head_y = y - 1
        elif 247.5 <= angle_ and angle_ < 292.5: # 6
            head_x = x
            head_y = y - 1
        elif 292.5 <= angle_ and angle_ < 337.5: # 7
            head_x = x + 1
            head_y = y - 1
        return head_x, head_y

    def reset(self):
        # reset player position
        self.player_x = random.randint(1, self.screen_n_x - 2)
        self.player_y = random.randint(1, self.screen_n_y - 2)
        self.p_dire = random.uniform(0.0, 2*pi)
        self.p_head_x, self.p_head_y = self.ang2head(self.p_dire, self.player_x, self.player_y)
        self.p_prev_x = self.player_x
        self.p_prev_y = self.player_y
        self.energy = self.full_gas
        self.achiv = 0.0

        # reset competitor position
        self.comptr_x = random.randint(1, self.screen_n_x - 2)
        self.comptr_y = random.randint(1, self.screen_n_y - 2)
        self.c_dire = random.uniform(0.0, 2*pi)
        self.c_head_x, self.c_head_y = self.ang2head(self.c_dire, self.comptr_x, self.comptr_y)

        # reset other variables
        self.reward = 0
        self.terminal = False
