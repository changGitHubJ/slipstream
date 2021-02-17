import matplotlib.pyplot as plt
import numpy as np
import os
import random

from math import pi

class Slipstream:
    def __init__(self, plot=False):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.field_n_rows = 32
        self.field_n_cols = 8
        self.screen_n_rows = self.field_n_rows
        self.screen_n_cols = self.field_n_cols + 2
        self.max_time = 32
        self.enable_actions = (0, 1, 2, 3)
        # self.frame_rate = 2

        # variables
        self.reset()

        # animation
        self.plot = plot
        if self.plot:
            plt.ion()
            self.fig = plt.figure()
        self.img_cnt = 0

    def update(self, actionP, actionC):
        """
        action:
            0: dash
            1: move left
            2: go straight
            3: move right
        """
        # update player position
        if actionP == self.enable_actions[0]: # dash
            if self.player_energy > 0:
                self.player_row += 2
                self.player_energy -= 1
                print("splint!")
            else: # no leg
                self.player_row += 1
            self.player_col = self.player_col
        elif actionP == self.enable_actions[1]: # move left
            if self.player_col == 0:
                self.player_row += 1
                self.player_col = self.field_n_cols - 2
            elif self.player_col == 1:
                self.player_row += 1
                self.player_col = self.field_n_cols - 1
            else: # self.player_col >= 2:
                self.player_row += 1
                self.player_col -= 2
        elif actionP == self.enable_actions[2]: # go straight
            self.player_row += 1
            self.player_col = self.player_col
        elif actionP == self.enable_actions[3]: # move right
            if self.player_col == self.field_n_cols - 1:
                self.player_row += 1
                self.player_col = 1
            elif self.player_col == self.field_n_cols - 2:
                self.player_row += 1
                self.player_col = 0
            else: # self.player_col <= self.field_n_cols - 3:
                self.player_row += 1
                self.player_col += 2 
        else:
            # do nothing
            pass

        # update competitor position
        if actionC == 0: # dash
            if self.comptr_energy > 0:
                self.comptr_row += 2
                self.comptr_energy -= 1
            else: # no leg
                self.comptr_row += 1
            self.comptr_col = self.comptr_col
        elif actionC == 1: # move left
            if self.comptr_col == 0:
                self.comptr_row += 1
                self.comptr_col = self.field_n_cols - 1
            else: # self.comptr_col > 0:
                self.comptr_row += 1
                self.comptr_col -= 1
        elif actionC == 2: # go straight
            self.comptr_row += 1
            self.comptr_col = self.comptr_col
        elif actionC == 3: # move right
            if self.comptr_col == self.field_n_cols - 1:
                self.comptr_row += 1
                self.comptr_col = 0
            else: # self.comptr_col < self.field_n_cols - 1
                self.comptr_row += 1
                self.comptr_col += 1
            
        else:
            # do nothing
            pass

        # slip stream
        if self.player_row == self.comptr_row - 1 and abs(self.player_col - self.comptr_col) <= 1:
            self.player_energy += 2
            print("recover 2")
        elif self.player_row == self.comptr_row - 2 and abs(self.player_col - self.comptr_col) <= 1:
            self.player_energy += 1
            print("recover 1")
        if self.comptr_row == self.player_row - 1 and abs(self.comptr_col - self.player_col) <= 1:
            self.comptr_energy += 2
        elif self.comptr_row == self.player_row - 2 and abs(self.comptr_col - self.player_col) <= 1:
            self.comptr_energy += 1

        # collision detection
        self.reward = 0
        self.terminal = False
        if self.player_row >= self.field_n_rows - 1 or self.comptr_row >= self.field_n_rows - 1:
            self.terminal = True
            if self.player_row > self.comptr_row: # win
                self.rewardP = 1
                self.rewardC = -1
            elif self.player_row < self.comptr_row: # defeated
                self.rewardP = - 1
                self.rewardC = 1
            else: # draw
                self.rewardP = -1
                self.rewardC = -1

        # update time
        if self.time < self.max_time - 1:
            self.time += 1
        else:
            self.terminal = True
            self.rewardP = -1
            self.rewardC = -1

    def draw(self):
        # draw player
        if self.player_row > self.field_n_rows - 1:
            self.player_row = self.field_n_rows - 1
        self.screen[self.player_row, self.player_col, self.time] = 1

        # draw competitor
        if self.comptr_row > self.field_n_rows - 1:
            self.comptr_row = self.field_n_rows - 1
        self.screen[self.comptr_row, self.comptr_col, self.time] = 0.5

        # draw energy consumption
        if self.player_energy > 0:
            val = min(self.player_energy, self.screen_n_rows - 1)
            for i in range(val):
                self.screen[i, self.screen_n_cols - 2, self.time] = 0.75
        if self.comptr_energy > 0:
            val = min(self.comptr_energy, self.screen_n_rows - 1)
            for i in range(val):
                self.screen[i, self.screen_n_cols - 1, self.time] = 0.25

    def observe(self):
        self.draw()
        if self.plot:
            self.update_plot()
        # self.save_images()
        return self.screen, self.rewardP, self.rewardC, self.terminal

    def step(self, actionP, actionC):
        self.update(actionP, actionC)
        return self.screen, self.rewardP, self.rewardC, self.terminal

    def reset(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols, self.max_time))
        self.time = 0

        # reset player position
        self.player_row = 0
        self.player_col = np.random.randint(self.field_n_cols)

        # reset competitor position
        self.comptr_row = 0
        self.comptr_col = np.random.randint(self.field_n_cols)

        # reset other variables
        self.rewardP = 0
        self.rewardC = 0
        self.terminal = False
        self.player_energy = 5
        self.comptr_energy = 6

    def update_plot(self):
        self.fig.clear()
        plt.imshow(self.screen[:, :, self.time])
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.pause(0.1)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def save_images(self):
        plt.imshow(self.screen[:, :, self.time])
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        #plt.show()
        plt.savefig("./gif14/%03d.png"%self.img_cnt)
        self.img_cnt += 1
