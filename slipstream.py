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
        self.field_n_cols = 12
        self.screen_n_rows = self.field_n_rows
        self.n_players = 4
        self.screen_n_cols = self.field_n_cols + self.n_players
        self.max_time = 48

        # variables
        self.reset()

        # animation
        self.plot = plot
        if self.plot:
            plt.ion()
            self.fig = plt.figure(figsize=[3, 7])
        self.img_cnt = 0

    def update(self, action):
        # McEwan(energy=5)
        # |   |   | P |   |   |
        # | 1 |   | 2 |   | 3 |
        # |   |   | 0 |   |   |
        # |   |   | 4 |   |   |
        # |   |   |   |   |   |
        if action[0] == 0: # dash
            if self.player_energy[0] > 0:
                self.player_row[0] += 2
                self.player_energy[0] -= 1
                # print("splint!")
            else: # no leg
                self.player_row[0] += 1
            self.player_col[0] = self.player_col[0]
        elif action[0] == 1: # move left
            self.player_row[0] += 1
            self.player_col[0] -= 2
            if self.player_col[0] < 0:
                self.player_col[0] += self.field_n_cols
        elif action[0] == 2: # go straight
            self.player_row[0] += 1
            self.player_col[0] = self.player_col[0]
        elif action[0] == 3: # move right
            self.player_row[0] += 1
            self.player_col[0] += 2
            if self.player_col[0] > self.field_n_cols - 1:
                self.player_col[0] -= self.field_n_cols
        elif action[0] == 4: # super dash
            if self.player_energy[0] > 1:
                self.player_row[0] += 3
                self.player_energy[0] -= 2
            elif self.player_energy[0] > 0:
                self.player_row[0] += 2
                self.player_energy[0] -= 1
            else: # no leg
                self.player_row[0] += 1
            self.player_col[0] = self.player_col[0]
        else:
            # do nothing
            pass

        # Petacchi(energy=6)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   | 4 |   |   |
        # |   |   |   |   |   |
        if action[1] == 0: # dash
            if self.player_energy[1] > 0:
                self.player_row[1] += 2
                self.player_energy[1] -= 1
            else: # no leg
                self.player_row[1] += 1
            self.player_col[1] = self.player_col[1]
        elif action[1] == 1: # move left
            self.player_row[1] += 1
            self.player_col[1] -= 1
            if self.player_col[1] < 0:
                self.player_col[1] += self.field_n_cols
        elif action[1] == 2: # go straight
            self.player_row[1] += 1
            self.player_col[1] = self.player_col[1]
        elif action[1] == 3: # move right
            self.player_row[1] += 1
            self.player_col[1] += 1
            if self.player_col[1] > self.field_n_cols - 1:
                self.player_col[1] -= self.field_n_cols
        elif action[1] == 4: # super dash
            if self.player_energy[1] > 1:
                self.player_row[1] += 3
                self.player_energy[1] -= 2
            elif self.player_energy[1] > 0:
                self.player_row[1] += 2
                self.player_energy[1] -= 1
            else: # no leg
                self.player_row[1] += 1
            self.player_col[1] = self.player_col[1]
        else:
            # do nothing
            pass

        # Cavendish(energy=5)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   |   |   |   |
        # |   |   | 4 |   |   |
        if action[2] == 0: # dash
            if self.player_energy[2] > 0:
                self.player_row[2] += 2
                self.player_energy[2] -= 1
                # print("splint!")
            else: # no leg
                self.player_row[2] += 1
            self.player_col[2] = self.player_col[2]
        elif action[2] == 1: # move left
            self.player_row[2] += 1
            self.player_col[2] -= 1
            if self.player_col[2] < 0:
                self.player_col[2] += self.field_n_cols
        elif action[2] == 2: # go straight
            self.player_row[2] += 1
            self.player_col[2] = self.player_col[2]
        elif action[2] == 3: # move right
            self.player_row[2] += 1
            self.player_col[2] += 1
            if self.player_col[2] > self.field_n_cols - 1:
                self.player_col[2] -= self.field_n_cols
        elif action[2] == 4: # super dash
            if self.player_energy[2] > 2:
                self.player_row[2] += 4
                self.player_energy[2] -= 3
            elif self.player_energy[2] > 1:
                self.player_row[2] += 3
                self.player_energy[2] -= 2
            elif self.player_energy[2] > 0:
                self.player_row[2] += 2
                self.player_energy[2] -= 1
            else: # no leg
                self.player_row[2] += 1
            self.player_col[2] = self.player_col[2]
        else:
            # do nothing
            pass

        # CCancellara(energy=8)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   |   |   |   |
        # |   |   |   |   |   |
        if action[3] == 0 or action[3] == 4: # dash
            if self.player_energy[3] > 0:
                self.player_row[3] += 2
                self.player_energy[3] -= 1
            else: # no leg
                self.player_row[3] += 1
            self.player_col[3] = self.player_col[3]
        elif action[3] == 1: # move left
            self.player_row[3] += 1
            self.player_col[3] -= 1
            if self.player_col[3] < 0:
                self.player_col[3] += self.field_n_cols
        elif action[3] == 2: # go straight
            self.player_row[3] += 1
            self.player_col[3] = self.player_col[3]
        elif action[3] == 3: # move right
            self.player_row[3] += 1
            self.player_col[3] += 1
            if self.player_col[3] > self.field_n_cols - 1:
                self.player_col[3] -= self.field_n_cols
        else:
            # do nothing
            pass

        # slip stream
        for i in range(self.n_players):
            recover1 = False
            recover2 = False
            for j in range(self.n_players):
                if i != j:
                    if self.player_row[i] == self.player_row[j] - 1 and abs(self.player_col[i] - self.player_col[j]) <= 1:
                        recover2 = True
                        break
                    elif self.player_row[i] == self.player_row[j] - 2 and abs(self.player_col[i] - self.player_col[j]) <= 1:
                        recover1 = True
            if recover2:
                self.player_energy[i] += 2
            elif recover1:
                self.player_energy[i] += 1

        # collision detection
        self.terminal = False
        for i in range(self.n_players):
            if self.player_row[i] >= self.field_n_rows - 1:
                self.terminal = True
        if self.terminal:
            max_value = max(self.player_row)
            drawn = False
            for i in range(self.n_players):
                if self.player_row[i] == max_value:
                    for j in range(i + 1, self.n_players): # check even
                        if self.player_row[j] == max_value:
                            drawn = True
                            break
                    if not drawn:
                        self.reward[i] = 1.0
                    break


        # update time
        if self.time < self.max_time - 1:
            self.time += 1

    def draw(self):
        # try:
        for i in range(self.n_players):
            # draw player
            if self.player_row[i] > self.field_n_rows - 1:
                self.player_row[i] = self.field_n_rows - 1
            self.screen[self.player_row[i], self.player_col[i], self.time] = 1.0 - i*0.2

            # draw energy consumption
            if self.player_energy[i] > 0:
                val = min(self.player_energy[i], self.screen_n_rows - 1)
                for j in range(val):
                    self.screen[j, self.screen_n_cols - self.n_players + i, self.time] = 1.0 - i*0.2
        # except:
        #     print("exception: %d, %d, %d, %d"%(self.player_col[0], self.player_col[1], self.player_col[2], self.player_col[3]))

    def observe(self, show=False):
        self.draw()
        if self.plot and show:
            self.update_plot()
        # self.save_images()
        return self.screen, self.reward, self.terminal

    def step(self, action):
        self.update(action)
        return self.screen, self.reward, self.terminal

    def rand_ints_nodup(self, a, b, k):
        ns = []
        while len(ns) < k:
            n = random.randint(a, b)
            if not n in ns:
                ns.append(n)
        return ns

    def reset(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols, self.max_time))
        self.time = 0

        # reset player position
        self.player_row = np.zeros(self.n_players, dtype=np.int8)
        self.player_col = self.rand_ints_nodup(0, self.field_n_cols - 1, self.n_players)
        
        # reset other variables
        self.reward = np.ones(self.n_players, dtype=np.int8)*-1
        self.terminal = False
        self.player_energy = np.zeros(self.n_players, dtype=np.int8)
        self.player_energy[0] = 5
        self.player_energy[1] = 6
        self.player_energy[2] = 5
        self.player_energy[3] = 10

    def update_plot(self):
        self.fig.clear()
        plt.imshow(self.screen[:, :, self.time], cmap="jet")
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.pause(0.1)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def save_images(self):
        plt.imshow(self.screen[:, :, self.time], cmap="jet")
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        #plt.show()
        plt.savefig("./gif11/%03d.png"%self.img_cnt)
        self.img_cnt += 1
