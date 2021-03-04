from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#from catch_ball import CatchBall
from slipstream import Slipstream
from dqn_agent import DQNAgent

def main(trial, epoch):
    # environmet, agent
    env = Slipstream(plot=True)
    agent1 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelMcEwan%d_%d"%(trial, epoch))
    agent1.load_model()
    agent2 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelPetacchi%d_%d"%(trial, epoch))
    agent2.load_model()
    agent3 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCavendish%d_%d"%(trial, epoch))
    agent3.load_model()
    agent4 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCancellara%d_%d"%(trial, epoch))
    #agent4.load_model()
    agent4.compile()

    count_game = 0
    win1 = 0
    win2 = 0
    win3 = 0
    win4 = 0
    while True:
        state, reward, terminal = env.observe(show=True)

        if terminal:
            REWARD = reward
            if REWARD[0] > 0:
                win1 += 1
            elif REWARD[1] > 0:
                win2 += 1
            elif REWARD[2] > 0:
                win3 += 1
            elif REWARD[3] > 0:
                win4 += 1
            count_game += 1
            msg = "REWARD: {:d},{:d},{:d},{:d} | WIN(p): {:.3f},{:.3f},{:.3f},{:.3f}".format(REWARD[0], REWARD[1], REWARD[2], REWARD[3], win1 / count_game, win2 / count_game, win3 / count_game, win4 / count_game)
            print(msg)
            env.reset()
            if count_game >= 300:
                break
        else:
            action_t_1 = agent1.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent1.exploration)
            action_t_2 = agent2.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent2.exploration)
            action_t_3 = agent3.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent3.exploration)
            action_t_4 = agent4.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent4.exploration)
            env.step([action_t_1, action_t_2, action_t_3, action_t_4])

if __name__ == "__main__":
    main(0, 49999)
