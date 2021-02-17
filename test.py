from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#from catch_ball import CatchBall
from slipstream import Slipstream
from dqn_agent import DQNAgent

def main(trial):
    # environmet, agent
    env = Slipstream(plot=True)
    agentP = DQNAgent(env.enable_actions, [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelPlayer" + str(trial))
    agentP.load_model()
    agentC = DQNAgent(env.enable_actions, [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCompetitor" + str(trial))
    agentC.load_model()

    count_game = 0
    count_win = 0
    while True:
        state, rewardP, rewardC, terminal = env.observe()

        if terminal:
            REWARD = rewardP
            print("REWARD: %.03d"%REWARD)
            env.reset()
            if REWARD > 0:
                count_win += 1
            count_game += 1
            if count_game >= 100:
                break
        else:
            actionP = agentP.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), 0.0)
            actionC = agentC.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), 0.0)
            env.step(actionP, actionC)
    print("COUNT_WIN: %d/%d"%(count_win, count_game))

if __name__ == "__main__":
    main(14)
