from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#from catch_ball import CatchBall
from slipstream import Slipstream
from dqn_agent import DQNAgent

if __name__ == "__main__":
    # environmet, agent
    env = Slipstream(plot=True)
    agent = DQNAgent(env.enable_actions, [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name)
    agent.load_model()

    while True:
        state, reward, terminal = env.observe()

        if terminal:
            REWARD = reward
            print("REWARD: %.03d"%REWARD)
            env.reset()
        else:
            action = agent.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), 0.0)
            env.step(action)