from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

#from catch_ball import CatchBall
from slipstream import Slipstream
from dqn_agent import DQNAgent


def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


def animate(step):
    global win, lose
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()

        # for log
        # if reward_t == 1:
        #     win += 1
        # elif reward_t == -1:
        #     lose += 1

        print("reward = %f"%reward_t)

    else:
        state_t = state_t_1

        # execute action in environment
        action_t = agent.select_action(state_t, 0.0)
        env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, terminal = env.observe()

    # animate
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    input_size = np.zeros(2, dtype=np.int32)
    input_size[0] = 16
    input_size[1] = 16
    #env = CatchBall(input_size)
    env = Slipstream(input_size)
    agent = DQNAgent(env.enable_actions, env.name, input_size)
    agent.load_model(args.model_path)

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    # animate
    fig = plt.figure(figsize=(env.screen_n_y / 2, env.screen_n_x / 2))
    fig.canvas.set_window_title("{}-{}".format(env.name, agent.name))
    img = plt.imshow(state_t_1, interpolation="none", cmap="jet")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / env.frame_rate), blit=True)

    if args.save:
        # save animation (requires ImageMagick)
        ani_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp", "demo-{}.gif".format(env.name))
        ani.save(ani_path, writer="imagemagick", fps=env.frame_rate)
    else:
        # show animation
        plt.show()
