import numpy as np

from dqn_agent import DQNAgent
from slipstream import Slipstream

def animate(step):
    global win, lose
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()

        print("REWARD: %.2f"%reward_t)

    else:
        state_t = state_t_1

        # execute action in environment
        action_t = agent.select_action(state_t, 0.0)
        env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, terminal = env.observe(debug=True)

if __name__ == "__main__":
    # environmet, agent
    input_size = np.zeros(2, dtype=np.int32)
    input_size[0] = 8
    input_size[1] = 8
    env = Slipstream(input_size)
    agent = DQNAgent(env.enable_actions, env.name, input_size)
    agent.load_model()

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    cnt = 0
    while True:
        animate(cnt)
        cnt += 1