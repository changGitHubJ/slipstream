
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
    state_t_1, reward_t, terminal = env.observe()

if __name__ == "__main__":
    # environmet, agent
    env = Slipstream()
    agent = DQNAgent(env.enable_actions, env.name)
    agent.load_model()

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    cnt = 0
    while True:
        animate(cnt)
        cnt += 1