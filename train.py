import numpy as np

#from catch_ball import CatchBall
from slipstream import Slipstream
from dqn_agent import DQNAgent


if __name__ == "__main__":
    # parameters
    n_epochs = 5000

    # environment, agent
    input_size = np.zeros(2, dtype=np.int32)
    input_size[0] = 16
    input_size[1] = 16
    #env = CatchBall(input_size)
    env = Slipstream(input_size)
    agent = DQNAgent(env.enable_actions, env.name, input_size)

    # variables
    win = 0

    for e in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            # experience replay
            agent.experience_replay()

            # for log
            frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(state_t))
            if reward_t == 1:
                win += 1

        print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
            e, n_epochs - 1, win, loss / frame, Q_max / frame))

    # save model
    agent.save_model()
