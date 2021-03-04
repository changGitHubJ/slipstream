import numpy as np
import sys

from slipstream import Slipstream
from dqn_agent import DQNAgent

def main(trial):
    # parameters
    n_epochs = 50000

    # environment, agent
    env = Slipstream(plot=False)
    agent1 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelMcEｗan" + str(trial))
    agent1.compile()
    agent2 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelPetacchi" + str(trial))
    agent2.compile()
    agent3 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCavendish" + str(trial))
    agent3.compile()
    agent4 = DQNAgent([0, 1, 2, 3, 4], [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCancellara" + str(trial))
    agent4.compile()

    training_log = []
    win1 = 0
    win2 = 0
    win3 = 0
    win4 = 0
    for e in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        show = False
        if e%200 == 0:
            show = True
        state_t_1, reward_t, terminal = env.observe(show=show)

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t_1 = agent1.select_action(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent1.exploration)
            action_t_2 = agent2.select_action(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent2.exploration)
            action_t_3 = agent3.select_action(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent3.exploration)
            action_t_4 = agent4.select_action(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent4.exploration)
            env.step([action_t_1, action_t_2, action_t_3, action_t_4])

            # observe environment
            state_t_1, reward_t, terminal = env.observe(show=show)

            # store experience
            agent1.store_experience(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), action_t_1, reward_t[0], state_t_1.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), terminal)
            agent2.store_experience(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), action_t_2, reward_t[1], state_t_1.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), terminal)
            agent3.store_experience(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), action_t_3, reward_t[2], state_t_1.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), terminal)
            agent4.store_experience(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), action_t_4, reward_t[3], state_t_1.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), terminal)

            if terminal:
                # experience replay
                if reward_t[0] > 0:
                    agent1.experience_replay()
                elif reward_t[1] > 0:
                    agent2.experience_replay()
                elif reward_t[2] > 0:
                    agent3.experience_replay()
                elif reward_t[3] > 0:
                    agent4.experience_replay()

            # for log
            frame += 1
            loss += agent1.current_loss
            Q_max += np.max(agent1.Q_values(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time)))
            
        REWARD = reward_t
        if REWARD[0] > 0:
            win1 += 1
        elif REWARD[1] > 0:
            win2 += 1
        elif REWARD[2] > 0:
            win3 += 1
        elif REWARD[3] > 0:
            win4 += 1
        msg = "EPOCH: {:03d}/{:03d} | REWARD: {:d},{:d},{:d},{:d} | WIN(p): {:.3f},{:.3f},{:.3f},{:.3f} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(e, n_epochs - 1, REWARD[0], REWARD[1], REWARD[2], REWARD[3], win1 / (e + 1), win2 / (e + 1), win3 / (e + 1), win4 / (e + 1), loss / frame, Q_max / frame)
        print(msg)
        training_log.append(msg + "\n")

        # save model
        if (e + 1)%5000== 0 and e != 0:
            agent1.save_model(e)
            agent2.save_model(e)
            agent3.save_model(e)
            agent4.save_model(e)

    # save log
    with open("./training_log" + str(trial) + ".txt", "w") as f:
        for log in training_log:
            f.write(log)

def test():
    trial = 0
    main(trial)

if __name__ == "__main__":
    args = sys.argv
    trial = int(args[1])
    main(trial)

    #test()
    