import gym
import gym_traffic
#import gym.wrappers

if __name__ == "__main__":
    env = gym.make("traffic-v1")
    #env = gym.wrappers.Monitor(env, "dqn")
    episodes = 2000

    for e in range(episodes):
        total_steps = 0
        obs = env.reset()

        reward_previous = 0
        reward_current = 0
        total_reward = reward_previous - reward_current

        while total_steps < 7000:
            # for our random agent, sample from the action_space randomly.
            action = env.action_space.sample()
            obs, reward_previous, done, _ = env.step(action)
            print(obs.shape)
            
            if (action == 0):
                for i in range(15):
                    total_steps += 1
                    next_state, reward_current, done, _ = env.step(0)

                for i in range(25):
                    total_steps += 1
                    next_state, reward_current, done, _ = env.step(2)

                # for i in range(10):
                #     total_steps += 1
                #     next_state, reward_current, done, _ = env.step(1)

            if (action == 1):
                for i in range(15):
                    total_steps += 1
                    next_state, reward_current, done, _ = env.step(1)

                for i in range(25):
                    total_steps += 1
                    next_state, reward_current, done, _ = env.step(4)

                # for i in range(10):
                #     total_steps += 1
                #     next_state, reward_current, done, _ = env.step(0)

            total_reward = reward_previous - reward_current

            if total_steps % 10 == 0:
                print("Episode " + str(e) + " Total reward: " + str(total_reward))
            if done:
                break
            
            #next_state, reward_current, done, _ = env.step()
            
            # save to replay buffer in DQN

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()
env.env.close()