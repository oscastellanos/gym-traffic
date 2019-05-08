import gym
import gym_traffic
import numpy as np
import tensorflow as tf
import random
import gym.wrappers

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def store_transition(self, obs0, act, rwd, obs1, done):
        data = (obs0, act, rwd, obs1, done)
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs0, act, rwd, obs1, done = map(np.stack, zip(*batch))
        return obs0, act, rwd, obs1, done

    def print(self):
        print(self.buffer)


class QValueNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 10, tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            value = tf.layers.dense(h1, self.act_dim,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value

    def get_q_value(self, obs, reuse=False):
        q_value = self.step(obs, reuse)
        return q_value


class DQN(object):
    def __init__(self, act_dim, obs_dim, lr_q_value, gamma, epsilon):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_q_value = lr_q_value
        self.gamma = gamma
        self.epsilon = epsilon

        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.RWD = tf.placeholder(tf.float32, [None], name="reward")
        self.TARGET_Q = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.DONE = tf.placeholder(tf.float32, [None], name="done")

        q_value = QValueNetwork(self.act_dim, 'q_value')
        target_q_value = QValueNetwork(self.act_dim, 'target_q_value')
        self.memory = ReplayBuffer(capacity=int(1e6))

        self.q_value0 = q_value.get_q_value(self.OBS0)

        self.action_onehot = tf.one_hot(self.ACT, self.act_dim, dtype=tf.float32)
        self.q_value_onehot = tf.reduce_sum(tf.multiply(self.q_value0, self.action_onehot), axis=1)

        self.target_q_value1 = self.RWD + (1. - self.DONE) * self.gamma \
                               * tf.reduce_max(target_q_value.get_q_value(self.OBS1), axis=1)

        q_value_loss = tf.reduce_mean(tf.square(self.q_value_onehot - self.TARGET_Q))
        self.q_value_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_q_value).minimize(q_value_loss)

        self.q_value_params = tf.global_variables('q_value')
        self.target_q_value_params = tf.global_variables('target_q_value')
        self.target_updates = [tf.assign(tq, q) for tq, q in zip(self.target_q_value_params, self.q_value_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_updates)

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action = self.sess.run(self.q_value0, feed_dict={self.OBS0: obs})
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(action, axis=1)[0]
        return action

    def learn(self):
        obs0, act, rwd, obs1, done = self.memory.sample(batch_size=10)
        print(self.memory)
        with tf.variable_scope('target'):
            target_q_value1 = self.sess.run(self.target_q_value1,
                                        feed_dict={self.OBS1: obs1, self.RWD: rwd, self.DONE: np.float32(done)})
            self.variable_summaries(target_q_value1)
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./summaries", sess.graph)
        self.sess.run(self.q_value_train_op,feed_dict={self.OBS0: obs0, self.ACT: act,
                                                       self.TARGET_Q: target_q_value1})
        summary = self.sess.run(merged)
        train_writer.add_summary(summary)
        self.sess.run(self.target_updates)

def phase(env, action, frames):
    t_step = 0
    for _ in range(frames):
        next_state, reward_current, done, _ = env.step(action)
        t_step += 1
    return next_state, reward_current, done, _, t_step

if __name__ == "__main__":
    env = gym.make('traffic-v1')
    #env = gym.wrappers.Monitor(env, "dqn")
    env.seed(1)
    env = env.unwrapped

    agent = DQN(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[1],
                lr_q_value=0.02, gamma=0.999, epsilon=0.3)

    nepisode = 1000
    iteration = 0

    epsilon_step = 10
    epsilon_decay = 0.99
    epsilon_min = 0.001
    
    episodes = 2000

    for e in range(episodes):
        total_steps = 0
        obs0, reward_previous, don, signal = env.reset()

        reward_current = 0
        total_reward = reward_previous - reward_current

        if (signal == 0):
            status = 0
        elif (signal == 1):
            status = 1
        next_state = obs0

        while total_steps < 10000:
            env.render()
            action = agent.step(next_state)
            #print(next_state)
            #action = env.action_space.sample()
            #obs1, reward_previous, done, _ = env.step(action)
            
            if (status == 0 and action == 0):
                print("Status is: 0. Action is 0.")
                status = 0
                next_state, reward_current, done, _, t_step= phase(env, 0, 15)
                
                total_steps += t_step
                
            elif (status == 0 and action == 1):
                print("Status is 0. Action is now 1. Switching to Status 1.")
                phase(env, 2, 25)
                #print("Action is 1. Status is 0. Lights are H-Y, V-R -> H-R, V-G")
                status = 1
                next_state, reward_current, done, _, t_step = phase(env, 1, 45)
                total_steps += t_step
                
            
            elif (status == 1 and action == 1):
                print("Status is 1. Action is 1.")
                status = 1
                next_state, reward_current, done, _, t_step = phase(env, 1, 15)
                total_steps += t_step
                    
            
            elif (status == 1 and action == 0):
                print("Status is 1. Action is now 0. Switching to Status 0.")
                phase(env, 4, 25)
                status = 0
                next_state, reward_current, done, _, t_step = phase(env, 0, 45)
                total_steps += t_step
                
                
            total_reward = reward_previous - reward_current

            agent.memory.store_transition(obs0, action, total_reward, next_state, done)
            
            if total_steps % 10 == 0:
                print("Episode " + str(e) + " Total reward: " + str(total_reward))
                #print("Total reward: " + str(total_reward))
            
            # if iteration >= 10:
            #     #print("Hello, I'm here!")
            #     agent.learn()
            #     #if iteration % epsilon_step == 0:
            #     agent.epsilon = max([agent.epsilon * 0.99, 0.001])

            iteration += 1

            if done:
                env.render()
                break
            #next_state, reward_current, done, _ = env.step()
            
            # save to replay buffer in DQN

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()
env.env.close()
