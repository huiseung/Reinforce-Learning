"""
미 완성 코드!
"""



import gym

import numpy as np

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.metrics as metrics





class PG_Agent():
    def __init__(self, input_dim, output_dim, hidden_units=[32, 32]):
        """

        :param input_dim: state
        :param output_dim: policy probability for all action(state)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.actorBuild()

    def actorBuild(self):
        inputs = layers.Input(shape=(self.input_dim,), name="inputs")
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units=units, name=f"FC_{i}")(x)
            x = layers.LeakyReLU(name=f"leakyReLU_{i}")(x)
        outputs = layers.Dense(units=self.output_dim, activation='softmax', name="outputs")(x)
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.optimizer = optimizers.Adam()

    def update(self, states, actions, rewards):
        """

        :param states: 한 에프소드에서 발생한 state들, 2-D array (total_step, state_dim)
        :param actions: 한 에피소드에서 발생한 action들, 1-D array (total_step, action)
        :param rewards: 한 에피소드에서 발생한 reward들, 1-D array (total_step, reward)
        :return: void
        """
        actor_variable = self.model.trainable_variables
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(actor_variable)
            actor_loss = self.loss(states, actions, rewards)
        actor_grads = actor_tape.gradient(actor_loss, actor_variable)
        self.optimizer.apply_gradients(zip(actor_grads, actor_variable))

    def loss(self, states, actions, rewards):
        action_prob = self.model.predict(states)
        action_onehot = tf.one_hot(actions, depth=self.output_dim)
        log_action_prob = tf.math.log(tf.math.reduce_sum(action_prob*action_onehot, axis=1))
        G_list = self.compute_G(rewards, gamma=0.9)
        loss = tf.math.reduce_sum(log_action_prob*G_list)
        return loss

    def compute_G(self, rewards, gamma=0.9):
        """

        :param rewards: 1-D array
        :param gamma: float
        :return G_list: 1-D array
        """
        G_list = np.zeros_like(rewards, dtype=np.float32)
        G_t = 0
        for t in reversed(range(len(rewards))):
            G_t = G_t*gamma + rewards[t]
            G_list[t] = G_t

        return G_list



    def get_action(self, state, i):
        """

        :param state:
        :param i:
        :return:
        """
        eps = (-1/200)*i+1
        action_prob = self.model.predict(state[np.newaxis, :])
        greedy_action_prob = eps*1/2+(1-eps)*action_prob

        return np.argmax(greedy_action_prob)

def sampling(env, agent):
    """
    episode 하나를 진행시킨다.
    :param env:
    :param agent:
    :return:
    """
    states = []
    actions = []
    rewards = []
    done = False
    score = 0
    i = 0

    state = env.reset()
    while not done:
        action = agent.get_action(state, i)
        next_state, reward, done, info = env.step(action)
        score += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        i += 1

        if done:
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

    return states, actions, rewards, score


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = PG_Agent(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, hidden_units=[16, 16])

    num_episode = 3
    #for i_episode in range(num_episode):
    states, actions, rewards, score = sampling(env=env, agent=agent)
    actor_variable = agent.model.trainable_variables
    with tf.GradientTape() as actor_tape:
        actor_tape.watch(actor_variable)
        actor_loss = agent.loss(states, actions, rewards)
    #actor_grads = actor_tape.gradient(actor_loss, actor_variable)
    #print(actor_grads)

        #print(f"{i_episode+1}-episode\tsocre: {score}")
        #agent.update(states, actions, rewards)
    env.close()

