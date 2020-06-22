import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import tensorflow as tf

class ActorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.output_dim)

        self.loss_history = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class PG_Agent():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actor = ActorModel(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.01)

        self.logprob_history = []
        self.reward_history = []
        self.score_history = []


    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.actor(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        agent.logprob_history.append(-m.log_prob(action))
        action = action.data.numpy()
        return action

    def update(self, gamma=0.9):
        G_t = 0
        G_list = []
        # G_T, G_{T-1}, ... G_{1} 계산
        for r in self.reward_history[::-1]:
            G_t = r + gamma*G_t
            G_list.insert(0, G_t)

        G_list = torch.FloatTensor(G_list)
        eps = 10e-9
        G_list = (G_list-G_list.mean())/(G_list.std() + eps)
        agent.optimizer.zero_grad()
        for i in range(len(agent.logprob_history)):
            loss = agent.logprob_history[i]*G_list[i]
            loss.backward()
        agent.optimizer.step()

        self.actor.loss_history.append(loss.item())
        self.logprob_history = []
        self.reward_history = []


def sampling(env, agent, render=False):
    done = False
    state = env.reset()
    score = 0
    if render:
        env.render()
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
        agent.reward_history.append(reward)
    agent.score_history.append(score)
    env.close()
    return score


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.seed(1)
    torch.manual_seed(1)

    agent = PG_Agent(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)

    num_episode = 500
    for i_episode in range(num_episode):
        score = sampling(env, agent)
        if i_episode%50 == 0:
            print(f"Episode: {i_episode}\tScore: {score}")
        if score >= env.spec.reward_threshold:
            break
        agent.update(gamma=0.9)
    print(f"Train Finsh Score: {sampling(env, agent, render=True)}")


    window = int(num_episode/20)

    fig, ((ax1, (ax2))) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(agent.score_history).rolling(window).mean()
    rolling_std = pd.Series(agent.score_history).rolling(window).std()

    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(agent.score_history)), rolling_mean-rolling_std, rolling_mean+rolling_std,
                     color='orange', alpha=0.2)
    ax1.set_title(f"Episode - Moving Average Score({window}-episode window)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Moving Average Score")

    ax2.plot(agent.score_history)
    ax2.set_title("Episode - Score")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score")

    fig.tight_layout(pad=2)
    fig.savefig("./CartPole by PG.png")
    plt.show()
