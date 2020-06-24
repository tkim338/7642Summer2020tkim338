import numpy
import gym
import time
import Box2D
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import math

class SampleClass:
    def __init__(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done

class ReplayMemory:
    memory = []
    capacity = 0
    def __init__(self, capacity):
        self.capacity = capacity
    def save(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, next_state, reward, done))
    def sample(self, sample_sz):
        return random.sample(self.memory, sample_sz)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Inputs = env.action_space, Outputs = 500, Hidden = 300
        self.fc1 = nn.Linear(500, 300)
        self.fc2 = nn.Linear(300, 4)
    def forward(self, state):
        x = functional.relu(self.fc1(state))
        return functional.relu(self.fc2(x))

def get_action(full_state, pol_nn, eps):
    a = numpy.random.random()
    if a < eps:
        return random.choice([0,1,2,3])
    else:
        return pol_nn(full_state).max(1)[1].view(1,1)

def update_policy(mem, pol_nn, targ_nn, optmzr):
    sample_sz = 1
    if len(mem.memory) > sample_sz:
        sample = mem.sample(sample_sz)
        sample_tuple = SampleClass(*zip(*sample)) # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343

    return pol_nn


def train(num_episodes, epsilon, policy, target, replay_memory, optmzr):
    reward_history = []
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        epsilon *= 0.99
        done = False
        for n in range(500):
            if ep > 0:
                env.render()
            if abs(state[0]) > 0.45:
                ep_reward -= 100
                break
            if done:
                break
            action = get_action(state, policy, epsilon)
            next_state, reward, done, info = env.step(action)
            replay_memory.save(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            policy = update_policy(replay_memory, policy, target, optmzr)

        reward_history.append(ep_reward)
        if len(reward_history) > 100:
            average_reward = numpy.mean(reward_history[-100:])
            if average_reward > 200:
                break
        else:
            average_reward = numpy.mean(reward_history)
        print('Episode reward: ', ep_reward, '; Mean (100) reward: ', average_reward)

def test(model):
    return

# input params
sample_size = 100
gamma = 0.999
initial_epsilon = 0.99
episodes = 1000

# setup
env = gym.make('LunarLander-v2').unwrapped
policy_nn = DQN().to('cpu')
target_nn = DQN().to('cpu')
target_nn.load_state_dict(policy_nn.state_dict())
target_nn.eval()

optimizer = optim.Adam(policy_nn.parameters())
memory = ReplayMemory(10000)

print('Starting...')
train(episodes, initial_epsilon, policy_nn, target_nn, memory, optimizer)