from keras import *
from keras.activations import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
import numpy
import random
import gym
import tensorflow
import csv

class ReplayMemory:
	def __init__(self, cap):
		self.capacity = cap
		self.memory = []
	def store_transition(self, trans):
		if len(self.memory) < self.capacity:
			self.memory.append(trans)
	def get_sample(self, num):
		if len(self.memory) >= num:
			return self.parse_samples(random.sample(self.memory, num))
		else:
			return self.parse_samples(self.memory)
	def parse_samples(self, samples):
		states = numpy.array([sample.state for sample in samples])
		actions = numpy.array([sample.action for sample in samples])
		rewards = numpy.array([sample.reward for sample in samples])
		next_states = numpy.array([sample.next_state for sample in samples])
		done = numpy.array([sample.done for sample in samples])
		return (states, actions, rewards, next_states, done)

class Transition:
	def __init__(self, state, action, reward, next_state, done):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.done = done

class LunarLanderLearner:

	def __init__(self):
		self.replay_memory = ReplayMemory(500000)
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01
		self.alpha = 0.001
		self.gamma = 0.99
		self.env = gym.make('LunarLander-v2').unwrapped
		self.actions = [0, 1, 2, 3]
		self.state_dimensions = 8
		self.reward_history = []
		self.state = None
		self.steps = 500
		self.policy_net = self.init_policy()
		self.batch_size = 50

	def init_policy(self):
		policy = Sequential()
		policy.add(Dense(400, input_dim=self.state_dimensions, activation=relu))
		policy.add(Dense(200, activation=relu))
		policy.add(Dense(100, activation=relu))
		policy.add(Dense(len(self.actions), activation=linear))
		policy.compile(loss=mean_squared_error, optimizer=Adam(lr=self.alpha))
		return policy

	def get_action(self):
		e = random.random()
		if e < self.epsilon:
			return random.choice(self.actions)
		else:
			return self.optimal_action()

	def optimal_action(self):
		actions = self.policy_net.predict(numpy.array([self.state]))
		return numpy.argmax(actions)

	def gradient_descent(self, sample_batch): # (states, actions, rewards, next_states, done's)
		if len(sample_batch[0]) < self.batch_size:
			return

		new_policies = sample_batch[2] + self.gamma * (1 - sample_batch[4]) * numpy.amax(self.policy_net.predict_on_batch(sample_batch[3]), axis=1)
		batch_policy = self.policy_net.predict_on_batch(sample_batch[0])
		batch_policy[[range(len(sample_batch[0]))], sample_batch[1]] = new_policies

		self.policy_net.fit(sample_batch[0], batch_policy, verbose=False)

	def train(self, num_episodes):

		for ep in range(num_episodes):
			self.state = self.env.reset()
			episode_reward = 0

			for step in range(self.steps):
				if ep > 1000:
					self.env.render()
				action = self.get_action()
				next_state, reward, done, info = self.env.step(action)
				trans = Transition(self.state, action, reward, next_state, done)
				self.replay_memory.store_transition(trans)
				episode_reward += reward
				self.state = next_state
				sample_batch = self.replay_memory.get_sample(self.batch_size)
				if step % 10 == 0:
					self.gradient_descent(sample_batch)
				if done:
					break
			self.reward_history.append(episode_reward)
			self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
			avg_reward = numpy.mean(self.reward_history[-100:])
			print('Episode:',ep,'Epsilon:',self.epsilon,'Reward:',episode_reward,'Avg Reward',avg_reward)
			if avg_reward > 200:
				break


tensorflow.compat.v1.disable_eager_execution()

LLL = LunarLanderLearner()
LLL.train(1000)

with open("reward_history.csv", "w", newline="") as f:
	writer = csv.writer(f)
	for r in LLL.reward_history:
		writer.writerow([r])