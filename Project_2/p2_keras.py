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
import pickle

class ReplayMemory:
	def __init__(self, cap):
		self.capacity = cap
		self.memory = []
	def store_transition(self, trans):
		if len(self.memory) < self.capacity:
			self.memory.append(trans)
		else:
			for i in range(len(self.memory)):
				m = self.memory[i]
				if m.reward < trans.reward:
					self.memory[i] = trans
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

	def __init__(self, alpha=0.001, gamma=0.99, batch_size=100):
		self.replay_memory = ReplayMemory(500000)
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01
		self.alpha = alpha
		self.gamma = gamma
		self.env = gym.make('LunarLander-v2').unwrapped
		self.actions = [0, 1, 2, 3]
		self.state_dimensions = 8
		self.reward_history = []
		self.state = None
		self.steps = 500
		self.policy_net = self.init_policy(alpha)
		self.batch_size = batch_size

	def init_policy(self, alpha):
		policy = Sequential()
		policy.add(Dense(500, input_dim=self.state_dimensions, activation=relu))
		policy.add(Dense(200, activation=relu))
		policy.add(Dense(50, activation=relu))
		policy.add(Dense(len(self.actions), activation=linear))
		policy.compile(loss=mean_squared_error, optimizer=Adam(lr=alpha))
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

		# non-terminal states
		new_policies = sample_batch[2] + self.gamma * (1 - sample_batch[4]) * numpy.amax(self.policy_net.predict_on_batch(sample_batch[3]), axis=1)
		# terminal states
		batch_policy = self.policy_net.predict_on_batch(sample_batch[0])
		# combine states
		batch_policy[[range(len(sample_batch[0]))], sample_batch[1]] = new_policies

		self.policy_net.fit(sample_batch[0], batch_policy, verbose=False)

	def train(self, num_episodes):
		counter = 0
		for ep in range(num_episodes):
			self.state = self.env.reset()
			episode_reward = 0

			for step in range(self.steps):
				# if ep > 1000:
				# 	self.env.render()
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
				counter += 1
			else:
				counter = 0
			if counter >= 10:
				break

	def test(self, num_episodes):
		test_history = []
		store_epsilon = self.epsilon
		self.epsilon = 0
		for ep in range(num_episodes):
			self.state = self.env.reset()
			episode_reward = 0
			for step in range(self.steps):
				# self.env.render()
				action = self.get_action()
				self.state, reward, done, info = self.env.step(action)
				episode_reward += reward
				if done:
					break
			test_history.append(episode_reward)
		self.epsilon = store_epsilon
		return test_history

	def save_training_history(self, filename):
		with open(filename, "w", newline="") as f:
			writer = csv.writer(f)
			for r in self.reward_history:
				writer.writerow([r])

	def save_testing_history(self, num_tests):
		test_history = self.test(num_tests)
		with open("testing_history.csv", "w", newline="") as f:
			writer = csv.writer(f)
			for r in test_history:
				writer.writerow([r])

def test_alpha(num):
	alphas = [0.0001, 0.01, 0.1]
	for a in alphas:
		lta = LunarLanderLearner(alpha=a)
		lta.env.seed(11)
		random.seed(11)
		lta.train(num)
		lta.save_training_history('alpha_training_'+str(a)+'.csv')

def test_gamma(num):
	gammas = [0.8, 0.9, 1]
	for g in gammas:
		ltg = LunarLanderLearner(gamma=g)
		ltg.env.seed(11)
		random.seed(11)
		ltg.train(num)
		ltg.save_training_history('gamma_training_'+str(g)+'.csv')

def test_batch_size(num):
	batch_sizes = [25, 50, 150]
	for bs in batch_sizes:
		ltbs = LunarLanderLearner(batch_size=bs)
		ltbs.env.seed(11)
		random.seed(11)
		ltbs.train(num)
		ltbs.save_training_history('batch_size_training_'+str(bs)+'.csv')

tensorflow.compat.v1.disable_eager_execution()

LLL = LunarLanderLearner()
LLL.env.seed(11)
random.seed(11)
LLL.train(1000)
LLL.save_training_history("training_history.csv")
LLL.save_testing_history(100)

test_alpha(1000)
test_gamma(1000)
test_batch_size(1000)


