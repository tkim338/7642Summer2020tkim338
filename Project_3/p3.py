import numpy as np
import random
import pulp
import csv

def grid_key(grid_list):
	return tuple(grid_list[0]), tuple(grid_list[1])

focus_state = grid_key([[0.0, 'Bb', 'A', 0.0], [0.0, 0.0, 0.0, 0.0]])
focus_action_A = 4  # stick
focus_action_B = 1  # South
# focus_q = []
# focus_q_learning = []

V = pulp.LpVariable("V")
pulp_vars = {}
pulp_vars[0] = pulp.LpVariable('pi_up', lowBound=0)
pulp_vars[1] = pulp.LpVariable('pi_down', lowBound=0)
pulp_vars[2] = pulp.LpVariable('pi_right', lowBound=0)
pulp_vars[3] = pulp.LpVariable('pi_left', lowBound=0)
pulp_vars[4] = pulp.LpVariable('pi_stay', lowBound=0)

class Soccer:
	def __init__(self):

		self.grid = np.zeros([2, 4]).tolist()
		self.A = Player(0, 2, 'A', False, 0)
		self.B = Player(0, 1, 'B', True, 3)
		self.A.opponent = self.B
		self.B.opponent = self.A

		self.grid[self.A.y][self.A.x] = self.A.toString()
		self.grid[self.B.y][self.B.x] = self.B.toString()

		self.done = False
		self.steps = 0

	def reset(self):
		self.grid = np.zeros([2, 4]).tolist()
		self.A.x = 2
		self.A.y = 0
		self.A.hasBall = False
		self.A.first_action = True
		self.B.x = 1
		self.B.y = 0
		self.B.hasBall = True
		self.A.first_action = True

		self.grid[self.A.y][self.A.x] = self.A.toString()
		self.grid[self.B.y][self.B.x] = self.B.toString()

		self.done = False
		self.steps = 0

	def render(self):
		print(self.grid[0][:])
		print(self.grid[1][:])
		print('A score: ', self.A.score, '; B score: ', self.B.score)

	def processTurn(self, A_action, B_action):
		# if grid_key(self.grid) == focus_state and A_action == focus_action_A:
		# 	focus_q_learning.append(self.A.q[grid_key(self.grid)][A_action][B_action])
		# 	if B_action == focus_action_B:
		# 		focus_q.append(self.A.q[grid_key(self.grid)][A_action][B_action])
		temp_grid = self.grid[:]

		if np.random.random() > 0.5:
			self.processAction(self.A, temp_grid, A_action, B_action)
			self.processAction(self.B, temp_grid, B_action, A_action)
		else:
			self.processAction(self.B, temp_grid, B_action, A_action)
			self.processAction(self.A, temp_grid, A_action, B_action)

		self.steps += 1
		if self.steps > 100:
			self.done = True

	def processAction(self, player, temp_grid, action, opponent_action):
		# temp_grid = self.grid

		actions = {0: self.up, 1: self.down, 2: self.right, 3: self.left, 4: self.stay}
		actions[action](player)

		self.grid = np.zeros([2, 4]).tolist()
		self.grid[player.y][player.x] = player.toString()
		self.grid[player.opponent.y][player.opponent.x] = player.opponent.toString()

		self.done, player_scored = self.checkGoal(player)
		if self.done:
			if player_scored:
				reward = 100
			else:
				reward = -100
		else:
			reward = 0

		player.update(grid_key(temp_grid), action, opponent_action, reward)

	def up(self, player):
		if player.y > 0:
			if self.grid[player.y - 1][player.x] == 0:
				player.y -= 1
			else:
				self.checkCollision(player)
		return player

	def down(self, player):
		if player.y < 1:
			if self.grid[player.y + 1][player.x] == 0:
				player.y += 1
			else:
				self.checkCollision(player)
		return player

	def right(self, player):
		if player.x < 3:
			if self.grid[player.y][player.x + 1] == 0:
				player.x += 1
			else:
				self.checkCollision(player)

	def left(self, player):
		if player.x > 0:
			if self.grid[player.y][player.x - 1] == 0:
				player.x -= 1
			else:
				self.checkCollision(player)
		return player

	def stay(self, player):
		return

	def checkCollision(self, player):
		if player.hasBall:
			player.hasBall = False
			player.opponent.hasBall = True
		player.score -= 1
		player.opponent.score -= 1

	def checkGoal(self, player):
		if player.hasBall:
			if player.x == player.goal_x:
				player.score += 100
				player.opponent.score -= 100
				return True, True
			elif player.x == player.opponent.goal_x:
				player.score -= 100
				player.opponent.score += 100
				return True, False
		return False, False


class Player:
	def __init__(self, y, x, name, hasBall, goal_x):
		self.x = x
		self.y = y
		self.name = name
		self.hasBall = hasBall
		self.goal_x = goal_x

		self.actions = [0, 1, 2, 3, 4]  # up, down, right, left, and stick
		self.opponent = None
		self.score = 0

		self.q = {}
		self.alpha = 0.999
		self.gamma = 0.9

		self.prev_state = None
		self.prev_self_action = None
		self.prev_opponent_action = None

		self.q_history = []
		self.epsilon = 0.999

		self.ql = False

		# self.temp = []

	def toString(self):
		if self.hasBall:
			return str(self.name + 'b')
		else:
			return str(self.name)

	def update(self, game_state, action, opponent_action, reward):
		if self.ql:
			if game_state not in self.q:
				self.q[game_state] = np.zeros(5).tolist()
				self.q[game_state][action] = reward

			if self.prev_state:
				s = self.prev_state
				a1 = self.prev_self_action
				v = self.q[game_state][action]

				new_q = self.q[s][a1] + self.alpha * (reward + self.gamma * v - self.q[s][a1])
				self.q[s][a1] = new_q

		else:
			if game_state not in self.q:
				self.q[game_state] = np.zeros([5, 5]).tolist()
				self.q[game_state][action][opponent_action] = reward

			if self.prev_state:
				s = self.prev_state
				a1 = self.prev_self_action
				a2 = self.prev_opponent_action
				v = self.q[game_state][action][opponent_action]

				new_q = self.q[s][a1][a2] + self.alpha * (reward + self.gamma * v - self.q[s][a1][a2])
				self.q[s][a1][a2] = new_q

		self.prev_state = game_state
		self.prev_self_action = action
		self.prev_opponent_action = opponent_action
		self.alpha = max(0.0001, 0.9999 * self.alpha)
		self.epsilon = max(0.0001, 0.9999 * self.epsilon)

	def getAction(self, method, game_state):
		if method == 'random':
			return random.choice(self.actions)
		elif method == 'foe-q':
			return self.nash_foe_q(game_state)
		elif method == 'friend-q':
			return self.nash_friend_q(game_state)
		elif method == 'q-learning':
			return self.q_learning(game_state)
		elif method == 'correlated-q':
			return self.utilitarian_ce_q(game_state)

	def utilitarian_ce_q(self, game_state):
		max_reward = -np.inf
		best_self_action = random.choice(self.actions)
		if random.random() > self.epsilon:
			if game_state in self.q and game_state in self.opponent.q:
				action_pairs = np.array(self.q[game_state]) + np.array(self.opponent.q[game_state])
				action_pairs = action_pairs.tolist()

				for self_action in range(len(action_pairs)):
					opponent_action = action_pairs[self_action].index(max(action_pairs[self_action]))
					if action_pairs[self_action][opponent_action] > max_reward:
						max_reward = action_pairs[self_action][opponent_action]
						best_self_action = self_action

		return best_self_action

	def q_learning(self, game_state):
		max_reward = -np.inf
		best_self_action = random.choice(self.actions)
		if game_state in self.q and random.random() > self.epsilon:
			action_pairs = self.q[game_state]
			for self_action in range(len(action_pairs)):
				reward = action_pairs[self_action]
				if reward > max_reward:
					max_reward = reward
					best_self_action = self_action

		return best_self_action

	def nash_friend_q(self, game_state):
		max_reward = -np.inf
		best_self_action = random.choice(self.actions)
		best_opponent_action = random.choice(self.actions)
		if game_state in self.q and random.random() > self.epsilon:
			# if self.name == 'A':
			action_pairs = self.q[game_state]
			# else:
			# 	action_pairs = self.opponent.q[game_state]

			for self_action in range(len(action_pairs)):
				opponent_action = action_pairs[self_action].index(max(action_pairs[self_action]))
				if action_pairs[self_action][opponent_action] > max_reward:
					max_reward = action_pairs[self_action][opponent_action]
					best_self_action = self_action
					best_opponent_action = opponent_action
		# if self.name == 'A':
		return best_self_action
		# else:
		# 	return best_opponent_action

	def nash_foe_q(self, game_state):
		best_self_action = random.choice(self.actions)
		if game_state in self.q and random.random() > self.epsilon:
			action_pairs = self.q[game_state] # 5x5

			Lp_prob = pulp.LpProblem('Problem', pulp.LpMaximize)

			Lp_prob += V
			Lp_prob += pulp_vars[0]+pulp_vars[1]+pulp_vars[2]+pulp_vars[3]+pulp_vars[4] == 1
			for ap in action_pairs:
				Lp_prob += ap[0]*pulp_vars[0] + ap[1]*pulp_vars[1] + ap[2]*pulp_vars[2] + ap[3]*pulp_vars[3] + ap[4]*pulp_vars[4] >= V
			Lp_prob.solve()
			pi = []
			for i in range(5):
				pi.append(max(0, pulp.value(pulp_vars[i])))
			best_self_action = np.random.choice(self.actions, p=pi)

		return best_self_action

def run_games(num, s, policy_type):
	s.reset()
	q_history = []
	prev_q = 0
	for i in range(num):
		print(i,'/',num)
		while not s.done:
			# s.render()
			A_action = s.A.getAction(policy_type, grid_key(s.grid))
			B_action = s.B.getAction(policy_type, grid_key(s.grid))
			s.processTurn(A_action, B_action)
		# s.render()
		# print('=====================================')
		if policy_type == 'q-learning':
			q_history.append(abs(prev_q - s.A.q[focus_state][focus_action_A]))
			prev_q = s.A.q[focus_state][focus_action_A]
		else:
			q_history.append(abs(prev_q - s.A.q[focus_state][focus_action_A][focus_action_B]))
			prev_q = s.A.q[focus_state][focus_action_A][focus_action_B]
		s.reset()

	with open("./data/" + policy_type + "_history.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for r in q_history:
			writer.writerow([r])

	print('done')
	return s

def run_random_games(num, s):
	return run_games(num, s, 'random')

def run_friend_games(num, s):
	return run_games(num, s, 'friend-q')

def run_foe_games(num, s):
	return run_games(num, s, 'foe-q')

def run_q_learning_games(num, s):
	s.A.ql = True
	s.B.ql = True
	return run_games(num, s, 'q-learning')

def run_correlated_q_games(num, s):
	return run_games(num, s, 'correlated-q')

n = 100000
# run_random_games(n, Soccer())
run_friend_games(n, Soccer())
# run_foe_games(n, Soccer())
# run_q_learning_games(n, Soccer())
# run_correlated_q_games(n, Soccer())