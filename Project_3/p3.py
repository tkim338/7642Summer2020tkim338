import numpy as np
import random
import pulp
import csv

def grid_key(grid_list):
	return tuple(grid_list[0]), tuple(grid_list[1])

focus_state = grid_key([[0.0, 'Bb', 'A', 0.0], [0.0, 0.0, 0.0, 0.0]])
focus_action_A = 4  # stick
focus_action_B = 1  # South
focus_q = []

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
		if grid_key(self.grid) == focus_state and A_action == focus_action_A and B_action == focus_action_B:
			focus_q.append(self.A.q[grid_key(self.grid)][A_action][B_action])
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
		self.alpha = 0.5
		self.gamma = 0.8

		self.prev_state = None
		self.prev_self_action = None
		self.prev_opponent_action = None

		self.q_history = []
		self.epsilon = 0.5

		# self.first_action = True
		self.prev_score = 0

		self.temp = []

	def toString(self):
		if self.hasBall:
			return str(self.name + 'b')
		else:
			return str(self.name)

	def update(self, game_state, action, opponent_action, reward):
		if game_state not in self.q:
			self.q[game_state] = np.zeros([5, 5]).tolist()
			self.q[game_state][action][opponent_action] = reward

		if self.prev_state:
			s = self.prev_state
			a1 = self.prev_self_action
			a2 = self.prev_opponent_action
			v = self.q[game_state][action][opponent_action]
			# prev_score = self.prev_score
			# new_q = (1 - self.alpha) * self.q[s][a1][a2] + self.alpha * ((1 - self.gamma) * prev_score + self.gamma * v)
			# new_q = (1 - self.alpha) * self.q[s][a1][a2] + self.alpha * (score + self.gamma * v)
			# q_table[prev_ob][prev_action] += alpha * (rew + gamma * q_table[ob][next_action] - q_table[prev_ob][prev_action])
			new_q = self.q[s][a1][a2] + self.alpha * (reward + self.gamma * v - self.q[s][a1][a2])

			if a1 == focus_action_A and a2 == focus_action_B and s == focus_state:
				self.q_history.append(abs(self.q[s][a1][a2] - new_q))
				self.temp.append(new_q)

			self.q[s][a1][a2] = new_q

		self.prev_state = game_state
		self.prev_self_action = action
		self.prev_opponent_action = opponent_action
		# self.prev_score = score

	def getAction(self, method, game_state):
		# if self.first_action:
		# 	self.first_action = False
		# 	return 1, 4
		if method == 'random':
			return random.choice(self.actions)
		# elif method == 'correlated-q':
		# elif method == 'foe-q':
		elif method == 'friend-q':
			return self.nash_friend_q(game_state)
		elif method == 'q-learning':
			return self.q_learning(game_state)

	# def foe_q_action(self):
	#
	def nash_friend_q(self, game_state):
		max_reward = -np.inf
		best_self_action = random.choice(self.actions)
		if game_state in self.q and random.random() > self.epsilon:
			action_pairs = self.q[game_state]
			for self_action in range(len(action_pairs)):
				opponent_action = action_pairs[self_action].index(max(action_pairs[self_action]))
				if action_pairs[self_action][opponent_action] > max_reward:
					max_reward = action_pairs[self_action][opponent_action]
					best_self_action = self_action
		self.alpha = max(0.0001, 0.9999*self.alpha)
		return best_self_action

#
#
# def nash_foe_q(self):


def run_random_games(num, s):
	# s = Soccer()
	s.reset()
	for i in range(num):
		while not s.done:
			# s.render()
			A_action = s.A.getAction('random', grid_key(s.grid))
			B_action = s.B.getAction('random', grid_key(s.grid))
			s.processTurn(A_action, B_action)
		s.render()
		print('=====================================')
		s.reset()

	with open("random_q_history.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for r in s.A.q_history:
			writer.writerow([r])

	print('done')
	return s


def run_friend_games(num, s):
	# s = Soccer()
	s.reset()
	for i in range(num):
		# A_action = focus_action_A
		# B_action = focus_action_B
		# s.processTurn(A_action, B_action)
		while not s.done:
			s.render()
			A_action = s.A.getAction('friend-q', grid_key(s.grid))
			B_action = s.B.getAction('friend-q', grid_key(s.grid))
			s.processTurn(A_action, B_action)
		s.render()
		print('=====================================')
		s.reset()

	with open("friend_q_history.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for r in s.A.q_history:
			writer.writerow([r])

	with open("temp.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for r in s.A.temp:
			writer.writerow([r])

	print('done')
	return s


# s = Soccer()
# s.render()
#
# s.processTurn(4, 2) # down
# s.render()
# s.processTurn(2, 4) # right
# s.render()
# s.processTurn(4, 4) # right
# s.render()

# print(s.grid)
# print(tuple(s.grid))
# print(s.done)

soc = Soccer()
soc = run_random_games(100000, soc)
# soc.A.q_history = []
# soc.A.temp = []
# soc = run_friend_games(100000, soc)
