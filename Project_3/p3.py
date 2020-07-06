import numpy as np
import random
import pulp
import csv

def grid_key(grid_list):
	return tuple(grid_list[0]), tuple(grid_list[1])

focus_state = grid_key([[0.0, 'Bb', 'A', 0.0], [0.0, 0.0, 0.0, 0.0]])
focus_action_A = 4  # stick
focus_action_B = 1  # South

class Soccer:
	def __init__(self):

		self.grid = np.zeros([2, 4]).tolist()
		self.A = Player(0, 2, 'A', False, 0)
		self.B = Player(0, 1, 'B', True, 3)
		self.A.opponent = self.B
		self.B.opponent = self.A

		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

		self.done = False

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

		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

		self.done = False

	def render(self):
		print(self.grid[0][:])
		print(self.grid[1][:])
		print('A score: ', self.A.score, '; B score: ', self.B.score)

	def processTurn(self, A_action, B_action):
		if np.random.random() > 0.5:
			goal_A, score_A = self.processAction(self.A, A_action)
			goal_B, score_B = self.processAction(self.B, B_action)
		else:
			goal_B, score_B = self.processAction(self.B, B_action)
			goal_A, score_A = self.processAction(self.A, A_action)

		self.A.update(grid_key(self.grid), A_action, B_action, score_A)
		self.B.update(grid_key(self.grid), B_action, A_action, score_B)

		self.grid = np.zeros([2, 4]).tolist()
		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

		if goal_A or goal_B:
			self.done = True

	def processAction(self, player, action):
		actions = {0: self.up, 1: self.down, 2: self.right, 3: self.left, 4: self.stay}
		actions[action](player)
		return self.checkGoal(player)

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
				return True, 100
			elif player.x == player.opponent.goal_x:
				player.score -= 100
				player.opponent.score += 100
				return True, -100
		return False, 0


class Player:
	def __init__(self, y, x, name, hasBall, goal_x):
		self.x = x
		self.y = y
		self.name = name
		self.hasBall = hasBall
		self.goal_x = goal_x

		if self.hasBall:
			self.toString = self.name + 'b'
		else:
			self.toString = self.name

		self.actions = [0, 1, 2, 3, 4]  # up, down, right, left, and stick
		self.opponent = None
		self.score = 0

		self.q = {}
		self.alpha = 0.05
		self.gamma = 0.9

		self.prev_state = None
		self.prev_self_action = None
		self.prev_opponent_action = None

		self.q_history = []
		self.epsilon = 0.999

		self.first_action = True

	def update(self, game_state, action, opponent_action, score):
		if self.hasBall:
			self.toString = self.name + 'b'
		else:
			self.toString = self.name
		if game_state not in self.q:
			self.q[game_state] = np.zeros([5, 5]).tolist()
			self.q[game_state][action][opponent_action] = score

		if self.prev_state:
			s = self.prev_state
			a1 = self.prev_self_action
			a2 = self.prev_opponent_action
			v = self.q[game_state][action][opponent_action]
			new_q = (1 - self.alpha) * self.q[s][a1][a2] + self.alpha * ((1 - self.gamma) * score + self.gamma * v)

			if a1 == focus_action_A and a2 == focus_action_B and s == focus_state:
				self.q_history.append(abs(self.q[s][a1][a2] - new_q))

			self.q[s][a1][a2] = new_q

		self.prev_state = game_state
		self.prev_self_action = action
		self.prev_opponent_action = opponent_action

	def getActions(self, method, game_state):
		if self.first_action:
			self.first_action = False
			return 1, 4
		if method == 'random':
			return random.choice(self.actions), random.choice(self.opponent.actions)
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
		best_opponent_action = random.choice(self.opponent.actions)
		if game_state in self.q and random.random() > self.epsilon:
			action_pairs = self.q[game_state]
			for self_action in range(len(action_pairs)):
				opponent_action = action_pairs[self_action].index(max(action_pairs[self_action]))
				if action_pairs[self_action][opponent_action] > max_reward:
					max_reward = action_pairs[self_action][opponent_action]
					best_self_action = self_action
					best_opponent_action = opponent_action
		self.epsilon = max(0.001, 0.9999*self.epsilon)
		return best_self_action, best_opponent_action

#
#
# def nash_foe_q(self):


def run_random_games(num, s):
	# s = Soccer()
	for i in range(num):
		while not s.done:
			# s.render()
			A_action, B_action = s.A.getActions('random', grid_key(s.grid))
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
	for i in range(num):
		A_action = focus_action_A
		B_action = focus_action_B
		s.processTurn(A_action, B_action)
		while not s.done:
			# s.render()
			A_action, B_action = s.A.getActions('friend-q', grid_key(s.grid))
			s.processTurn(A_action, B_action)
		s.render()
		print('=====================================')
		s.reset()

	with open("friend_q_history.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for r in s.A.q_history:
			writer.writerow([r])

	print('done')
	return s


# s = Soccer()
#
# s.A.hasBall = False
# s.B.hasBall = True
#
# s.render()
#
# s.processTurn(4, 1) # down
# s.render()
# s.processTurn(4, 2) # right
# s.render()
# s.processTurn(4, 2) # right
# s.render()
#
# print(s.grid)
# print(tuple(s.grid))
# print(s.done)

soc = Soccer()
# soc = run_random_games(100000, soc)
soc = run_friend_games(100000, soc)
