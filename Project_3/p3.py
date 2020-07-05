import numpy as np
import random

class Soccer:
	def __init__(self):
		self.grid = np.zeros([2,4]).tolist()
		if np.random.random() > 0.5:
			A_has_ball = True
		else:
			A_has_ball = False
		self.A = Player(0, 2, 'A', A_has_ball)
		self.B = Player(0, 1, 'B', not A_has_ball)
		self.A.opponent = self.B
		self.B.opponent = self.A

		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

	def reset(self):
		self.grid = np.zeros([2, 4]).tolist()
		if np.random.random() > 0.5:
			A_has_ball = True
		else:
			A_has_ball = False
		self.A.x = 2
		self.A.y = 0
		self.A.hasBall = A_has_ball
		self.B.x = 1
		self.B.y = 0
		self.B.hasBall = not A_has_ball

		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

	def render(self):
		self.A.update()
		self.B.update()

		self.grid = np.zeros([2, 4]).tolist()
		self.grid[self.A.y][self.A.x] = self.A.toString
		self.grid[self.B.y][self.B.x] = self.B.toString

		print(self.grid[0][:])
		print(self.grid[1][:])
		print('A score: ', self.A.score, '; B score: ', self.B.score)

	def processTurn(self, A_action, B_action):
		if np.random.random() > 0.5:
			self.processAction(self.A, A_action)
			self.processAction(self.B, B_action)
		else:
			self.processAction(self.B, B_action)
			self.processAction(self.A, A_action)

	def processAction(self, player, action):
		actions = {0: self.up, 1: self.down, 2: self.right, 3: self.left, 4: self.stay}
		actions[action](player)
		self.checkGoal(player)

	def up(self, player):
		if player.y > 0:
			if self.grid[player.y-1][player.x] == 0:
				player.y -= 1
			else:
				self.checkCollision(player)
		return player

	def down(self, player):
		if player.y < 1:
			if self.grid[player.y+1][player.x] == 0:
				player.y += 1
			else:
				self.checkCollision(player)
		return player

	def right(self, player):
		if player.x < 3:
			if self.grid[player.y][player.x+1] == 0:
				player.x += 1
			else:
				self.checkCollision(player)

	def left(self, player):
		if player.x > 0:
			if self.grid[player.y][player.x-1] == 0:
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
		if player.x == 0 or player.x == 3:
			player.score += 100
			player.opponent.score -= 100
			self.reset()

class Player:
	def __init__(self, y, x, name, hasBall):
		self.x = x
		self.y = y
		self.name = name
		self.hasBall = hasBall
		self.toString = ''
		self.update()
		self.actions = [0,1,2,3,4] # up, down, right, left, and stick
		self.opponent = None
		self.score = 0

	def update(self):
		if self.hasBall:
			self.toString = self.name+'b'
		else:
			self.toString = self.name

	def getAction(self):
		return random.choice(self.actions)

s = Soccer()

s.A.hasBall = False
s.B.hasBall = True

s.render()

s.processTurn(4, 1) # down
s.render()
s.processTurn(4, 2) # right
s.render()
s.processTurn(4, 2) # right
s.render()