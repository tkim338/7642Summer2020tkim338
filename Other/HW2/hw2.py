#    /-> 1 -\
# 0 -        -> 3 -> 4 -> 5-> 6
#    \-> 2 -/

class hw2:
	def __init__(self, p, vs, rs):
		self.prob = p
		self.values = vs
		self.rewards = rs
	
	def computeReturn(self, n):
		# max = 5
		r = self.values[0]
		for i in range(1,n+1):
			if i == 1:
				r += self.prob*self.rewards[0] + (1-self.prob)*self.rewards[1]
			elif i == 2:
				r += self.prob*self.rewards[2] + (1-self.prob)*self.rewards[3]
			else:
				r += self.rewards[i+1]
		
		if n == 0:
			return r
		elif n == 1:
			r += self.prob*self.values[1] + (1-self.prob)*self.values[2]
		else:
			r += self.values[n+1]
		
		return r
	
	def computeLambda(self):
		rate = 0.01
		threshold = 0.000001
		goal = self.computeReturn(5)
		
		lamb = 0.5
		sum = 0
		
		attempts = 0
	
		while abs(sum-goal) > threshold and attempts < 200:
			sum = 0
			attempts += 1
			remainder = 1
			for n in range(1,5):
				sum += (1-lamb) * lamb**(n-1) * self.computeReturn(n)
				remainder -= (1-lamb) * lamb**(n-1)
			sum += remainder * goal
			
			lamb += rate * (goal-sum)
			print(str(lamb) + ',\t' + str(sum) +',\t'+ str(goal))
		
		return lamb

t1 = hw2(0.81, [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0], [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6])
t2 = hw2(0.22, [12.3, -5.2, 0.0, 25.4, 10.6, 9.2, 0.0], [-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1])
t3 = hw2(0.64, [-6.5, 4.9, 7.8, -2.3, 25.5, -10.2, 0.0], [-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9])


p=0.1

V=[0.0,24.3,11.2,11.1,24.4,24.5,0.0]

rewards=[8.1,3.3,-4.7,2.4,5.6,1.4,7.9]



q = hw2(p, V, rewards)

for i in range(1,6):
	print(q.computeReturn(i))
print('-----------------------------')

print(q.computeLambda())