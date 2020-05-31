class ndie:
	
	value = []
	isBad = []
	theta = 0.00001
	
	def __init__(self,badList):
		self.isBad = badList
		self.value = [0] * 100#len(badList)
		print(badList)
		self.findExpectedValue()
		print()
		
	def findOptimum(self, state):
		expectation = self.expectedReturn(state) # expected new state if re-roll

		if expectation > 0:
			self.value[state] = expectation
		else:
			self.value[state] = 0
	
	def expectedReturn(self, num):
		total = 0
		for i in range(0,len(self.isBad)):
			if self.isBad[i] == 1:
				total -= num
			else:
				total += i+1 + self.value[num+i+1]
		return total/len(self.isBad)
		
	def findExpectedValue(self):
		delta = 1
		while delta > self.theta:
			delta = 0
			"Looping over all the states"
			for i in range(0,len(self.isBad)*2):
				oldvalue = self.value[i]
				self.findOptimum(i)
				diff = abs(oldvalue-self.value[i])
				delta = max(delta,diff)
		print(self.value[0])

#n = ndie([1,1,1,0,0,0])
#n = ndie([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
#n = ndie([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

# ndie([0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0])
# ndie([0,1,1,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0])
# ndie([0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,1])
# ndie([0,0,1,1,0,1,1])
# ndie([0,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1])
# ndie([0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0])
# ndie([0,1,0])
# ndie([0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1])
# ndie([0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0])
# ndie([0,0,0,1])

ndie([0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0])
ndie([0,1,1,1,0,0,1,0,1,0,1,0,1,0,0,0,1,1])
ndie([0,0,1,0,1,0,1,1,0])
ndie([0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1])
ndie([0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,0])
ndie([0,1,1,0,0,1,0,0,1,1,1,0,1,0,1,0,1,0,1])
ndie([0,0,1,1,1,0,0,0,1,0])
ndie([0,1,1,0,1,1,0])
ndie([0,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0])
ndie([0,1,1,0,1,1,1,1,1,0,1])