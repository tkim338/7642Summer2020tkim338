class ndie:
	
	state = 0 # capital
	isBad = []
	
	def __init__(self,badList):
		self.isBad = badList
				
	def computeState(self):
		self.state += self.expectedReturn()
	
	def expectedReturn(self):
		total = 0
		for i in range(0,len(self.isBad)):
			if self.isBad[i] == 1:
				total -= self.state
			else:
				total += i+1
		return total/len(self.isBad)

n = ndie([1,1,1,0,0,0])
#n = ndie([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
#n = ndie([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

for i in range(0,100):
	print('state: '+str(n.state)+', return: '+str(n.expectedReturn()))
	n.computeState()