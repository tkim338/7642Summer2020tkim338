#delta_w_t = alpha * (next_state - curr_state) * sum(lamb**(t-k) * grad_w * state_k)

import matplotlib.pyplot as plt
import numpy as np
import math

def computeGT(): # compute ground truth
	values = [0,0,0,0,0,0,1]
	gt = [[0],[0],[0],[0],[0],[0],[1]]
	alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	for i in range(100):
		for ind in range(0,7):
			gt[ind].append(values[ind])
			values[ind] = computeReturn(values, ind)
	#print(values)
	
	# plt.figure()
	# for ind in range(0,7):
		# plt.plot(gt[ind], label=alpha[ind])
	# plt.legend(alpha)
	# plt.title('True values of states')
	# plt.xlabel('Iterations')
	# plt.ylabel('Value')
	# plt.show()
	
	return values
	
def generateTrainingSets(sets=100, sequences=10):
	trainingData = []
	for i in range(0,sets):
		set = []
		for j in range(0,sequences):
			ind = 3
			sequence = [ind]
			while True:
				if np.random.random() > 0.5:
					ind += 1
				else:
					ind -= 1
				sequence.append(ind)
				if ind == 6 or ind == 0:
					break
			set.append(sequence)
		trainingData.append(set)
	#printTrainingData(trainingData)
	return trainingData
	
def printTrainingData(data):
	print('==================== training data ====================')
	for set in data:
		for seq in set:
			print(seq)
	print('==================== training data ====================')
	
def computeReturn(vals, i, lr=0.5):
	if i == 0:
		return 0
	elif i == len(vals)-1:
		return 1
	else:
		#return vals[i] + lr * ((vals[i-1]+vals[i+1])/2 - vals[i])
		return (vals[i-1]+vals[i+1])/2
		
def updateEstimate(currEstimate, newSeq, lamb, lr=0.1):
	delta = [0] * len(currEstimate)
	for ind in range(0,len(newSeq)-1): # skip end state
		state = newSeq[ind]
		
		newEstimate = 0
		endWeight = 1
		for i in range(ind+1, len(newSeq)):
			n = i - ind
			newEstimate += (1-lamb) * lamb**(n-1) * computeReturn(currEstimate, newSeq[i])
			endWeight -= (1-lamb) * lamb**(n-1)
		newEstimate += endWeight * computeReturn(currEstimate, newSeq[-1])

		delta[state] += lr * (newEstimate - currEstimate[state])
		currEstimate[state] += lr * (newEstimate - currEstimate[state])
	return currEstimate#, delta

def estimateValues(data, lamb, lr, threshold=0.001):
	estimates_list = []
	n = -1
	for set in data:
		n += 1
		estimates = [0.0,0,0,0,0,0,1]
		prevEstimates = [0,0,0,0,0,0,0]
		while max(abs(np.array(estimates) - np.array(prevEstimates))) > threshold:
			prevEstimates = estimates[:]
			est_delta = np.array([0.0]*7)
			for seq in set:
				estimates = updateEstimate(estimates, seq, lamb, lr)
				# temp, delta = updateEstimate(estimates, seq, lamb, lr)
				# est_delta += np.array(delta)
			# estimates += est_delta
		estimates_list.append(estimates)
	return estimates_list

def computeError(gt, data, lamb, lr):
	est_list = estimateValues(data, lamb, lr)
	errors = []
	for e in est_list:
		errors.append(math.sqrt(sum((np.array(e) - np.array(gt))**2)/len(e)))
	return sum(errors)/len(errors)

def getFigure3(gt, data):
	errors = []
	lambdas = []
	for lamb in np.arange(0.0,1.1,0.1):
		errors.append(computeError(gt, data, lamb, 0.1))
		lambdas.append(lamb)
		
	plt.figure()
	plt.plot(lambdas, errors)
	plt.title('Average error')
	plt.xlabel('lambda')
	plt.ylabel('RMS error')
	plt.show()

#np.random.seed(1)
gt = computeGT()
data = generateTrainingSets(100, 10)

#figure3(gt, data)
err = computeError(gt, data, lamb=1, lr=0.1)
print(err)