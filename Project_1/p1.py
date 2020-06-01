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
	
def computeReturn(vals, i):
	if i == 0:
		return 0
	elif i == len(vals)-1:
		return 1
	else:
		return (vals[i-1]+vals[i+1])/2
		
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

def computeRMSE(gt, values):
	return math.sqrt(sum((values - gt)**2)/len(gt))

##########################################################################################

def computeSum(subSeq, lamb, values):
	summation = 0
	origVal = values[subSeq[0]]
	finalLamb = 1
	for k in range(1, len(subSeq)-1): # exclude first and last states
		newVal = values[subSeq[k]]
		#print('===='+str(subSeq)+'; '+str(origVal)+', '+str(newVal))
		summation += (1 - lamb) * lamb**(k-1) * (newVal - origVal)
		finalLamb -= (1 - lamb) * lamb**(k-1)
		#print(summation)
	terminalVal = values[subSeq[-1]]
	summation += finalLamb * (terminalVal - origVal)
	#print(summation)
	return summation

def computeDelta(seq, lamb, alpha, values):
	delta = np.array(values[:] * 0.0)
	for i in range(0, len(seq)):
		delta[seq[i]] += alpha * computeSum(seq[i:], lamb, values)
	return delta

def processSet(set, lamb, alpha=0.01, threshold=0.0001):
	values = np.array([0,0,0,0,0,0,1.0])
	deltas = np.array([1.0]*7)
	while max(abs(deltas)) > threshold:
	#for i in range(0,100):
		deltas = np.array([0.0]*7)
		for seq in set:
			d = computeDelta(seq, lamb, alpha, values)
			deltas += d#/len(set)
			#values += d
		values += deltas
		#print('values: '+str(values))
		#print('prevValues: '+str(prevValues))
	#print(values)
	return values

def computeError(gt, data, lamb):
	errors = np.array([])
	for set in data:
		values = processSet(set, lamb)
		errors = np.append(errors, computeRMSE(gt, values))
	return np.mean(errors)

##########################################################################################

def getFigure3(gt, data):
	errors = []
	lambdas = []
	for lamb in np.arange(0.0,1.1,0.1):
		errors.append(computeError(gt, data, lamb))
		lambdas.append(lamb)
		
	plt.figure()
	plt.plot(lambdas, errors)
	plt.title('Average error')
	plt.xlabel('lambda')
	plt.ylabel('RMS error')
	plt.show()

np.random.seed(1)
gt = computeGT()
data = generateTrainingSets(100, 10)

getFigure3(gt, data)
#print(computeError(gt, data, 0.5))