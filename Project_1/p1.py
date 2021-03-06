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
	for i in range(0, sets):
		set = []
		for j in range(0, sequences):
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
	# exclude end (terminal) values in error calculation
	gt = gt[1:-1]
	values = values[1:-1]
	return math.sqrt(sum((values - gt)**2)/len(gt))

##########################################################################################

def computeSum(subSeq, lamb, values):
	summation = 0
	origVal = values[subSeq[0]]
	finalLamb = 1
	for k in range(1, len(subSeq)-1):  # exclude first and last states
		newVal = values[subSeq[k]]
		summation += (1 - lamb) * lamb**(k-1) * (newVal - origVal)
		finalLamb -= (1 - lamb) * lamb**(k-1)
	terminalVal = values[subSeq[-1]]
	summation += finalLamb * (terminalVal - origVal)
	return summation

def computeDelta(seq, lamb, alpha, values):
	delta = np.array(values[:] * 0.0)
	for i in range(0, len(seq)):
		delta[seq[i]] += alpha * computeSum(seq[i:], lamb, values)
	return delta

def processSet_fig3(set, lamb, alpha, values, threshold=0.0001):
	#values=np.array([0, 0, 0, 0, 0, 0, 1.0])
	deltas = np.array([1.0]*7)
	n = 0
	
	while max(abs(deltas)) > threshold and n < 100:
		n += 1
		deltas = np.array([0.0]*7)
		for seq in set:
			d = computeDelta(seq, lamb, alpha, values)
			deltas += d
		values += deltas
	return values

def processSet_fig4_5(set, lamb, alpha):
	values = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
	for seq in set:
		values += computeDelta(seq, lamb, alpha, values)
	return values

def computeError(gt, data, lamb, alpha=0.01, fig=3, init=np.array([0, 0, 0, 0, 0, 0, 1.0]), threshold=0.0001):
	errors = np.array([])
	values = np.array([])
	for set in data:
		if fig == 3:
			values = processSet_fig3(set, lamb, alpha, init, threshold)
		elif fig == 4 or fig == 5:
			values = processSet_fig4_5(set, lamb, alpha)
		errors = np.append(errors, computeRMSE(gt, values))
	return np.mean(errors)

##########################################################################################

def getFigure3(gt, data):
	errors = []
	lambdas = []
	for lamb in np.arange(0.0, 1.1, 0.1):
		errors.append(computeError(gt, data, lamb, fig=3))
		lambdas.append(lamb)
		
	plt.figure()
	plt.plot(lambdas, errors)
	plt.xlabel('lambda')
	plt.ylabel('RMS error')
	plt.show()

def getFigure4(gt, data):
	handles = []
	chartData = []
	for lamb in [0, 0.3, 0.8, 1]:
		label = 'lambda = '+str(lamb)
		errors = []
		alphas = []
		for alpha in np.arange(0.0, 0.61, 0.01):
			errors.append(computeError(gt, data, lamb, alpha, fig=4))
			alphas.append(alpha)
		chartData.append([alphas, errors, label])

	plt.figure()
	for datum in chartData:
		h = plt.plot(datum[0], datum[1], label=datum[2])
		handles += h
	plt.legend(handles=handles)
	axes = plt.gca()
	axes.set_ylim([0, 0.8])
	#plt.title('Figure 4: Average error with varying alpha and lambda')
	plt.xlabel('alpha')
	plt.ylabel('RMS error')
	plt.show()

def getFigure5(gt, data):
	errors = []
	lambdas = []
	for lamb in np.arange(0.0, 1.1, 0.1):
		min_error = [1e100]
		for alpha in np.arange(0, 0.61, 0.01):
			e = computeError(gt, data, lamb, alpha, fig=5)
			if e < min_error[0]:
				min_error = [e, lamb]
				e = min_error
		errors.append(min_error[0])
		lambdas.append(min_error[1])

	plt.figure()
	plt.plot(lambdas, errors)
	#plt.title('Figure 5: Average error at best alpha value')
	plt.xlabel('lambda')
	plt.ylabel('RMS error using best alpha')
	plt.show()

def testFigure3():
	np.random.seed(1)
	gt = computeGT()
	data = generateTrainingSets(100, 10)
	
	plt.figure()
	handle_list = []
	
	mod_init = []
	mod_alpha = []
	mod_threshold = []
	lambdas = []
	for lamb in np.arange(0.0, 1.1, 0.1):
		mod_init.append(computeError(gt, data, lamb, fig=3, init=np.array([0, 1, 1, 1, 1, 1, 1.0])))
		mod_threshold.append(computeError(gt, data, lamb, fig=3, threshold=0.001))
		mod_alpha.append(computeError(gt, data, lamb, alpha=0.02, fig=3))
		lambdas.append(lamb)
		
	handle_list += plt.plot(lambdas, mod_init, label='Initial estimates: [0, 1, 1, 1, 1, 1, 1]')
	handle_list += plt.plot(lambdas, mod_threshold, label='Convergence threshold: 0.001')
	handle_list += plt.plot(lambdas, mod_alpha, label='Learning rate: 0.02')
	
	plt.xlabel('lambda')
	plt.ylabel('RMS error')
	plt.legend(handles=handle_list)
	plt.show()
	
def testSeedsFigure3():
	plt.figure()
	handle_list = []
	plot_data = [[],[],[],[],[]]
	lambdas = np.arange(0.0, 1.1, 0.1)

	for n in range(0,5):
		np.random.seed(n+2)
		gt = computeGT()
		data = generateTrainingSets(100, 10)

		for lamb in np.arange(0.0, 1.1, 0.1):
			plot_data[n].append(computeError(gt, data, lamb, fig=3))
	
	for i in range(0,5):
		handle_list += plt.plot(lambdas, plot_data[i], label='Seed: '+str(i+2))
	
	plt.xlabel('lambda')
	plt.ylabel('RMS error')
	plt.legend(handles=handle_list)
	plt.show()

np.random.seed(1)
gt = computeGT()
data = generateTrainingSets(100, 10)

getFigure3(gt, data)
getFigure4(gt, data)
getFigure5(gt, data)

#testFigure3()
#testSeedsFigure3()