import numpy as np

def init_patrons(num_patrons):
	patrons = np.array([])
	for n in range(num_patrons):
		np.append(patrons, [-1, 0, 1])
	return patrons

def update_patrons(patrons, new_episode, new_outcome):
	if all(ep == 1) or all(ep == 0):
		return 0, patrons
	else:
		if new_outcome == 1: # instigator present and peacemaker absent
			for ind in new_episode:
				p = new_episode[ind]
				if p == 1 and 1 in p: # none of these patrons can be the peacemaker
					new_episode[ind] = p[p != -1]




at_establishment = [[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]]
fight_occurred = [0, 1, 0, 0, 0, 1, 0]

weights = np.ones((1, len(at_establishment[0])))
at_establishment = np.array(at_establishment)

output = ''
for i in range(len(at_establishment)):
	ep = at_establishment[i]
	fight = fight_occurred[i]

	if all(ep == 1) or all(ep == 0):
		output += str(0) # no fight
		print(ep, 0)
	else:
		prediction = int(np.dot(weights, ep.reshape(len(ep), 1))[0][0])
		prediction = max(prediction, 1)
		print(ep, prediction)

		# learning step
		if np.count_nonzero(weights) != 2:
			if fight == 1: # instigator present and peacemaker absent,
				mask = ep == 1


	
print(output)