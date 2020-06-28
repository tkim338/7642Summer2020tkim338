import numpy as np

class BarFight:
	def __init__(self, num_patrons):
		self.neutral_set = np.array(range(num_patrons))
		self.peacekeeper_set = np.array(range(num_patrons))
		self.instigator_set = np.array(range(num_patrons))
		self.num_patrons = num_patrons
		self.history = {}

	def update_patrons(self, new_episode, new_outcome):
		self.history[str(new_episode)] = new_outcome
		if new_outcome == 1: # fight occurred, peacekeeper absent, instigator present
			for patron_id in range(len(new_episode)):
				# none of these patrons can be a peacekeeper, remove from set
				if new_episode[patron_id] == 1:
					self.peacekeeper_set = self.peacekeeper_set[self.peacekeeper_set != patron_id]

			possible_instigators = np.where(new_episode == 1)[0]
			remaining_suspects = np.where(new_episode == 1)[0]
			for suspect in possible_instigators:
				if suspect not in self.instigator_set:
					remaining_suspects = remaining_suspects[remaining_suspects != suspect]
			if len(remaining_suspects) == 1:
				self.instigator_set = np.array(remaining_suspects)

			# if sum(new_episode) == 1: # only one patron, has to be instigator
			# 	instigator = int(np.where(new_episode == 1)[0])
			# 	self.neutral_set = self.neutral_set[self.neutral_set != instigator]
			# 	self.instigator_set = np.array([instigator])

		else: # no fight occurred
			if sum(new_episode) == 1: # only one patron, is not instigator
				patron_id = int(np.where(new_episode == 1)[0])
				self.instigator_set = self.instigator_set[self.instigator_set != patron_id]

		if len(self.instigator_set) == 1:
			if new_episode[int(self.instigator_set[0])] == 1 and new_outcome == 0: # instigator present, but no fight, so peacekeeper must also be present
				for patron_id in range(len(new_episode)):
					if new_episode[patron_id] == 0: # any absent patrons are not the peacekeeper
						self.peacekeeper_set = self.peacekeeper_set[self.peacekeeper_set != patron_id]

		if len(self.peacekeeper_set) == 1: # last patron must be peacekeeper
			peacekeeper = int(self.peacekeeper_set[0])
			self.neutral_set = self.neutral_set[self.neutral_set != peacekeeper]
			self.instigator_set = self.instigator_set[self.instigator_set != peacekeeper]

		for patron_id in range(len(new_episode)):
			if patron_id not in self.neutral_set and patron_id in self.instigator_set and patron_id not in self.peacekeeper_set: # patron must be instigator
				self.instigator_set = np.array([patron_id])
			if patron_id not in self.neutral_set and patron_id not in self.instigator_set and patron_id in self.peacekeeper_set:  # patron must be peacekeeper
				self.peacekeeper_set = np.array([patron_id])


	def make_prediction(self, ep):
		if all(ep == 0) or all(ep == 1):
			return 0
		if str(ep) in self.history:
			return self.history[str(ep)]

		permutations = np.array(self.get_permutations(0))
		possible_output = np.array([])
		for p in permutations:
			if -1 in p and 1 in p:
				possible_output = np.append(possible_output, np.dot(p, ep.T))
		possible_output[possible_output <= -1] = 0
		possible_output[possible_output > 1] = 1
		unique_values = np.unique(possible_output)
		if len(unique_values) > 1:
			return 2
		else:
			return max(int(unique_values[0]), 0)

	def get_permutations(self, patron_id):
		if patron_id >= self.num_patrons:
			return None

		permutations = []
		sub_permutations = self.get_permutations(patron_id + 1)
		if patron_id in self.peacekeeper_set:
			if not sub_permutations:
				permutations.append([-1])
			else:
				for sub in sub_permutations:
					permutations.append([-1] + sub)
		if patron_id in self.neutral_set:
			if not sub_permutations:
				permutations.append([0])
			else:
				for sub in sub_permutations:
					permutations.append([0] + sub)
		if patron_id in self.instigator_set:
			if not sub_permutations:
				permutations.append([1])
			else:
				for sub in sub_permutations:
					permutations.append([1] + sub)
		return permutations


# at_establishment = [[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]]
# fight_occurred = [0, 1, 0, 0, 0, 1, 0]
at_establishment = [[1,1],[0,1],[0,1],[0,1],[0,1],[1,1],[0,1],[0,1]]
fight_occurred = [0,1,1,1,1,0,1,1]

bf = BarFight(len(at_establishment[0]))
output = ''
for i in range(len(at_establishment)):
	episode = np.array(at_establishment[i])
	#print(episode, bf.make_prediction(np.array(episode)), fight_occurred[i])
	output += str(bf.make_prediction(np.array(episode)))
	bf.update_patrons(np.array(episode), fight_occurred[i])

print(output)
# print('sets')
# print(bf.peacekeeper_set)
# print(bf.neutral_set)
# print(bf.instigator_set)

