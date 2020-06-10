import numpy
import gym


def optimal_action(table, state):
    max_reward = 0
    best_action = 0
    for action in range(len(table[state])):  # [0, 1, 2, 3]
        reward = table[state][action]
        if reward > max_reward:
            max_reward = reward
            best_action = action
    return best_action


def get_action(table, state, eps):
    a = numpy.random.random()
    if a < eps:
        return numpy.random.randint(4)
    else:
        return optimal_action(table, state)


def print_policy(table):
    mapping = {0: '<', 1: 'v', 2: '>', 3: '^'}
    output_string = ''
    for state in range(len(table)):
        best_action = optimal_action(table, state)
        output_string += mapping[best_action]
    return output_string

# input params
amap="SFFG"
gamma=0.95
alpha=0.1
epsilon=0.23
episodes=18804
seed=105338

# setup
# numpy.random.seed(seed)
# desc = generate_map(amap)
env = gym.make('Taxi-v3').unwrapped
# env.seed(seed)
env.render()
# q_table = numpy.zeros((len(amap), 4))  # [[0] * 4] * len(amap)
#
# for ep in range(episodes):
#     ob = env.reset()
#     rew = 0
#     next_action = get_action(q_table, ob, epsilon)
#     don = False
#     while not don:
#         prev_ob = ob  # ob = state
#         prev_action = next_action
#         ob, rew, don, info = env.step(next_action)  # act(epsilon, env, next_action)
#         next_action = get_action(q_table, ob, epsilon)
#
#         # update q_table[state][action]
#         q_table[prev_ob][prev_action] += alpha*(rew + gamma*q_table[ob][next_action] - q_table[prev_ob][prev_action])
#
# print(q_table)
# print(print_policy(q_table))
