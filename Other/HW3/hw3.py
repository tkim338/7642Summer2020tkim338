import numpy
import gym

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


def generate_map(map_string):
    count = len(map_string)
    dim = int(numpy.sqrt(count))
    output_map = []
    for col in range(0, dim):
        output_map.append(map_string[col*dim:(col+1)*dim])
    return output_map


def optimal_action(table, state):
    max_reward = 0
    best_action = 0
    for action in range(len(table[state])):  # [0, 1, 2, 3]
        reward = table[state][action]
        if reward > max_reward:
            max_reward = reward
            best_action = action
    return best_action


# def greedy_action(environ, action):
#     observation, reward, done, info = environ.step(action)
#     return observation, reward, done
#
#
# def random_action(environ):
#     observation, reward, done, info = environ.step(numpy.random.randint(4))
#     return observation, reward, done
#
#
# def act(eps, environ, action):
#     a = numpy.random.random()
#     if a < eps:
#         return random_action(environ)
#     else:
#         return greedy_action(environ, action)


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
# amap = "SFFG"
# gamma = 1.0
# alpha = 0.24
# epsilon = 0.09
# episodes = 49553
# seed = 202404

# amap = "SFFFHFFFFFFFFFFG"
# gamma = 1.0
# alpha = 0.25
# epsilon = 0.29
# episodes = 14697
# seed = 741684

# amap = "SFFFFHFFFFFFFFFFFFFFFFFFG"
# gamma = 0.91
# alpha = 0.12
# epsilon = 0.13
# episodes = 42271
# seed = 983459

amap="SFFG"
gamma=0.95
alpha=0.1
epsilon=0.23
episodes=18804
seed=105338

# setup
numpy.random.seed(seed)
desc = generate_map(amap)
env = gym.make('FrozenLake-v0', desc=desc).unwrapped
env.seed(seed)

q_table = numpy.zeros((len(amap), 4))  # [[0] * 4] * len(amap)

for ep in range(episodes):
    ob = env.reset()
    rew = 0
    next_action = get_action(q_table, ob, epsilon)
    don = False
    while not don:
        prev_ob = ob  # ob = state
        prev_action = next_action
        ob, rew, don, info = env.step(next_action)  # act(epsilon, env, next_action)
        next_action = get_action(q_table, ob, epsilon)

        # update q_table[state][action]
        q_table[prev_ob][prev_action] += alpha*(rew + gamma*q_table[ob][next_action] - q_table[prev_ob][prev_action])

print(q_table)
print(print_policy(q_table))
