import numpy
import gym
import time


def compute_reward(state, action):
    # - 0: move south
    # - 1: move north
    # - 2: move east
    # - 3: move west
    # - 4: pickup passenger
    # - 5: dropoff passenger
    return


# def optimal_action(environment):
#     max_reward = None
#     possible_actions = []
#     action_list = environment.P[environment.s]
#     for action in range(environment.nA):  # [0, 1, 2, 3, 4, 5]
#         reward = action_list[action][0][2]
#         if max_reward is None or reward > max_reward:
#             max_reward = reward
#             possible_actions = [action]
#         elif reward == max_reward:
#             possible_actions.append(action)
#     return possible_actions[numpy.random.randint(0, len(possible_actions))]


def optimal_action(environment, policy):
    max_reward = None
    possible_actions = []
    action_list = environment.P[environment.s]
    for action in range(environment.nA):  # [0, 1, 2, 3, 4, 5]
        reward = action_list[action][0][2]
        if max_reward is None or reward > max_reward:
            max_reward = reward
            possible_actions = [action]
        elif reward == max_reward:
            possible_actions.append(action)
    return possible_actions[numpy.random.randint(0, len(possible_actions))]


def policy_action(environment, policy):
    max_reward = None
    best_action = None
    for action in range(environment.nA):  # [0, 1, 2, 3, 4, 5]
        reward = policy[environment.s][action]
        if max_reward is None or reward > max_reward:
            max_reward = reward
            best_action = action
    return best_action


def get_action(environment, policy, eps):
    a = numpy.random.random()
    if a < eps:
        return numpy.random.randint(environment.nA)
    else:
        return policy_action(environment, policy)


# input params
gamma = 0.9
alpha = 0.5
epsilon = 0.3
episodes = 10000000

# setup
env = gym.make('Taxi-v3').unwrapped
pol = numpy.zeros((env.nS, env.nA))

for ep in range(episodes):
    env.reset()
    # env.render()
    rew = 0
    next_action = get_action(env, pol, epsilon)
    don = False
    while not don and rew > -20:
        prev_s = env.s
        prev_action = next_action
        ob, rew, don, info = env.step(next_action)  # act(epsilon, env, next_action)
        next_action = get_action(env, pol, epsilon)

        #env.render()
        # update policy[state][action]
        pol[prev_s][prev_action] += alpha*(rew + gamma*pol[env.s][get_action(env, pol, 0.0)] - pol[prev_s][prev_action])
        #time.sleep(0.1)

env.render()
print('done')
