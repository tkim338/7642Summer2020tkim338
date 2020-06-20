import numpy
import gym
import time
import Box2D
import random
import matplotlib.pyplot as plt


def parse_state(state):
    for i in range(len(state)):
        state[i] = round(state[i], 2)
    return state.tobytes()


def get_vertical(state):
    state = numpy.asarray([state[1], state[3], state[6], state[7]])
    return parse_state(state)


def get_horizontal(state):
    state = numpy.asarray([state[0], state[2], state[4], state[5], state[6], state[7]])
    return parse_state(state)


def specific_control(state, policy, actions):
    max_reward = None
    best_action = None
    for a in actions:
        if (state, a) in policy:
            reward = policy[(state, a)]
            if max_reward is None or reward > max_reward:
                max_reward = reward
                best_action = a
    if best_action:
        return best_action, max_reward
    else:
        return random.choice(actions), 0


def policy_action(full_state, vertical_policy, horizontal_policy):
    vertical_action, vertical_reward = specific_control(get_vertical(full_state), vertical_policy, [0, 2])
    horizontal_action, horizontal_reward = specific_control(get_horizontal(full_state), horizontal_policy, [0, 1, 3])
    if vertical_reward > horizontal_reward:
        return vertical_action
    else:
        return horizontal_action


def get_action(full_state, vertical_policy, horizontal_policy, eps):
    b = numpy.random.random()
    if b < 0.1:
        return 0
    a = numpy.random.random()
    if a < eps:
        return random.choice([0, 1, 2, 3])
    else:
        return policy_action(full_state, vertical_policy, horizontal_policy)


def update_policy(policy, state, action, reward, a, g, prev_policy):
    if (state, action) in policy:
        policy[(state, action)] += a * (reward + g * policy[(state, action)] - prev_policy)
    else:
        policy[(state, action)] = reward
    return policy

# input params
gamma = 0.95
alpha = 0.2
epsilon = 0.1
episodes = 100000

# setup
env = gym.make('LunarLander-v2').unwrapped
# pol = numpy.zeros((env.nS, env.nA))
v_pol = {}  # {((state tuple), action int), reward}
h_pol = {}
seed = None
render = True

env.seed(seed)
total_reward = 0
steps = 0

reward_history = []
episode_num = []

# plt.xlabel('Episode')
# plt.ylabel('Cumulative Episode Reward')
# plt.show(block=False)
# axes = plt.gca()
# axes.set_xlim([0, episodes])
# axes.set_ylim([-200, 200])

for ep in range(episodes):
    ob = env.reset()
    cum_rew = 0
    completed = False
    next_action = get_action(ob, v_pol, h_pol, epsilon)
    prev_v_pol = 0
    prev_h_pol = 0

    while not completed and cum_rew > -200 and abs(ob[0]) < 0.3:
        ob, rew, completed, info = env.step(next_action)
        cum_rew += rew

        if next_action in [0, 2]:
            v_pol = update_policy(v_pol, get_vertical(ob), next_action, rew, alpha, gamma, prev_v_pol)
            prev_v_pol = v_pol[(get_vertical(ob), next_action)]
        if next_action in [0, 1, 3]:
            h_pol = update_policy(h_pol, get_horizontal(ob), next_action, rew, alpha, gamma, prev_h_pol)
            prev_h_pol = h_pol[(get_horizontal(ob), next_action)]

        env.render()

        next_action = get_action(ob, v_pol, h_pol, epsilon)

    print(str(cum_rew) + ',\t' + str(ep))

        # reward_history.append(cum_rew)
        # episode_num.append(ep)
        # #if ep % 10000 == 0:
        # plt.plot(episode_num, reward_history, 'r.')
        # plt.draw()

print('asdfasdf')