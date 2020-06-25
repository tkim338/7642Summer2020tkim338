import numpy
import gym
import time
import Box2D
import random
import matplotlib.pyplot as plt
import csv

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.
    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center
    if angle_targ > 0.4: angle_targ = 0.4    # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*numpy.abs(s[0])           # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > numpy.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1
    return a


def parse_state(state):
    for i in range(len(state)):
        state[i] = round(state[i], 1)
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
        #return heuristic(None, full_state)
    else:
        return policy_action(full_state, vertical_policy, horizontal_policy)

def update_policy(policy, state, action, reward, a, g, prev_policy):
    if reward < -200:
        return policy
    scaled_a = a * (reward+200)/400
    scaled_a = max(scaled_a, 0.01)
    scaled_a = min(scaled_a, 0.99)
    if (state, action) in policy:
        policy[(state, action)] += scaled_a * (reward + g * policy[(state, action)] - prev_policy)
    else:
        policy[(state, action)] = reward
    return policy

def modify_reward(state, reward, step_num):
    r = reward
    if step_num > 500:
        r -= 100
    if abs(state[0]) > 0.2: # horizontal coordinate
        r -= 100
    if abs(state[4]) > 0.2: # angle [radians]
        r -= 50
    if state[3] > 0: # vertical speed
        r -= 50
    return r

# input params
gamma = 0.99
alpha = 0.001
epsilon = 0.99
episodes = 5000

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
    step = 0
    while not completed and cum_rew > -200 and abs(ob[0]) < 0.4 and step < 600:
        ob, rew, completed, info = env.step(next_action)
        #rew = modify_reward(ob, rew, step)
        cum_rew += rew

        if next_action in [0, 2]:
            v_pol = update_policy(v_pol, get_vertical(ob), next_action, rew, alpha, gamma, prev_v_pol)
            prev_v_pol = v_pol[(get_vertical(ob), next_action)]
        if next_action in [0, 1, 3]:
            h_pol = update_policy(h_pol, get_horizontal(ob), next_action, rew, alpha, gamma, prev_h_pol)
            prev_h_pol = h_pol[(get_horizontal(ob), next_action)]
        # if ep > 5000:
        #     env.render()

        next_action = get_action(ob, v_pol, h_pol, epsilon)
        step += 1
        # epsilon *= 0.9999
        epsilon = (-1/1000)*ep + 1
        epsilon = max(epsilon, 0.1)

    reward_history.append(cum_rew)

    print('Episode reward:', round(cum_rew, 3), '\tAverage reward:', round(numpy.mean(reward_history[-10:]), 3), '\tEpisode:', ep, '\tepsilon:', round(epsilon, 3))


with open('sarsa_training_history.csv', "w", newline="") as f:
    writer = csv.writer(f)
    for r in reward_history:
        writer.writerow([r])