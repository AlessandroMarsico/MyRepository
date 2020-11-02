import numpy as np


def step(state, action, transition_probabilities, rewards):
    probs = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probs)
    reward = rewards[state][action][next_state]
    return next_state, reward


def exploration_policy(state, possible_actions):
    return np.random.choice(possible_actions[state])


transition_probabilities = [
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]
    ]

rewards = [
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [40, 0, 0], [0, 0, 0]]
]

possible_actions = [[0, 1, 2], [0, 2], [1]]
Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions

for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions


alpha0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.90  # discount factor
state = 0  # initial state

for iteration in range(10000):
    action = exploration_policy(state, possible_actions)
    next_state, reward = step(state, action, transition_probabilities, rewards)
    next_value = np.max(Q_values[next_state])
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state

print(Q_values)
print(np.argmax(Q_values, axis=1))
