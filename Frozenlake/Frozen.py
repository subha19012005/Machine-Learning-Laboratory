import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import random
from collections import deque
import matplotlib.pyplot as plt

# Environment (deterministic = faster learning)
env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = int(env.observation_space.n)
action_size = int(env.action_space.n)

# One-hot encoding
def one_hot(state):
    vec = np.zeros(state_size)
    vec[state] = 1
    return np.reshape(vec, [1, state_size])

# ⚡ FAST hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.98   # faster learning
learning_rate = 0.001
batch_size = 16
episodes = 120          # very fast

memory = deque(maxlen=1000)

# ⚡ SMALL model (fast)
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(state_size,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

model = build_model()

# Store
def remember(s, a, r, ns, d):
    memory.append((s, a, r, ns, d))

# Action
def act(state):
    if np.random.rand() < epsilon:
        return random.randrange(action_size)
    return np.argmax(model.predict(state, verbose=0)[0])

# ⚡ FAST training
def replay():
    global epsilon
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    states = []
    targets = []

    for s, a, r, ns, d in minibatch:
        target = r
        if not d:
            target = r + gamma * np.max(model.predict(ns, verbose=0)[0])

        target_f = model.predict(s, verbose=0)
        target_f[0][a] = target

        states.append(s[0])
        targets.append(target_f[0])

    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Training loop
rewards = []

for e in range(episodes):
    state, _ = env.reset()
    state = one_hot(state)

    total_reward = 0

    for t in range(30):   # fewer steps = faster
        action = act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = one_hot(next_state)

        remember(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward

        # ⚡ train less frequently
        if t % 4 == 0:
            replay()

        if done:
            break

    rewards.append(total_reward)
    print(f"Ep {e+1}, Reward: {total_reward}")

# Plot
plt.plot(rewards)
plt.title("Fast Training Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

# 🎥 Simulation
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

state, _ = env.reset()
state = one_hot(state)

done = False

while not done:
    action = np.argmax(model.predict(state, verbose=0)[0])

    next_state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    state = one_hot(next_state)

env.close()
