import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

print("🎮 Guaranteed Successful Simulation\n")

# Optimal path for FrozenLake (4x4)
# Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
safe_path = [1, 1, 2, 1, 2, 2]

state, _ = env.reset()

for action in safe_path:
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    time.sleep(0.5)

    if done:
        break

if reward == 1:
    print("\n✅ SUCCESS! Reached Goal 🎯")
else:
    print("\n❌ Something went wrong")

env.close()
