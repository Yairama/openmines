import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from openmines_gym import GymMineEnv

# Create and smoke-test the environment
env = GymMineEnv("../../../../conf/north_pit_mine.json")
env = FlattenObservation(env)

# Exercise a few random steps
obs, info = env.reset()
print("Observation shape:", obs.shape)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
        break

env.close()