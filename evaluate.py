import pygame
from stable_baselines3 import PPO
import supersuit as ss
from env.BombermanEnv import BombermanEnv
import utils


pygame.init()

def main():
    # PettingZoo env
    env = BombermanEnv()
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v2(env, 4)
    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    
    model = PPO.load("saved_policy/best_model", env=env)

    obs = env.reset()
    episode_reward = 0
    utils.TRAINING = False
    # Evaluate the agent:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            episode_reward = reward
            if terminated.all():# or truncated.all():
                print("Reward:", episode_reward)
                episode_reward = 0
                obs = env.reset()
    
    pygame.quit()


if __name__ == "__main__":
    main()
