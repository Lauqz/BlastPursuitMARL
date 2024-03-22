import pygame
import supersuit as ss
from stable_baselines3 import DQN, PPO, SAC, HER
from stable_baselines3.common.callbacks import EvalCallback
from env.BombermanEnv import BombermanEnv
from CustomNetwork.CustomCNN import CustomCNN


# Initialize Pygame
pygame.init()

def main():
    # PettingZoo env
    env = BombermanEnv()
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v2(env, 4)
    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Stable baselines algorithm
    eval_callback = EvalCallback(
        env,
        n_eval_episodes=30,
        best_model_save_path="saved_policy",
        eval_freq=20000,
    )

    # Load custom policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    pygame.display.set_caption("BlastPursuit RL")

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard/", policy_kwargs=policy_kwargs)
    print(model.policy)
    model.learn(total_timesteps=15000000, callback = eval_callback)

    pygame.quit()


if __name__ == "__main__":
    main()

