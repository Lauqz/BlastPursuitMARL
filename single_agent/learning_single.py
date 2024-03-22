import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs import register
from gymnasium.wrappers import FlattenObservation
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import random
import pygame
import stable_baselines3
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


# Global variables
GRID_SIZE = 7
INDESTRUCTIBLE_WALL_PERCENTAGE = 8
DESTRUCTIBLE_WALL_PERCENTAGE = 12
NUM_BOMBERAGENTS = 1 # 1 FOR NOW
NUM_KILLERAGENTS = 4
TRAINING = True

# Cell types
EMPTY = 0
INDESTRUCTIBLE_WALL = 1
DESTRUCTIBLE_WALL = 2
BAGENT = 3
BOMB = 4
KAGENT = 5

# Image paths
BOMBERMAN_IMAGE_PATH = '../img/bomberman.png'
KILLERMAN_IMAGE_PATH = '../img/killer.png'
INDESTRUCTIBLE_WALL_IMAGE_PATH = '../img/indestructiblewall.png'
DESTRUCTIBLE_WALL_IMAGE_PATH = '../img/destructiblewall.jpg'
BOMB_IMAGE_PATH = '../img/bomb.png'

# Create a dictionary for image paths
CELL_IMAGES = {
    EMPTY: None,  # You can use None for empty cells
    INDESTRUCTIBLE_WALL: INDESTRUCTIBLE_WALL_IMAGE_PATH,
    DESTRUCTIBLE_WALL: DESTRUCTIBLE_WALL_IMAGE_PATH,
    BAGENT: BOMBERMAN_IMAGE_PATH,
    KAGENT: KILLERMAN_IMAGE_PATH,
    BOMB: BOMB_IMAGE_PATH,
}

# Initialize Pygame
pygame.init()

# Set up display
CELL_SIZE = 20
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bomberman RL")

# Custom policy
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # INPUT: (1x7x7) -> (32x?x?) (32x2x2) 
            nn.Conv2d(n_input_channels, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# BOMBER Agent Class
class BAgent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    def move(self, direction, grid, infos):
        # Calculate new position based on direction
        new_x, new_y = self.x, self.y
        if direction == 'up':
            new_y -= 1
        elif direction == 'down':
            new_y += 1
        elif direction == 'left':
            new_x -= 1
        elif direction == 'right':
            new_x += 1

        # Check if new position is within grid bounds and either empty or a bomb
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            if grid[new_x][new_y] in [EMPTY, BOMB]:
                grid[self.x][self.y] = EMPTY
                self.x, self.y = new_x, new_y
            elif grid[new_x][new_y] in [INDESTRUCTIBLE_WALL, DESTRUCTIBLE_WALL]:
                infos['bad_move'] = True
        else:
            infos['bad_move'] = True

    def make_decision(self, action, grid, bombs, infos):
        if action == 4:  # 20% chance to plant a bomb
            self.plant_bomb(bombs, grid)
        else:
            directions = {0:'up', 1:'down', 2:'left', 3:'right'}
            self.move(directions[action], grid, infos)

    def plant_bomb(self, bombs, grid):
        if grid[self.x][self.y] == EMPTY:
            bomb = Bomb(self.x, self.y)
            bombs.append(bomb)
            grid[self.x][self.y] = BOMB

    def check_collision_with_killerman(self, killermen):
        for killerman in killermen:
            if killerman.alive and self.x == killerman.x and self.y == killerman.y:
                self.alive = False

    def check_bomb_radius(self, bombs):
        for bomb in bombs: 
            for dx in range(bomb.x - bomb.radius, bomb.x + bomb.radius + 1):
                for dy in range(bomb.y -bomb.radius, bomb.y + bomb.radius + 1):
                    if (self.x,self.y) == (dx,dy):
                        return True
                    else:
                        return False

# Killerman Class
class Killerman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    def move(self, grid):
        # Calculate random movement
        directions = ['up', 'down', 'left', 'right']
        random_direction = random.choice(directions)
        new_x, new_y = self.x, self.y

        if random_direction == 'up':
            new_y -= 1
        elif random_direction == 'down':
            new_y += 1
        elif random_direction == 'left':
            new_x -= 1
        elif random_direction == 'right':
            new_x += 1

        # Check if new position is within grid bounds and either empty or a bomb
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            if grid[new_x][new_y] in [EMPTY, BOMB, BAGENT]:
                grid[self.x][self.y] = EMPTY
                self.x, self.y = new_x, new_y

# Bomb Class
class Bomb:
    def __init__(self, x, y, countdown=15, radius=1):
        self.x = x
        self.y = y
        self.countdown = countdown
        self.radius = radius

    def tick(self):
        self.countdown -= 1

    def has_exploded(self):
        return self.countdown <= 0

    def explode(self, grid, bagents, kagents, infos):
        # Affect the center cell
        if grid[self.x][self.y] == BOMB:
            grid[self.x][self.y] = EMPTY

        # Affect surrounding cells
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if grid[nx][ny] == DESTRUCTIBLE_WALL:
                        grid[nx][ny] = EMPTY
                        infos['wall'] = True
                    elif grid[nx][ny] == BAGENT:
                        for agent in bagents:
                            if agent.x == nx and agent.y == ny:
                                agent.alive = False
                    elif grid[nx][ny] == KAGENT:
                        for agent in kagents:
                            if agent.x == nx and agent.y == ny:
                                agent.alive = False
                                infos['kill'] = True
        for agent in bagents:
            if self.x - self.radius <= agent.x <= self.x + self.radius and \
               self.y - self.radius <= agent.y <= self.y + self.radius:
                agent.alive = False
        for agent in kagents:
            if self.x - self.radius <= agent.x <= self.x + self.radius and \
               self.y - self.radius <= agent.y <= self.y + self.radius:
                agent.alive = False

def initialize_grid():
    grid = np.full((GRID_SIZE, GRID_SIZE), EMPTY)

    # Add indestructible walls
    num_indestructible_walls = int(GRID_SIZE * GRID_SIZE * INDESTRUCTIBLE_WALL_PERCENTAGE / 100)
    while num_indestructible_walls > 0:
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[x][y] == EMPTY:
            grid[x][y] = INDESTRUCTIBLE_WALL
            num_indestructible_walls -= 1

    # Add destructible walls
    num_destructible_walls = int(GRID_SIZE * GRID_SIZE * DESTRUCTIBLE_WALL_PERCENTAGE / 100)
    while num_destructible_walls > 0:
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[x][y] == EMPTY:
            grid[x][y] = DESTRUCTIBLE_WALL
            num_destructible_walls -= 1
    
    bombers = []
    for _ in range(NUM_BOMBERAGENTS):
        while True:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if grid[x][y] == EMPTY:
                if sanity_check(x,y,grid):
                    agent = BAgent(x, y)
                    bombers.append(agent)
                    grid[x][y] = BAGENT
                    break

    # Initialize Killerman agents
    killers = []
    for _ in range(NUM_KILLERAGENTS):
        while True:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if grid[x][y] == EMPTY:
                if sanity_check(x,y,grid):
                    agent = Killerman(x, y)
                    killers.append(agent)
                    grid[x][y] = KAGENT
                    break
    return grid, bombers, killers

def sanity_check(x,y,grid):
    tmp = False
    # Agent at center
    if x not in [0, GRID_SIZE - 1] and y not in [0, GRID_SIZE - 1]:
        if grid[x-1][y] == EMPTY or grid[x+1][y] == EMPTY or grid[x][y-1] == EMPTY or grid[x][y+1] == EMPTY:
            tmp = True
    # Agent on corner
    elif x in [0, GRID_SIZE - 1] and y in [0, GRID_SIZE - 1]:
        if x == 0 and y == 0:
            if grid[x+1][y] == EMPTY or grid[x][y+1] == EMPTY:
                tmp = True
        elif x == 0 and y == GRID_SIZE - 1:
            if grid[x+1][y] == EMPTY or grid[x][y-1] == EMPTY:
                tmp = True
        elif x == GRID_SIZE - 1 and y == 0:
            if grid[x-1][y] == EMPTY or grid[x][y+1] == EMPTY:
                tmp = True
        elif x == GRID_SIZE - 1 and y == GRID_SIZE - 1:
            if grid[x-1][y] == EMPTY or grid[x][y-1] == EMPTY:
                tmp = True
    # Agent on X border
    elif x in [0, GRID_SIZE- 1]:
        if x == 0:
            if grid[x+1][y] == EMPTY or grid[x][y-1] == EMPTY or grid[x][y+1] == EMPTY:
                tmp = True
        elif x == GRID_SIZE -1:
            if grid[x-1][y] == EMPTY or grid[x][y-1] == EMPTY or grid[x][y+1] == EMPTY:
                tmp = True
    # Agent on Y border
    elif y in [0, GRID_SIZE- 1]:
        if y == 0:
            if grid[x+1][y] == EMPTY or grid[x-1][y] == EMPTY or grid[x][y+1] == EMPTY:
                tmp = True
        elif y == GRID_SIZE -1:
            if grid[x+1][y] == EMPTY or grid[x-1][y] == EMPTY or grid[x][y-1] == EMPTY:
                tmp = True
    return tmp


# Modify the draw_grid function
def draw_grid(grid, bagents, kagents, bombs, screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_type = grid[x][y]
            image_path = CELL_IMAGES[cell_type]

            if image_path is not None:
                image = load_scaled_image(image_path)
                screen.blit(image, rect)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)

    for agent in bagents:
        if agent.alive:
            agent_rect = pygame.Rect(agent.x * CELL_SIZE, agent.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            image = load_scaled_image(CELL_IMAGES[BAGENT])
            screen.blit(image, agent_rect)
    for agent in kagents:
        if agent.alive:
            agent_rect = pygame.Rect(agent.x * CELL_SIZE, agent.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            image = load_scaled_image(CELL_IMAGES[KAGENT])
            screen.blit(image, agent_rect)
    for bomb in bombs:
        bomb_rect = pygame.Rect(bomb.x * CELL_SIZE, bomb.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        image = load_scaled_image(CELL_IMAGES[BOMB])
        screen.blit(image, bomb_rect)


class BombermanEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BombermanEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # 4 moves + plant bomb
        self.observation_space = spaces.Box(low=0, high=5, shape=(1, GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # Initialize game
        self.initialize_game()

    def initialize_game(self):
        self.clock = pygame.time.Clock()
        self.grid, self.bombers, self.killers = initialize_grid()
        self.bombs = []
        if TRAINING:
            self.render_mode = None
        else:
            self.render_mode = "human"
            screen.fill((0, 0, 0))
            draw_grid(self.grid, self.bombers, self.killers, self.bombs, screen)
        self.infos = {'wall': 0, 'kill': 0, 'bad_move': 0}

        print(self.grid)

    def step(self, action):
        # Implement the effect of an action and calculate reward
        reward = 0
        done = False

        for bomb in self.bombs[:]:
                bomb.tick()
                if bomb.has_exploded():
                    bomb.explode(self.grid, self.bombers, self.killers, self.infos)
                    self.bombs.remove(bomb)

        for agent in self.bombers:
            if agent.alive:
                agent.make_decision(action, self.grid, self.bombs, self.infos)

        for killerman in self.killers:
            if killerman.alive:
                killerman.move(self.grid)  # Move the Killerman agents

        # Check collisions for Bomberman and Killerman
        for bomber in self.bombers:
            if bomber.alive:
                bomber.check_collision_with_killerman(self.killers)

        # If bomber died, episode is concluded            
        for bomber in self.bombers:
            # Positive reward for killing a killer
            if self.infos['kill']:
                reward += 10

            # Positive reward for destroying a wall
            if self.infos['wall']:
                reward += 0.2

            # Negative reward for being killed either by bomb or killer
            if not bomber.alive:
                reward += -20
                done = True
                print('BOMBERMAN DIED :(')

            if all([not killer.alive for killer in self.killers]):
                reward += 50
                done = True
                print('ALL KILLED WOW')

            if not self.infos['kill'] and not self.infos['wall'] and bomber.alive and action != 4:
                reward += -0.1

            if self.infos['bad_move']:
                reward += -0.5

            if action == 4 and not (self.infos['kill'] or self.infos['wall']):
                reward += -0.5
            
            if bomber.check_bomb_radius(self.bombs):
                reward += -0.5

            # Small positive to encourage moving
            # else:
            #    reward += 0
            #    done = False
                
        #print("#### STEP INFO ####")
        #print("Chosen action", action)
        #print("Reward", reward)
        #print("Alive?", done)
        #print("Obs", self.grid)
        #print("##############")

        info = {}
        self.infos = {'wall': False, 'kill': False, 'bad_move': False}

        # Render
        if self.render_mode == "human":
           self.render()

        return np.expand_dims(self.grid, axis=0), reward, done, False, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.initialize_game() # Assuming initialize_grid() returns grid as first element
        return np.expand_dims(self.grid, axis=0), {seed:0}

    def render(self, mode='human', close=False):
        # Render the environment to the screen (if you want to visualize it)
        screen.fill((0, 0, 0))
        draw_grid(self.grid, self.bombers, self.killers, self.bombs, screen)  # Combine bombers and killermen in the agents list
        pygame.display.flip()
        self.clock.tick(5)  # Adjust the frame rate as needed
        # pass

# Function to load and scale images
def load_scaled_image(path):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))

def main():
    # grid, bombers, killers = initialize_grid()
    # bombs = []
    running = True

    # Gym env
    register(
        id='BombermanEnv',
        entry_point='learning_single:BombermanEnv',
        max_episode_steps=300
    )
    env = gym.make('BombermanEnv')
    input_shape = env.observation_space.shape[0]
    num_actions = env.action_space
    # Try new network with CNN
    #env = FlattenObservation(env) # Attempt, weird box obs
    #env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=4)

    done = False
    steps = 0

    # Stable baselines algorithm
    eval_callback = EvalCallback(
        env,
        n_eval_episodes=20,
        best_model_save_path="saved_policy",
        eval_freq=100000,
    )

    # Load custom policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard/", policy_kwargs=policy_kwargs) #DQN
    print(model.policy)
    model.learn(total_timesteps=20000000, callback = eval_callback)

    # Test
    TRAINING = False
    obs = env.reset()[0]
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        while not done:
            steps += 1

            # Stable baselines 
            # Predict function here, check how
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            env.render()

    pygame.quit()


if __name__ == "__main__":
    main()

