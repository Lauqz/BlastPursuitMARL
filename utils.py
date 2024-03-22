import random
import numpy as np
import pygame
from agents import BAgent
from agents import Killerman

# Global variables
GRID_SIZE = 7
INDESTRUCTIBLE_WALL_PERCENTAGE = 8
DESTRUCTIBLE_WALL_PERCENTAGE = 12
NUM_BOMBERAGENTS = 3 # 1 FOR NOW
NUM_KILLERAGENTS = 3
TRAINING = True

# Cell types
EMPTY = 0
INDESTRUCTIBLE_WALL = 1
DESTRUCTIBLE_WALL = 2
BAGENT = 3
BOMB = 4
KAGENT = 5

# Image paths
BOMBERMAN_IMAGE_PATH = 'img/bomberman.png'
KILLERMAN_IMAGE_PATH = 'img/killer.png'
INDESTRUCTIBLE_WALL_IMAGE_PATH = 'img/indestructiblewall.png'
DESTRUCTIBLE_WALL_IMAGE_PATH = 'img/destructiblewall.jpg'
BOMB_IMAGE_PATH = 'img/bomb.png'
VALUE_MAPPING = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1}

# Create a dictionary for image paths
CELL_IMAGES = {
    EMPTY: None,  # You can use None for empty cells
    INDESTRUCTIBLE_WALL: INDESTRUCTIBLE_WALL_IMAGE_PATH,
    DESTRUCTIBLE_WALL: DESTRUCTIBLE_WALL_IMAGE_PATH,
    BAGENT: BOMBERMAN_IMAGE_PATH,
    KAGENT: KILLERMAN_IMAGE_PATH,
    BOMB: BOMB_IMAGE_PATH,
}

# Set up display
CELL_SIZE = 20
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE


# Modify the draw_grid function
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
                    agent = BAgent.BAgent(x, y)
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
                    agent = Killerman.Killerman(x, y)
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

# Function to load and scale images
def load_scaled_image(path):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))