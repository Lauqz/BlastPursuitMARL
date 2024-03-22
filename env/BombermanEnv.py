import copy
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from agents.BAgent import BAgent
from pettingzoo import ParallelEnv
from typing import Optional, overload
from agents.Killerman import Killerman
from gymnasium.utils import EzPickle, seeding
import utils

class BombermanEnv(ParallelEnv, EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BombermanEnv, self).__init__()
        EzPickle.__init__(self)

        # Initialize game
        self.initialize_game()
        # Observation space
        self.observation_spaces = gym.spaces.Dict(
                {
                    id: spaces.Box(low=0, high=1, shape=(2, utils.GRID_SIZE, utils.GRID_SIZE), dtype=np.float32) #np.int8
                    for id in self.mapping
                }
            )

        # Action space
        self.action_spaces = gym.spaces.Dict(
                {
                    id:(spaces.Discrete(5) 
                    if isinstance(value, BAgent) else spaces.Discrete(4)) 
                    for id,value in self.mapping.items()
                }
            )

    def initialize_game(self):
        # Initialize clock and entities
        self.clock = pygame.time.Clock()
        self.grid, self.bombers, self.killers = utils.initialize_grid()
        self.bombs = []

        # Visualizing
        if utils.TRAINING:
            self.render_mode = None
        else:
            self.screen = pygame.display.set_mode((utils.WIDTH, utils.HEIGHT))
            self.render_mode = "human"
            self.screen.fill((0, 0, 0))
            utils.draw_grid(self.grid, self.bombers, self.killers, self.bombs, self.screen)

        # Resetting data
        db = {'Bomber'+str(i):j for i,j in zip(range(len(self.bombers)),self.bombers)}
        dk = {'Killer'+str(i):j for i,j in zip(range(len(self.killers)),self.killers)}
        db.update(dk)
        self.mapping =  db
        self.possible_agents = list(self.mapping.keys())
        self.suicide = False
        self.info = {i: 0 for i in self.possible_agents}
        self.reward = {i: 0 for i in self.possible_agents}
        self.done = {i: 0 for i in self.possible_agents} 
        self.truncations = {i: 0 for i in self.possible_agents}     
        self.agents = [key for key in self.possible_agents]
        self.steps = 0
        self.max_steps = 50 

        # Useful infos for reward
        self.infos = {
                        'wall': {v: 0 for k,v in self.mapping.items()}, 
                        'kill': {v: 0 for k,v in self.mapping.items()}, 
                        'bad_move': {v: 0 for k,v in self.mapping.items()}
                    }

    def get_position(self, agent):
        base = np.full((utils.GRID_SIZE, utils.GRID_SIZE), utils.EMPTY)
        base[agent.x][agent.y] = 1
        return base

    @overload
    def observation_space(self):
        return self.observation_space
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def step(self, action):
        # Implement the effect of an action and calculate reward
        self.reward = {i: 0 for i in self.possible_agents}
        agents_map = [self.mapping[key] for key in self.agents]

        # Avoid loops
        self.steps += 1
        if self.steps >= self.max_steps:
            self.reward = {k:-10 for k in self.reward}
            self.done = {k:1 for k in self.done}
            self.truncations = {k:1 for k in self.truncations}

        # Suicide checker
        check_suicide = [b.alive for b in self.bombers]
        for bomb in self.bombs[:]:
                bomb.tick()
                if bomb.has_exploded():
                    # Implement multi agent of infos
                    self.grid, self.bombers, self.killers, self.infos = bomb.explode(self.grid, self.bombers, self.killers, self.infos)

                    # Check suicide
                    alive_after = [b.alive for b in self.bombers]
                    if check_suicide != alive_after:
                        self.suicide = True
                    self.bombs.remove(bomb)

        for agent in agents_map:
            map = [key for key, item in self.mapping.items() if item == agent]
            if agent.alive:
                self.grid, self.bombs, self.infos = agent.make_decision(action[map[0]], self.grid, self.bombs, self.infos)
            if isinstance(agent, BAgent):
                self.killers, self.infos = agent.check_collision_with_killerman(self.killers, self.infos)
            if isinstance(agent, Killerman):
                self.bombers, self.infos = agent.check_collision_with_bomber(self.bombers, self.infos)

        # If check rewards shared by each agent           
        for agent in agents_map:
            map = [key for key, item in self.mapping.items() if item == agent][0]
            # Positive reward for killing another agent
            if self.infos['kill'][agent]:
                self.reward[map] += 10

            # Positive reward for destroying a wall
            if self.infos['wall'][agent]:
                self.reward[map] += 0.2

            # Bad move (crushing into wall) reward
            if self.infos['bad_move'][agent]:
                self.reward[map] += -0.5

            # Negative reward for being killed either by bomb or killer
            if not agent.alive:
                self.reward[map] += -20
                self.done[map] = 1
                self.truncations[map] = 1
                print('AGENT DIED :(')

        if all([not killer.alive for killer in self.killers]):
            for b in self.bombers:
                map = [key for key, item in self.mapping.items() if item == b][0]
                self.reward[map] += 50
            self.done = {k:1 for k in self.done}
            self.truncations = {k:1 for k in self.truncations}
            print('ALL KILLERS KILLED WOW')

        if all([not bomber.alive for bomber in self.bombers]):
            if not self.suicide:
                for k in self.killers:
                    map = [key for key, item in self.mapping.items() if item == k][0]
                    self.reward[map] += 50
            self.done = {k:1 for k in self.done}
            self.truncations = {k:1 for k in self.truncations}
            print('ALL BOMBERMEN KILLED WOW')

        alive_bombers = [a for a in agents_map if isinstance(a,BAgent)]
        for b in alive_bombers:
            map = [key for key, item in self.mapping.items() if item == b][0]
            # Small negative reward to encourage exploration (if didn't kill, didn't destroy a wall or placed a bomb)
            if not self.infos['kill'][b] and not self.infos['wall'][b] and b.alive and action[map] != 4:
                self.reward[map] += -0.1

            # Small negative reward if bomb exploded and did not kill anybody or did not destroy a wall
            if action[map] == 4 and not (self.infos['kill'][b] or self.infos['wall'][b]):
                self.reward[map] += -0.1
            
            # Small negative reward if bomber stay within the bomb radius
            if b.check_bomb_radius(self.bombs):
                self.reward[map] += -0.2

            # Small positive reward if bomb stays within killer radius?

        alive_killers = [a for a in agents_map if isinstance(a,BAgent)]
        for k in alive_killers:
            map = [key for key, item in self.mapping.items() if item == k][0]
            # Small negative reward if killer stay within bomb radius
            if k.check_bomb_radius(self.bombs):
                self.reward[map] = -0.2       
                
        #print("#### STEP INFO ####")
        #print("Chosen action", action)
        #print("Reward", reward)
        #print("Alive?", done)
        #print("Obs", self.grid)
        #print("##############")
                
        self.agents = [k for k in self.done.keys() if self.done[k]!=1]
        self.infos = {
                        'wall': {v: 0 for k,v in self.mapping.items() if self.done[k]==0}, 
                        'kill': {v: 0 for k,v in self.mapping.items() if self.done[k]==0}, 
                        'bad_move': {v: 0 for k,v in self.mapping.items() if self.done[k]==0}
                    }

        # Render
        if self.render_mode == "human":
           self.render()

        # Use vectorized mapping to create a normalized array
        normalized_grid = np.vectorize(utils.VALUE_MAPPING.get, otypes=[np.float32])(copy.deepcopy(self.grid))
        obs = {i:np.stack([normalized_grid, self.get_position(self.mapping[i])], axis=0) for i in self.possible_agents}
        
        # Create observation
        #obs = {i:np.stack([self.grid, self.get_position(self.mapping[i])], axis=0) for i in self.possible_agents}

        return obs, self.reward, self.done, self.truncations, {i: {} for i in self.possible_agents}

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.initialize_game() # Assuming initialize_grid() returns grid as first element

        # Use vectorized mapping to create a normalized array
        normalized_grid = np.vectorize(utils.VALUE_MAPPING.get, otypes=[np.float32])(copy.deepcopy(self.grid))
        obs = {i:np.stack([normalized_grid, self.get_position(self.mapping[i])], axis=0) for i in self.possible_agents}

        # Create observation
        # obs = {i:np.stack([self.grid, self.get_position(self.mapping[i])], axis=0) for i in self.possible_agents}

        return obs, {i: {} for i in self.possible_agents}

    def render1(self):
        pass 

    def render(self, mode='human', close=False):
        # Render the environment to the screen (if you want to visualize it)
        self.screen.fill((0, 0, 0))
        utils.draw_grid(self.grid, self.bombers, self.killers, self.bombs, self.screen)  # Combine bombers and killermen in the agents list
        pygame.display.flip()
        self.clock.tick(5)  # Adjust the frame rate as needed