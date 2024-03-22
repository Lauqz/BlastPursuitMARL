from agents.Bomb import Bomb
import utils 

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
        if 0 <= new_x < utils.GRID_SIZE and 0 <= new_y < utils.GRID_SIZE:
            if grid[new_x][new_y] in [utils.EMPTY, utils.BOMB]:
                grid[self.x][self.y] = utils.EMPTY
                grid[new_x][new_y] = utils.BAGENT
                self.x, self.y = new_x, new_y
            elif grid[new_x][new_y] in [utils.KAGENT]:
                grid[self.x][self.y] = utils.EMPTY
                grid[new_x][new_y] = utils.BAGENT
                self.x, self.y = new_x, new_y 
                infos['bad_move'][self] = True
            elif grid[new_x][new_y] in [utils.INDESTRUCTIBLE_WALL, utils.DESTRUCTIBLE_WALL, utils.BAGENT]:
                infos['bad_move'][self] = True
        else:
            infos['bad_move'][self] = True

        return grid, infos

    def make_decision(self, action, grid, bombs, infos):
        if action == 4:  # 20% chance to plant a bomb
            bombs, grid = self.plant_bomb(bombs, grid)
        else:
            directions = {0:'up', 1:'down', 2:'left', 3:'right'}
            grid, infos = self.move(directions[action], grid, infos)
        
        return grid, bombs, infos

    def plant_bomb(self, bombs, grid):
        #if grid[self.x][self.y] EMPTY:
        bomb = Bomb(self.x, self.y)
        bombs.append(bomb)
        grid[self.x][self.y] = utils.BOMB
        return bombs, grid

    def check_collision_with_killerman(self, killermen, infos):
        for killerman in killermen:
            if killerman.alive and self.x == killerman.x and self.y == killerman.y:
                self.alive = False
                infos['kill'][killerman] = True
        
        return killermen, infos

    def check_bomb_radius(self, bombs):
        for bomb in bombs: 
            for dx in range(bomb.x - bomb.radius, bomb.x + bomb.radius + 1):
                for dy in range(bomb.y -bomb.radius, bomb.y + bomb.radius + 1):
                    if (self.x,self.y) == (dx,dy):
                        return True
                    else:
                        return False