import utils 

# Killerman Class
class Killerman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    def make_decision(self, action, grid, bombs, infos):
        directions = {0:'up', 1:'down', 2:'left', 3:'right'}
        grid, infos = self.move(directions[action], grid, infos)

        return grid, bombs, infos

    def move(self, direction, grid, infos):
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
            if grid[new_x][new_y] in [utils.EMPTY, utils.BAGENT, utils.BOMB]:
                grid[self.x][self.y] = utils.EMPTY
                grid[new_x][new_y] = utils.KAGENT
                self.x, self.y = new_x, new_y
            elif grid[new_x][new_y] in [utils.INDESTRUCTIBLE_WALL, utils.DESTRUCTIBLE_WALL, utils.KAGENT]:
                infos['bad_move'][self] = True
        else:
            infos['bad_move'][self] = True
        
        return grid, infos

    def check_bomb_radius(self, bombs):
        for bomb in bombs: 
            for dx in range(bomb.x - bomb.radius, bomb.x + bomb.radius + 1):
                for dy in range(bomb.y -bomb.radius, bomb.y + bomb.radius + 1):
                    if (self.x,self.y) == (dx,dy):
                        return True
                    else:
                        return False
    
    def check_collision_with_bomber(self, bombermen, infos):
        for bomber in bombermen:
            if bomber.alive and self.x == bomber.x and self.y == bomber.y:
                bomber.alive = False
                infos['kill'][self] = True
        
        return bombermen, infos