import utils

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
        if grid[self.x][self.y] == utils.BOMB:
            grid[self.x][self.y] = utils.EMPTY
            
        # Affect surrounding cells
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < utils.GRID_SIZE and 0 <= ny < utils.GRID_SIZE:
                    if grid[nx][ny] == utils.DESTRUCTIBLE_WALL:
                        grid[nx][ny] = utils.EMPTY
                        for b in bagents:
                            infos['wall'][b] = True
                    elif grid[nx][ny] == utils.BAGENT:
                        for agent in bagents:
                            if agent.x == nx and agent.y == ny:
                                agent.alive = False
                    elif grid[nx][ny] == utils.KAGENT:
                        for agent in kagents:
                            if agent.x == nx and agent.y == ny:
                                agent.alive = False
                                for b in bagents:
                                    infos['kill'][b] = True
        for agent in bagents:
            if self.x - self.radius <= agent.x <= self.x + self.radius and \
               self.y - self.radius <= agent.y <= self.y + self.radius:
                agent.alive = False
        for agent in kagents:
            if self.x - self.radius <= agent.x <= self.x + self.radius and \
               self.y - self.radius <= agent.y <= self.y + self.radius:
                agent.alive = False
        
        return grid, bagents, kagents, infos
