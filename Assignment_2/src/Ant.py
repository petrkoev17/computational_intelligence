import random
import numpy as np

from Assignment_2.src.Coordinate import Coordinate
from Route import Route
from Direction import Direction

#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(Coordinate(self.start.x, self.start.y))

        directions = np.array([Direction.east, Direction.north, Direction.west, Direction.south])
        visited = []
        # FIX HARDCODED 100
        while self.current_position != self.end and route.size() < 0.25 * self.maze.length * self.maze.width:
            np.append(visited, self.current_position)
            surrounding_pheromones = self.maze.get_surrounding_pheromone(self.current_position)
            if self.current_position.add_direction(Direction.east) in visited:
                surrounding_pheromones[0] = 0
            if self.current_position.add_direction(Direction.north) in visited:
                surrounding_pheromones[1] = 0
            if self.current_position.add_direction(Direction.west) in visited:
                surrounding_pheromones[2] = 0
            if self.current_position.add_direction(Direction.south) in visited:
                surrounding_pheromones[3] = 0
            if np.all(surrounding_pheromones == 0):
                last_direction = route.remove_last()
                self.current_position = self.current_position.subtract_direction(last_direction)
            else:
                if route.size() > 0:
                    last_direction = route.get_last()
                    surrounding_pheromones[Direction.dir_to_int(last_direction)] *= 1.5
                surrounding_pheromones = surrounding_pheromones ** 1 / np.sum(surrounding_pheromones)
                choice = self.rand.choices(directions, weights=surrounding_pheromones.tolist(), k = 1)[0]
                self.current_position = self.current_position.add_direction(choice)
                visited.append(self.current_position)
                route.add(choice)

        return route
