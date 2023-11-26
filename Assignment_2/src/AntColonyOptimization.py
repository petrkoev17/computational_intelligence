import time

import numpy as np
from matplotlib import pyplot as plt

from Assignment_2.src.Ant import Ant
from Maze import Maze
from PathSpecification import PathSpecification

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification, rho):
        self.maze.reset()
        shortest_path = None
        for generation in range(self.generations):
            print(generation)
            routes = np.array([])
            #closest_to_end_goal = np.inf
            for ant in range(self.ants_per_gen):
                curr_ant = Ant(self.maze, path_specification)
                ant_shortest_path = curr_ant.find_route()

                np.append(routes, ant_shortest_path)
                if shortest_path is None or shortest_path.size() > ant_shortest_path.size():
                    shortest_path = ant_shortest_path
            self.maze.add_pheromone_routes(routes, self.q, rho)

            # # # print(np.array(self.maze.pheromones_maze))
            # # # print(np.array(self.maze.walls))
            # trace = np.array(self.maze.pheromones_maze)
            # print(trace[trace != 0].sum())
            # # # plt.show()
            # # # # print(no_wall_trace.mean())
            # # plt.imshow(trace, cmap='hot', interpolation='nearest')
            # # plt.show()
        return shortest_path