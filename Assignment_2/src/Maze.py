import traceback
import sys

import matplotlib.pyplot as plt
import numpy as np

from Assignment_2.src.Coordinate import Coordinate


# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.pheromones_maze = None
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.initialize_pheromones()
        self.pheromone_gains = []

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromones_maze = self.walls

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param route The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        curr_coordinate = Coordinate(route.get_start().x, route.get_start().y)
        for single_route_cell in route.get_route():
            curr_coordinate.add_direction(single_route_cell)
            curr_x = curr_coordinate.get_x()
            curr_y = curr_coordinate.get_y()
            # print(q / route.size())
            self.pheromones_maze[curr_x][curr_y] += q / route.size()
        self.pheromone_gains.append(q / route.size())
        return

     # Update pheromones for a list of routes
     # @param routes A list of routes
     # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q, rho):
        self.evaporate(rho)
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
       evaporation_proportion = 1 - rho
       for i in range(self.width):
           for j in range(self.length):
               self.pheromones_maze[i][j] = self.pheromones_maze[i][j] * evaporation_proportion
       return

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        current_x = position.get_x()
        current_y = position.get_y()
        res = np.zeros(shape=4)

        if current_x == (self.width - 1):
            east_prob = 0
        else:
            east_prob = self.get_pheromone(Coordinate(position.get_x() + 1, position.get_y()))
        res[0] = east_prob

        if current_y == 0:
            north_prob = 0
        else:
            north_prob = self.get_pheromone(Coordinate(position.get_x(), position.get_y() - 1))
        res[1] = north_prob

        if current_x == 0:
            west_prob = 0
        else:
            west_prob = self.get_pheromone(Coordinate(position.get_x() - 1, position.get_y()))
        res[2] = west_prob

        if current_y == (self.length - 1):
            south_prob = 0
        else:
            south_prob = self.get_pheromone(Coordinate(position.get_x(), position.get_y() + 1))
        res[3] = south_prob

        return res

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        return self.pheromones_maze[pos.get_x()][pos.get_y()]

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()