import time

from AntColonyOptimization import AntColonyOptimization
from GeneticAlgorithm import GeneticAlgorithm
from Maze import Maze
from PathSpecification import PathSpecification
from TSPData import TSPData

if __name__ == "__main__":
    gen = 30
    ants_per_gen = 15
    q = 500
    evap = 0.2

    # Construct the optimization objects
    maze = Maze.create_maze("./../data/hard_maze.txt")
    spec = PathSpecification.read_coordinates("./../data/hard_coordinates.txt")
    aco = AntColonyOptimization(maze, ants_per_gen, gen, q, evap)

    maze.start = spec.start
    maze.end = spec.end

    # maze.add_pheromone_route([Coordinate(0,0), Coordinate(0,4)], 10)
    # print(maze.pheromones_maze)
    # maze.evaporate(0.5)
    # print(maze.pheromones_maze)


    # Save starting time
    start_time = int(round(time.time() * 1000))

    # Run optimization
    shortest_route = aco.find_shortest_route(spec, evap)
    print(shortest_route.size())

    # Print time taken
    print("Time taken: " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    # Save solution
    shortest_route.write_to_file("./../data/easy_solution.txt")

    # Print route size
    print("Route size: " + str(shortest_route.size()))