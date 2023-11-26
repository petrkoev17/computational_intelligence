import math
import random

import numpy as np

from TSPData import TSPData


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        number_of_products = len(tsp_data.product_locations)

        # print("distances")
        # print(tsp_data.distances)

        # Step 1: Encode into chromosome
        chromosome = self.make_chromosome(number_of_products)

        # print("chromosome")
        # print(chromosome)
        # print("chromosome length")
        # print(len(chromosome))

        # Step 2: Calculate fitness
        fitness = self.fitness(tsp_data, chromosome)

        print("fitness")
        print(fitness)

        # Step 3: Randomly select an initial population of size pop_size number of chromosomes
        population = self.select_population(chromosome)

        # print("population")
        # print(population)

        for i in range(self.generations):
            # Step 4: Compute fitness for all chromosomes + fitness ratios
            fitness_of_population, fitness_ratios = self.compute_fitness_and_ratio(tsp_data, population)

            # print("fitness of population")
            # print(fitness_of_population)
            # print("fitness ratios")
            # print(fitness_ratios)

            # if i % 10 == 0:
            print(f"Generation: {i}")
            print(max(fitness_of_population))
            # if max(fitness_of_population) > 8635.578583765113:
            #     break

            elite_population = [x for _, x in sorted(zip(fitness_of_population, population), reverse=True)]
            offspring = elite_population[:10].copy()
            while len(offspring) < self.pop_size:
                # Step 5: Pick chromosomes for reproduction (Roulette Wheel Selection)
                chromosomes_for_reproduction = self.pick_chromosomes_roulette(fitness_ratios, population)

                # print("selected chromosomes for reproduction")
                # print(chromosomes_for_reproduction)

                # Step 6a: Crossover
                pc = 0.7
                crossed = self.crossover(chromosomes_for_reproduction, pc)
                offspring.append(crossed[0])
                offspring.append(crossed[1])

                # print("(potentially) crossed")
                # print(crossed)

            # print("offspring")
            # print(offspring)

            mutated_offspring = []

            for chrom in offspring:
                # Step 6b: Mutation
                pm = 1/18
                mutated = self.mutate(chrom, pm)
                mutated_offspring.append(mutated)

            # print("(potentially) mutated offspring")
            # print(mutated_offspring)

            # Step 7: Put the offspring in the new population
            new_population = list(mutated_offspring)

            # print("new population")
            # print(new_population)

            # Step 8: Substitute the old population with the new one
            population = list(new_population)

        fitness_for_last_population, last_fitness_ratios = self.compute_fitness_and_ratio(tsp_data, population)

        # print("final fitness")
        # print(fitness_for_last_population)
        # print("final ratios")
        # print(last_fitness_ratios)

        print("final population")
        print(population)

        index = 0
        max_fitness = 0

        for i, entry in enumerate(fitness_for_last_population):
            if entry > max_fitness:
                max_fitness = entry
                index = i

        optimal_solution_encoded = population[index]

        print("max fitness")
        print(max_fitness)

        print("length")
        print(10000000 / max_fitness)

        result = []

        for i in range(len(optimal_solution_encoded)):
            decoded = int(optimal_solution_encoded[i], 2)
            result.append(decoded)

        print(result)

        return result

    def make_chromosome(self, number_of_products):
        chromosome = []

        for i in range(number_of_products):
            chromosome.append(bin(i)[2:].zfill(math.ceil(math.log2(number_of_products))))

        return chromosome

    def fitness(self, tsp_data, chromosome):
        decoded = []

        for i in range(len(chromosome)):
            decoded.append(int(chromosome[i], 2))

        # tsp_data.distances[first][second]
        sum = tsp_data.get_start_distances()[decoded[0] - 1]

        for i in range(len(decoded) - 1):
            sum += tsp_data.get_distances()[decoded[i] - 1][decoded[i + 1] - 1]
        sum += tsp_data.get_end_distances()[decoded[len(decoded) - 1] - 1]

        return 10000000 / sum

    def select_population(self, chromosome):
        population = []

        for i in range(self.pop_size):
            shuffled = list(chromosome)
            random.shuffle(shuffled)
            population.append(shuffled)

        return population

    def compute_fitness_and_ratio(self, tsp_data, population):
        fitness_of_population = []

        for chrom in population:
            ft = self.fitness(tsp_data, chrom)
            fitness_of_population.append(ft)

        # Compute fitness ratios
        fitness_ratios = []
        sum_of_all_fitness = np.sum(fitness_of_population)

        for ft in fitness_of_population:
            fitness_ratios.append(ft * 100 / sum_of_all_fitness)

        return fitness_of_population, fitness_ratios

    def pick_chromosomes_roulette(self, fitness_ratios, population):
        cumulative_ratios = []
        cumulative_ratios.append(fitness_ratios[0])

        for i in range(1, len(fitness_ratios)):
            cumulative_ratios.append(cumulative_ratios[i - 1] + fitness_ratios[i])

        # print("cumulative ratios")
        # print(cumulative_ratios)

        selected_chromosomes = []

        for i in range(2):
            random_num = random.uniform(0, 1) * 100

            index = next(idx for idx, ratio in enumerate(cumulative_ratios) if ratio >= random_num)

            selected_chromosomes.append(population[index])

        return selected_chromosomes

    def crossover(self, chromosomes_for_reproduction, probability):
        if random.uniform(0, 1) < probability:
            new_chromosomes = []

            tour1 = chromosomes_for_reproduction[0]
            tour2 = chromosomes_for_reproduction[1]

            size = len(tour1)

            # choose two random numbers for the start and end indices of the slice
            # (one can be at index "size")
            number1 = random.randint(1, size - 1)
            number2 = random.randint(1, size - 1)

            while number1 == number2:
                number2 = random.randint(1, size - 1)

            # make the smaller the start and the larger the end
            start = min(number1, number2)
            end = max(number1, number2)

            # instantiate two child tours
            child1 = []
            child2 = []

            # add the sublist in between the start and end points to the children
            child1.extend(tour1[start:end])
            child2.extend(tour2[start:end])

            # iterate over each city in the parent tours
            for i in range(size):
                # get the index of the current city
                current_city_index = (end + i) % size

                # get the city at the current index in each of the two parent tours
                current_city_in_tour1 = tour1[current_city_index]
                current_city_in_tour2 = tour2[current_city_index]

                # if child 1 does not already contain the current city in tour 2, add it
                if current_city_in_tour2 not in child1:
                    child1.append(current_city_in_tour2)

                # if child 2 does not already contain the current city in tour 1, add it
                if current_city_in_tour1 not in child2:
                    child2.append(current_city_in_tour1)

            # rotate the lists so the original slice is in the same place as in the
            # parent tours
            child1 = child1[start:] + child1[:start]
            child2 = child2[start:] + child2[:start]

            # add the cities that are not in the slice from the other parent
            for i in range(size):
                if tour1[i] not in child2:
                    child2.append(tour1[i])
                if tour2[i] not in child1:
                    child1.append(tour2[i])

            # append the new child chromosomes to the new_chromosomes list
            new_chromosomes.append(child1)
            new_chromosomes.append(child2)

            # if new_chromosomes[0] == chromosomes_for_reproduction[0] and new_chromosomes[1] == chromosomes_for_reproduction[1]:
            #     print("kur")


            return new_chromosomes

        return chromosomes_for_reproduction



    def mutate(self, chromosome, probability):
        if random.uniform(0, 1) < probability:
            helper = list(chromosome)

            rand_index1 = random.randint(0, len(helper) - 1)
            rand_index2 = random.randint(0, len(helper) - 1)

            while rand_index1 == rand_index2:
                rand_index2 = random.randint(0, len(helper) - 1)

            temp = helper[rand_index1]
            helper[rand_index1] = helper[rand_index2]
            helper[rand_index2] = temp

            return helper

        return chromosome


if __name__ == "__main__":
    population_size = 100
    generations = 10000
    persist_file = "./../data/optimal_tsp"

    # Setup optimization
    tsp_data = TSPData.read_from_file(persist_file)
    ga = GeneticAlgorithm(generations, population_size)

    # Run optimzation and write to file
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "./../data/tsp_solution.txt")
