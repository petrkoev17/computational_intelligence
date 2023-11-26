import random
import numpy as np

from Assignment_3.src.Agent import Agent
from Assignment_3.src.Maze import Maze
from Assignment_3.src.MyQLearning import MyQLearning


class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        # TODO to select an action at random in State s
        valid_states = maze.get_valid_actions(agent)
        return random.choice(valid_states)

    def get_best_action(self, agent, maze, q_learning):
        # TODO to select the best possible action currently known in State s.
        valid_actions = maze.get_valid_actions(agent)
        q_scores = q_learning.get_action_values(maze.get_state(agent.x, agent.y), valid_actions)
        max_score = np.max(q_scores)
        best_actions = []

        max_actions = zip(valid_actions, q_scores)
        for x in max_actions:
            if x[1] == max_score:
                best_actions.append(x[0])

        random_best_action = random.choice(best_actions)

        return random_best_action

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        # TODO to select between random or best action selection based on epsilon.
        if random.random() < epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)

