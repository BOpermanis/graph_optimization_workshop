import numpy as np
from simpleai.search import SearchProblem
from collections import defaultdict
from pprint import pprint

from itertools import combinations
import pickle
from utils import key, check_if_clique, HashableSet
import math
import random


class MarketBasketProblem(SearchProblem):

    def initialize_graph(self, weights, k, is_goal_solution=None):
        self.k = k

        self.node2nodes = defaultdict(set)
        self.weights = weights

        self.nodes = set()
        for pairs in weights:
            self.nodes.update(pairs.split("||"))

        for edge in self.weights:
            i, j = edge.split("||")
            self.node2nodes[i].add(j)
            self.node2nodes[j].add(i)

        if is_goal_solution is not None:
            self.is_goal_state = is_goal_solution
            self.is_goal_value = self.value(is_goal_solution)

        self.initial_state = self.generate_random_state()

    def actions(self, state):
        # TODO
        switches = []
        return switches

    def result(self, state, action):
        # TODO
        return state

    def value(self, state):
        s = 0.0
        # TODO
        return s

    def heuristic(self, state):
        # man truukst izteeles :D
        return self.value(state)

    def generate_random_state(self):

        clique = HashableSet()
        # TODO

        return clique

    def is_goal(self, state):
        # prieksh salidzinashanas ar eksakto metodi
        return state == self.is_goal_state or self.is_goal_value <= self.value(state)


def checking_problem():
    with open("similarities.pickle", "rb") as conn:
        weights = pickle.load(conn)

    problem = MarketBasketProblem()
    problem.initialize_graph(weights, 10)

    state = problem.generate_random_state()
    state.__hash__()
    assert check_if_clique(state, problem.node2nodes)
    actions = problem.actions(state)
    for action in actions:
        state1 = problem.result(state, action)
        state1.__hash__()
        assert check_if_clique(state1, problem.node2nodes)

if __name__ == "__main__":
    checking_problem()