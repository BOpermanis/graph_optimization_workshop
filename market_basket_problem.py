import numpy as np
import random
from simpleai.search import SearchProblem, astar
from simpleai.search.local import hill_climbing
from collections import defaultdict
from itertools import chain
from pprint import pprint

from simpleai.search.utils import BoundedPriorityQueue, InverseTransformSampler
from simpleai.search.models import SearchNodeValueOrdered
from simpleai.search.local import _create_genetic_expander


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

    def get_best_N_plus(self, state):
        best_N_plus = None
        best_val = 0.0

        if len(state) >= self.k:
            return best_N_plus, best_val

        for item0 in state:
            for item in self.node2nodes[item0]:
                if state.issubset(self.node2nodes[item]):
                    basket = state | {item}
                    val = self.value(basket)
                    if val > best_val:
                        best_N_plus = basket

        return best_N_plus, best_val

    def get_N_0(self, state):
        best_N_0 = None
        best_val = 0.0
        for item0 in state:
            for item in self.node2nodes[item0]:
                if state.issubset(self.node2nodes[item]):
                    basket = state - {item0} | {item}
                    val = self.value(basket)
                    if val > best_val:
                        best_N_0 = basket
        return best_N_0, best_val

    def get_best_N_minus(self, state):
        best_N_minus = None
        best_val = 0.0
        for item0 in state:
            basket = state - {item0}
            val = self.value(basket)
            if val > best_val:
                best_N_minus = basket
        return best_N_minus, best_val

    def actions(self, state):
        switches = []
        for item0 in state:
            for item in self.node2nodes[item0]:
                if state.issubset(self.node2nodes[item]):
                    switches.append((item0, item))
        return switches

    def result(self, state, action):
        state = HashableSet(state)
        item_old, item_new = action
        state.remove(item_old)
        state.add(item_new)
        return state

    def value(self, state):
        s = 0.0
        for item1, item2 in combinations(state, 2):
            s += self.weights[key(item1, item2)]
        return s

    def heuristic(self, state):
        return self.value(state)

    def generate_random_state(self):

        item0 = random.choice([*self.nodes])
        clique = HashableSet([item0])
        # clique = {item0}
        neighbourhood = self.node2nodes[item0].copy()

        for _ in range(self.k-1):
            if len(neighbourhood) == 0:
                break
            item = random.choice([*neighbourhood])
            clique.add(item)
            neighbourhood.intersection_update(self.node2nodes[item])

        return clique

    def is_goal(self, state):
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