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
from utils import key
import math
import random


def check_if_clique(state, node2nodes):
    for n1, n2 in combinations(state, 2):
        if n1 not in node2nodes[n2]:
            return False
    return True

class MWC(SearchProblem):

    def initialize_graph(self, weights, k):
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

        return self.nodes - s_not

    def result(self, state, action):
        state.add(action)
        return state

    def value(self, state):
        s = 0.0
        for item1, item2 in combinations(state, 2):
            s += self.weights[key(item1, item2)]
        return s

    def generate_random_state(self):

        item0 = random.choice([*self.nodes])
        clique = {item0}
        neighbourhood = self.node2nodes[item0].copy()

        for _ in range(self.k-1):
            item = random.choice([*neighbourhood])
            clique.add(item)
            neighbourhood.intersection_update(self.node2nodes[item])

        return clique


def tabu_search(problem, n_iter=100):
    basket = problem.generate_random_state()
    val_best = problem.value(basket)
    basket_best = basket.copy()

    list_tabu = [basket]

    # for _ in range(n_iter):
    #     best_N_plus, val = problem.get_best_N_plus(basket)
    #     if val > val_best:
    #         basket =

if __name__ == "__main__":
    from simpleai.search.local import hill_climbing
    with open("similarities.pickle", "rb") as conn:
        weights = pickle.load(conn)

    problem = MWC()
    problem.initialize_graph(weights, 5)

    # result = hill_climbing(problem, 10)
    result = tabu_search(problem, 100)
    print(result)


