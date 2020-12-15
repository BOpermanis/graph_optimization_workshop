import pickle
import numpy as np
from simpleai.search.local import hill_climbing, beam, hill_climbing_stochastic, beam_best_first, hill_climbing_random_restarts, simulated_annealing


from market_basket_problem import MarketBasketProblem
from utils import check_if_clique

with open("similarities.pickle", "rb") as conn:
    weights = pickle.load(conn)


basket_size = 2
problem = MarketBasketProblem()
problem.initialize_graph(weights, basket_size)

result = hill_climbing(problem, iterations_limit=100)
# result = beam(problem, iterations_limit=100)
# result = hill_climbing_stochastic(problem, iterations_limit=100)
# result = beam_best_first(problem, iterations_limit=100)
# result = hill_climbing_random_restarts(problem, iterations_limit=100)
# result = simulated_annealing(problem, iterations_limit=100)

# with open("local_search_result.pickle", "wb") as conn:
#     pickle.dump((result, basket_size), conn)

print(result.state, problem.value(result.state))