import pickle
from simpleai.search import breadth_first, depth_first
from simpleai.search.traditional import greedy, iterative_limited_depth_first, astar

from market_basket_problem import MarketBasketProblem

with open("similarities.pickle", "rb") as conn:
    weights = pickle.load(conn)

with open("local_search_result.pickle", "rb") as conn:
    result, basket_size = pickle.load(conn)

print("basket_size: ", basket_size)
problem = MarketBasketProblem()
problem.initialize_graph(weights, basket_size, is_goal_solution=result.state)

result = breadth_first(problem)
# result = depth_first(problem)
# result = greedy(problem)
# result = iterative_limited_depth_first(problem)
# result = astar(problem)


print(result.state, problem.value(result.state))