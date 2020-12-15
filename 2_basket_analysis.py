import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint
import pickle

from utils import key

tab = pd.read_csv("Groceries_dataset.csv")
tab.fillna("", inplace=True)

item2clients = defaultdict(set)
for _, row in tab.iterrows():
    item2clients[row['itemDescription']].add(row['Member_number'])

weights = {}
for i1, (item1, clients1) in enumerate(item2clients.items()):
    for i2, (item2, clients2) in enumerate(item2clients.items()):
        if i2 > i1:
            intersection = len(clients1 & clients2)
            union = len(clients1) + len(clients2) - intersection

            if union > 0 and intersection > 0:
                weights[key(item1, item2)] = intersection / union

with open("similarities.pickle", "wb") as conn:
    pickle.dump(weights, conn)

