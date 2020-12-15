from itertools import combinations

def key(a, b):
    if a > b:
        return b + "||" + a
    return a + "||" + b

def check_if_clique(state, node2nodes):
    for n1, n2 in combinations(state, 2):
        if n1 not in node2nodes[n2]:
            return False
    return True


class HashableSet(set):
    def __hash__(self):
        return "||".join(sorted(list(self))).__hash__()


if __name__ == "__main__":
    s1 = HashableSet(["dog", "cat", "3"])
    s2 = HashableSet(["rex", "cat", "3"])
    u = set()
    u.add(s1)
    print(s2 in u)
    u.add(s2)
    print(s1 in u)