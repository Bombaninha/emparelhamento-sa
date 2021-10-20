from __future__ import annotations
import argparse
import sys
import math
import random as rd
from typing import List, Set, Union
import os

import adjacency_list
import adjacency_matrix

import Counter

from collections import defaultdict

MAX_NEIGHBOR_ITER = 3000

class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(list(self.keys())) == 0: return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = list(self.items())
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in list(self.keys()):
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in list(y.items()):
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

class Edge:
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color

    def has_same_attribute(self, edge: Edge) -> bool:
        return edge.vertex_u == self.vertex_u or edge.vertex_v == self.vertex_v or edge.color == self.color

    def __repr__(self) -> str:
        return f"({self.vertex_u},{self.vertex_v})[{self.color}][Gm(e) = {self.deggre}]"

    def __str__(self) -> str:
        return f"Edge({self.vertex_u},{self.vertex_v})[{self.color}]"

def get_neighbor(sol: Set[Edge], edge_list: List[Edge]) -> Set[Edge]:
    n_edges = len(edge_list)
    for _ in range(MAX_NEIGHBOR_ITER):
        edge = edge_list[rd.randint(0, n_edges - 1)]
        if edge in sol:
            sol.remove(edge)
            break

        # if the edge is not in set, try to add it if it doesn't violate
        # diversified matching restrictions
        can_increase = not any(edge.has_same_attribute(e) for e in sol)
        if can_increase:
            sol.add(edge)
            break

    return sol

def metropolis(sol: Set[Edge], temp: float, it: int, best: Set[Edge],
               edge_list: List[Edge]) -> Union[Set[Edge], Set[Edge]]:
    for _ in range(it):
        new_sol: Set[Edge] = get_neighbor(sol, edge_list)
        delta = len(sol) - len(new_sol) # since we want to maximize, invert difference calculation
        if rd.random() < min(math.exp(- delta / temp), 1.0):
            sol = new_sol

        if len(best) < len(sol):
            best = sol.copy()

    return sol, best

def simulated_annealing(sol: Set[Edge], edge_list: List[Edge],
                        init_temp: float, end_temp: float, decrease: float) -> Set[Edge]:
    best = sol.copy()
    temp = init_temp
    while end_temp < temp:
        sol, best = metropolis(sol, temp, 100, best, edge_list)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{temp = } ---- current solution: {len(sol)} ---- best solution found: {len(best)}")
        temp *= decrease

    return best

def test_solution(sol: Set[Edge]) -> None:
    test_sol = sol.copy()
    for edge in sol:
        test_sol.remove(edge)
        if any(edge.has_same_attribute(e) for e in test_sol):
            print("Something went wrong")
            print(f"{edge = } has at least on attribute in common with another edge.")

def greedy_initial_solution(edges):
    debug = False
    # Inicializa um dicionário vazio com valor inicial 0
    deggre_counter = Counter()
    
    # Para cada aresta, computa vezes em que os vértices aparecem
    for edge in edges:
        deggre_counter[edge.vertex_u] += 1
        deggre_counter[edge.vertex_v] += 1

    # Para cada aresta, calcula o grau médio e coloca na aresta
    for edge in edges:
        edge.deggre = (deggre_counter[edge.vertex_u] + deggre_counter[edge.vertex_v]) / 2
    
    # Ordena as arestas, para buscar de maneira gulosa
    sorted_edges = sorted(edges, key=lambda x: x.deggre, reverse=False)

    if(debug):
        print(sorted_edges)

    edges = set()
    edges.add(sorted_edges.pop(0))

    def can_add(edge, edges):
        colors = set()
        vertexes = set()

        for e in edges:
            colors.add(e.color)
            vertexes.add(e.vertex_v)
            vertexes.add(e.vertex_u)

        return edge.color not in colors and (edge.vertex_u not in vertexes and edge.vertex_v not in vertexes)


    for edge in sorted_edges:
        if(can_add(edge, edges)):
            edges.add(edge)
            if(debug):
                print(f"[OK] Aresta { edge } adicionada!")
        else:
            if(debug):
                print(f"[ERROR] Aresta { edge } não pode ser adicionada!")

    return edges

def main(parser: argparse.ArgumentParser) -> None:
    options = parser.parse_args()
    if not options.filepath:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()

    with open(options.filepath, 'r') as file:
        file.readline() #ignoring first line
        n_vertices = int(file.readline())
        graph = [list() for _ in range(n_vertices + 1)]
        def append_line(line):
            vertex_u = int(line[0])
            vertex_v = int(line[1])
            color = int(line[2])
            graph[vertex_u] += [vertex_v]
            graph[vertex_v] += [vertex_u]
            return Edge(vertex_u, vertex_v, color)

        file.readline() #ignoring next two lines
        file.readline()
        edge_list = list(map(append_line, [line.split() for line in file]))
        
        #print(greedy_initial_solution(edge_list))
        
        #solution = simulated_annealing(set(), edge_list, 1000, 0.001, 0.95)
        #print(f"Solução: {len(solution)}")
        #test_solution(solution)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instance file (mandatory)")

    main(parse)
