from __future__ import annotations
import argparse
import sys
import math
import random as rd
from typing import List, Set, Union
import os

import adjacency_list
import adjacency_matrix

MAX_NEIGHBOR_ITER = 3000

class Edge:
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color

    def has_same_attribute(self, edge: Edge) -> bool:
        return edge.vertex_u == self.vertex_u or edge.vertex_v == self.vertex_v or edge.color == self.color

    def __repr__(self) -> str:
        return f"({self.vertex_u},{self.vertex_v})[{self.color}]"

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

        solution = simulated_annealing(set(), edge_list, 1000, 0.001, 0.95)
        print(f"Solução: {len(solution)}")
        test_solution(solution)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instance file (mandatory)")

    main(parse)
