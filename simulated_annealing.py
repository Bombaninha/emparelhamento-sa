from __future__ import annotations
import argparse
import sys
import math
import random as rd
from typing import List, Set, Union, Dict
import os

MAX_NEIGHBOR_ITER = 3000

class Edge:
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color
        self.deggre = 0

    def has_same_attribute(self, edge: Edge) -> bool:
        vertices = [self.vertex_u, self.vertex_v]
        return edge.color == self.color or edge.vertex_u in vertices or edge.vertex_v in vertices

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

def greedy_initial_solution(edges: List[Edge], deggre_counter: Dict[int, int]) -> Set[Edge]:
    # Para cada aresta, calcula o grau médio e coloca na aresta
    for edge in edges:
        edge.deggre = (deggre_counter[edge.vertex_u] + deggre_counter[edge.vertex_v]) / 2
    
    # Ordena as arestas, para buscar de maneira gulosa
    sorted_edges = sorted(edges, key=lambda x: x.deggre, reverse=False)

    edges = set()
    edges.add(sorted_edges.pop(0))

    for edge in sorted_edges:
        if not any(edge.has_same_attribute(e) for e in edges):
            edges.add(edge)
            

    return edges

def main(parser: argparse.ArgumentParser) -> None:
    options = parser.parse_args()
    if not options.filepath:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()

    with open(options.filepath, 'r') as file:
        lines = file.readlines()
        n_vertices = int(lines[1].strip())

        graph = [list() for _ in range(n_vertices + 1)]

        edge_list = []
        deggre_counter = {}

        for line in lines[4:]:
            vertex_u, vertex_v, color = list(map(lambda x: int(x.strip().replace(' ', '')), line.strip().split('  ')))
            graph[vertex_u] += [vertex_v]
            graph[vertex_v] += [vertex_u]

            if vertex_u in deggre_counter:
                deggre_counter[vertex_u] += 1
            else:
                deggre_counter[vertex_u] = 1

            if(vertex_v in deggre_counter):
                deggre_counter[vertex_v] += 1
            else:
                deggre_counter[vertex_v] = 1

            edge_list.append(Edge(vertex_u, vertex_v, color))

        initial_sol = greedy_initial_solution(edge_list, deggre_counter)
        print(initial_sol)
        solution = simulated_annealing(initial_sol, edge_list, 1000, 0.001, 0.95)
        print(f"Solução: {len(solution)}")
        test_solution(solution)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instance file (mandatory)")

    main(parse)
