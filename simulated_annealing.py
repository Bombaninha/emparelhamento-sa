from __future__ import annotations
import argparse
import sys
import math
from datetime import datetime
import random as rd
from typing import List, Set, Union, Dict
import os

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
    for _ in range(n_edges):
        edge = rd.choice(edge_list)
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

def simulated_annealing(sol: Set[Edge], edge_list: List[Edge], metropolis_it: int,
                        init_temp: float, end_temp: float, discount: float, echo: bool = False) -> Set[Edge]:
    best = sol.copy()
    temp = init_temp
    while end_temp < temp:
        sol, best = metropolis(sol, temp, metropolis_it, best, edge_list)
        if echo:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{temp = } ---- current solution: {len(sol)} ---- best solution found: {len(best)}")
        temp *= discount

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
    opt = parser.parse_args()
    if not opt.filepath:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()

    print(f"Solving problem for instance {opt.filepath}.")

    with open(opt.filepath, 'r') as file:
        lines = file.readlines()
        n_vertices = int(lines[1].strip())

        graph = [list() for _ in range(n_vertices + 1)]

        edge_list = []
        deggre_counter = {}

        for line in lines[4:]:
            vertex_u, vertex_v, color = list(map(int, line.split()))
            graph[vertex_u] += [vertex_v]
            graph[vertex_v] += [vertex_u]

            if vertex_u in deggre_counter:
                deggre_counter[vertex_u] += 1
            else:
                deggre_counter[vertex_u] = 1

            if vertex_v in deggre_counter:
                deggre_counter[vertex_v] += 1
            else:
                deggre_counter[vertex_v] = 1

            edge_list.append(Edge(vertex_u, vertex_v, color))

        if opt.not_greedy:
            solution = set()
        else:
            solution = greedy_initial_solution(edge_list, deggre_counter)

        if opt.print:
            print(solution)

        solution = simulated_annealing(solution, edge_list, opt.metropolis_it,
                                       opt.init_temp, opt.end_temp, opt.discount, opt.echo)
        print(f"Solution: {len(solution)}")
        print()
        test_solution(solution)

        if opt.print:
            print(solution)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instance file (mandatory)")
    parse.add_argument("-t", "--initial-temperature", action="store", dest="init_temp",
                        type=int, default=1000, help="Initial temperature (default = 1000)")
    parse.add_argument("-e", "--end-temperature", action="store", dest="end_temp",
                        type=float, default=0.001, help="Ending temperature (default = 0.001)")
    parse.add_argument("-d", "--discount", action="store", dest="discount",
                        type=float, default=0.95, help="Discount value for temperature (default = 0.95)")
    parse.add_argument("-i", "--metropolis-iterations", action="store", dest="metropolis_it",
                        type=int, default=100, help="Number of metropolist iterations (default = 100)")
    parse.add_argument("-p", "--print-solution", action="store_true", dest="print", default=False,
                       help="Flag to indicate if the solution (edges set) should be printed")
    parse.add_argument("--echo-steps", action="store_true", dest="echo", default=False,
                       help="Flag to indicate if the solution steps should be printed")
    parse.add_argument("--wo-greedy", action="store_true", dest="not_greedy", default=False,
                       help="Flag to indicate if the initial solution is not the greedy one")

    rd.seed(datetime.now())
    main(parse)
