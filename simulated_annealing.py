from __future__ import annotations
import argparse
import sys
from math import exp
from datetime import datetime
from random import seed, random, choice
from typing import List, Set, Union, Dict
import os

class Edge:
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        '''
            Initializes an Edge object.

            It takes two vertexes and one color. Furthermore, the degree of each edge is initialized to 0.
        '''
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color
        self.deggre = 0

    def has_same_attribute(self, edge: Edge) -> bool:
        '''
            Compares two edges and sees if they share certain attributes.
            Two edges share attributes if they both have the same color or at least one vertex in common.

            Suppose the following instances:

            e1: Edge of color 1 connecting vertices 1 and 4
            e2: Edge of color 2 connecting vertices 1 and 5
            e3: Edge of color 1 connecting vertices 2 and 8
            e4: Edge of color 3 connecting vertices 2 and 4

            Examples:

            1) e1.has_same_attribute(e1) -> True, because they are the same edge
            2) e1.has_same_attribute(e2) -> True, because they share the vertex 1
            3) e1.has_same_attribute(e3) -> True, because they share the color 1
            4) e1.has_same_attribute(e4) -> True, because they share the vertex 4
        '''
        vertices = [self.vertex_u, self.vertex_v]
        return edge.color == self.color or edge.vertex_u in vertices or edge.vertex_v in vertices

    def __repr__(self) -> str:
        return f"({self.vertex_u},{self.vertex_v})[{self.color}][Gm(e) = {self.deggre}]"

    def __str__(self) -> str:
        return f"Edge({self.vertex_u},{self.vertex_v})[{self.color}]"

'''
    def __eq__(self, edge: Edge) -> bool:
       
            Compares two edges and sees if they are equal.
            Two edges are equal if they both have the same color and two vertexes in common.

            Suppose the following instances:

            e1: Edge of color 1 connecting vertices 1 and 4
            e2: Edge of color 2 connecting vertices 1 and 5
            e3: Edge of color 1 connecting vertices 2 and 8
            e4: Edge of color 1 connecting vertices 4 and 1

            Examples:

            1) e1 == e1 -> True, because they are the same edge
            2) e1 == e2 -> False, because they are not the same edge
            3) e1 == e3 -> False, because they are not the same edge
            4) e1 == e4 -> True, because they are the same edge, in another order
        
        vertices = [self.vertex_u, self.vertex_v]
        return edge.color == self.color and edge.vertex_u in vertices and edge.vertex_v in vertices
'''
def get_neighbor(sol: Set[Edge], edge_list: List[Edge]) -> Set[Edge]:
    n_edges = len(edge_list)
    for _ in range(n_edges):
        edge = choice(edge_list)
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
        new_sol: Set[Edge] = get_neighbor(sol.copy(), edge_list)
        delta = len(sol) - len(new_sol) # since we want to maximize, invert difference calculation
        try:
            prob = exp(- delta / temp)
        except OverflowError:
            prob = float('inf')
        if random() < min(prob, 1.0):
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
    '''
        Creates an initial solution to the diversified matching problem based on a greedy approach.

        The function performs the following steps:

        1) For each edge, it calculates the average degree of the vertices touching it and stores that value within the Edge structure.
        2) Sorts the set of edges, in ascending order, based on previously accumulated degree.
        3) Creates a set of edges for the solution, which is initialized with the first edge of the ordered set.
        4) Iterates over the rest of the set and adds all edges that do not share attributes with any of the edges that already make up the solution.
    '''

    # For each edge, it calculates the average degree of the vertices touching it.
    for edge in edges:
        edge.deggre = (deggre_counter[edge.vertex_u] + deggre_counter[edge.vertex_v]) / 2

    # Order the edges, to greedily search for the edge with the lowest average degree
    sorted_edges = sorted(edges, key=lambda x: x.deggre, reverse=False)

    # Initializes the set of edges by placing the first edge of the ordered list, it means that the edge of the smallest degree is the first element of the initial solution.
    edges = set()
    if(len(sorted_edges) > 0):
        edges.add(sorted_edges.pop(0))

        # sorted_edges e set(sorted_edges) dão valores diferentes, apesar de serem a mesma qtde de elementos. 2468 p/ 2430
        for edge in set(sorted_edges):
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

        
        #solution = simulated_annealing(solution, edge_list, opt.metropolis_it,
        #                               opt.init_temp, opt.end_temp, opt.discount, opt.echo)
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
                        type=float, default=1000, help="Initial temperature (default = 1000)")
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

    # BUG: Deprecated
    seed(datetime.now())
    main(parse)
