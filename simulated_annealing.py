from __future__ import annotations

import argparse
import errno
import logging as lg
import os
import random as rd
import sys
from datetime import datetime
from math import exp
from multiprocessing import Pool
from typing import Dict, List, Set, Union

import pandas as pd


class Edge:
    """Class that holds information regarding an edge of the graph.
        The edge has the following attributes:

        vertex_u (int): one of its vertices.
        vertex_v (int): other of its vertices.
        color (int): color type of the edge.
        degree (int): average degree of its vertices.
    """
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        """
            Initializes an Edge object.

            It takes two vertices and one color. Furthermore, the degree of each edge is initialized to 0.
        """
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color
        self.deggre = 0

    def has_same_attribute(self, edge: Edge) -> bool:
        """
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
        """
        vertices = [self.vertex_u, self.vertex_v]
        return edge.color == self.color or edge.vertex_u in vertices or edge.vertex_v in vertices

    def __repr__(self) -> str:
        return f"({self.vertex_u},{self.vertex_v})[{self.color}][Gm(e) = {self.deggre}]"

    def __str__(self) -> str:
        return f"Edge({self.vertex_u},{self.vertex_v})[{self.color}]"

def get_neighbor(sol: Set[Edge], edge_list: List[Edge]) -> Set[Edge]:
    """Function that returns a neighboring solution to the solution given.

    Args:
        sol (Set[Edge]): Current solution.
        edge_list (List[Edge]): LIst containing all edges in the graph.

    Returns:
        Set[Edge]: Neighboring solution.
    """
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

def metropolis(sol: Set[Edge], temp: float, itr: int, best: Set[Edge],
               edge_list: List[Edge]) -> Union[Set[Edge], Set[Edge]]:
    """Function that performs the Metropolis algorithm.

    Args:
        sol (Set[Edge]): Current solution to the problem.
        temp (float): Current temperature of the Simulated Annealing algorithm.
        itr (int): Number of iterations to run Metropolis.
        best (Set[Edge]): Current best solution.
        edge_list (List[Edge]): List of all edges in the graph.

    Returns:
        Union[Set[Edge], Set[Edge]]: Tuple containing current solution and current best solution respectively.
    """
    for _ in range(itr):
        new_sol: Set[Edge] = get_neighbor(sol.copy(), edge_list)
        delta = len(sol) - len(new_sol) # since we want to maximize, invert difference calculation
        try:
            prob = exp(- delta / temp)
        except OverflowError:
            prob = float('inf')
        if rd.random() < min(prob, 1.0):
            sol = new_sol

        if len(best) < len(sol):
            best = sol.copy()

    return sol, best

def simulated_annealing(sol: Set[Edge], edge_list: List[Edge], metropolis_it: int,
                        init_temp: float, end_temp: float, discount: float, echo: bool = False) -> Set[Edge]:
    """Function that performs the main Simulated Annealing algorithm.

    Args:
        sol (Set[Edge]): Initial solution to the problem.
        edge_list (List[Edge]): List containing all edges in the graph.
        metropolis_it (int): Number of iterations to run Metropolis algorithm with constant temperature.
        init_temp (float): Initial temperature value.
        end_temp (float): Ending temperature value.
        discount (float): Discount value to reduce tempereture each Metropolis run.
        echo (bool, optional): Flag indicating to show each step of the run. Defaults to False.

    Returns:
        Set[Edge]: Best solution found by the algorithm.
    """
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
    """Function that receives a solution to the problem and checks if the solution is correct.

    Args:
        sol (Set[Edge]): A solution to the problem.
    """
    test_sol = sol.copy()
    print("Testing solution correctness.")
    for edge in sol:
        test_sol.remove(edge)
        if any(edge.has_same_attribute(e) for e in test_sol):
            print("Something went wrong")
            print(f"{edge = } has at least on attribute in common with another edge.")
        else:
            print("Solution correct.")

def greedy_initial_solution(edges: List[Edge], deggre_counter: Dict[int, int]) -> Set[Edge]:
    """
        Creates an initial solution to the diversified matching problem based on a greedy approach.

        The function performs the following steps:

        1) For each edge, it calculates the average degree of the vertices touching it and stores that value within the
        Edge structure.
        2) Sorts the set of edges, in ascending order, based on previously accumulated degree.
        3) Creates a set of edges for the solution, which is initialized with the first edge of the ordered set.
        4) Iterates over the rest of the set and adds all edges that do not share attributes with any of the edges that
        already make up the solution.
    """

    # For each edge, it calculates the average degree of the vertices touching it.
    for edge in edges:
        edge.deggre = (deggre_counter[edge.vertex_u] + deggre_counter[edge.vertex_v]) / 2

    # Order the edges, to greedily search for the edge with the lowest average degree
    sorted_edges = sorted(edges, key=lambda x: x.deggre, reverse=False)

    # Initializes the set of edges by placing the first edge of the ordered list, it means that the edge of the smallest
    # degree is the first element of the initial solution.
    edges = set()
    if len(sorted_edges) > 0:
        edges.add(sorted_edges.pop(0))

        for edge in set(sorted_edges):
            if not any(edge.has_same_attribute(e) for e in edges):
                edges.add(edge)

    return edges

def mkdirs() -> None:
    """Function that creates necessary directories that hold data on multiple runs.

    Raises:
        OSError: The funcion raises an OSError if the directories can't be created.
    """
    try:
        os.mkdir("log")
        os.mkdir("results")
    except OSError as error:
        if error.errno != errno.EEXIST:
            print(f"Something wrong with directory creations, error message: {error.strerror}")
            raise OSError(error).with_traceback(error.__traceback__)

def run_sim(opt: argparse.Namespace, edge_list: Set[Edge],
            deggre_counter: Dict[int, int], itr: int = -1) -> Union[int, int]:
    """Function that runs Simulated Annealing function for a given instance of the problem.

    Args:
        opt (argparse.Namespace): Variable that holds all the parameters passed to the script.
        edge_list (Set[Edge]): List containing all edges in the graph.
        deggre_counter (Dict[int, int]): Dict that has the degree of all vertices in the graph.
        itr (int, optional): Iteration of this run (necessary to logging multiple runs). Defaults to -1.

    Returns:
        Union[int, int]: The size of the initial solution (given by the greedy method) and the size of the solution
        found by the Simulated Annealing method, respectively.
    """
    logging = (itr != -1)
    if logging: lg.info("Iteration %s started.", itr)

    if opt.not_greedy:
        init_sol = set()
    else:
        init_sol = greedy_initial_solution(edge_list, deggre_counter)
        if logging: lg.info("Iteration %s finished greedy.", itr)

    if opt.debug: print(init_sol)
    if logging: lg.info("Starting SA on iteration %s.", itr)

    solution = simulated_annealing(init_sol, edge_list, opt.metropolis_it,
                                   opt.init_temp, opt.end_temp, opt.discount, opt.echo)

    if opt.debug:
        test_solution(solution)
        print(solution)

    if logging: lg.info("Iteration %s finished.", itr)

    return len(init_sol), len(solution)

def solve_instance(opt: argparse.Namespace, path: str) -> List[Union[int, int]]:
    """Method that performs one or multiple runs of the Simulated Annealing of a given instance.

    Args:
        opt (argparse.Namespace): Object containing all parameters passed as arguments to the script.
        path (str): Path to the instace to solve.

    Returns:
        List[Union[int, int]]: List containing all results from the multiple runs of the algorithm for the given
        instance.
    """
    print(f"Solving problem for instance {path}.")
    solutions = list()

    with open(path, 'r') as file:
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

        if opt.n_runs > 1:
            def append_result(result):
                solutions.append(result)

            mkdirs()
            if opt.parallel:
                with Pool(processes=os.cpu_count() - 2) as pool:
                    _ = [pool.apply_async(run_sim,
                                          args=(opt, edge_list, deggre_counter, itr),
                                          callback=append_result) for itr in range(opt.n_runs)]
                    pool.close()
                    pool.join()
            else:
                solutions = [run_sim(opt, edge_list, deggre_counter, itr) for itr in range(opt.n_runs)]

        else:
            solutions.append(run_sim(opt, edge_list, deggre_counter, 1))
            print(f"Greedy solution: {solutions[0][0]}", end=" ---- ")
            print(f"Simulated Annealing solution: {solutions[0][1]}")

    print("Done.")
    return solutions

def main(parser: argparse.ArgumentParser) -> None:
    """Main function of the script.

    Args:
        parser (argparse.ArgumentParser): Argument parser regarding the arguments passed to the script.
    """
    opt = parser.parse_args()
    if not opt.filepath:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()


    start_date = datetime.now().strftime("%d%m%y_%H%M")
    def create_log(filename: str) -> None:
        lg.basicConfig(format='%(asctime)s: %(message)s',
                            datefmt="%d-%m-%Y %H:%M:%S" ,
                            filename=f"log/{filename}_{start_date}.log",
                            level=lg.INFO,
                            filemode = 'w')

    def save_to_csv(filename: str, solutions: List, opt_cols: List[str] = None) -> None:
        filepath = f"results/{filename}_{start_date}.csv"
        if opt_cols is None:
            pd.DataFrame(solutions).to_csv(filepath, index=False)
        else:
            pd.DataFrame(solutions, columns=opt_cols).to_csv(filepath, index=False)

    try:
        instances = os.listdir(opt.filepath)
        create_log("batch")
        solutions = dict()
        for inst in instances:
            solutions[inst] = solve_instance(opt, f"{opt.filepath}/{inst}")
        save_to_csv("batch", solutions)
    except NotADirectoryError:
        filename = f"{opt.filepath[-4:]}"
        if opt.n_runs > 1: create_log(filename)
        solutions = solve_instance(opt, opt.filepath)
        if opt.n_runs > 1: save_to_csv(filename, solutions, ["Greedy", "SA"])

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instances file (mandatory) (can be a file or a directory of files)")
    parse.add_argument("-t", "--initial-temperature", action="store", dest="init_temp",
                        type=float, default=2, help="Initial temperature (default = 2)")
    parse.add_argument("-e", "--end-temperature", action="store", dest="end_temp",
                        type=float, default=0.01, help="Ending temperature (default = 0.01)")
    parse.add_argument("-d", "--discount", action="store", dest="discount",
                        type=float, default=0.9, help="Discount value for temperature (default = 0.9)")
    parse.add_argument("-i", "--metropolis-iterations", action="store", dest="metropolis_it",
                        type=int, default=500, help="Number of metropolist iterations (default = 500)")
    parse.add_argument("--debug", action="store_true", dest="debug", default=False,
                       help="Flag to indicate if debug information should be printed to screen.")
    parse.add_argument("--echo-steps", action="store_true", dest="echo", default=False,
                       help="Flag to indicate if the solution steps should be printed")
    parse.add_argument("--wo-greedy", action="store_true", dest="not_greedy", default=False,
                       help="Flag to indicate if the initial solution is not the greedy one")
    parse.add_argument("-n", "--number-of-runs", action="store", type=int, dest="n_runs", default=1,
                       help="Number of multiple simulation runs (default = 1)")
    parse.add_argument("--parallel", action="store_true", dest="parallel", default=False,
                       help="Set the script to run simulations in parallel using number of available CPU")

    rd.seed(datetime.timestamp(datetime.now()))
    main(parse)
