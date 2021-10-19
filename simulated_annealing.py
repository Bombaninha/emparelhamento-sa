import argparse
import sys

import adjacency_list
import adjacency_matrix

class Edge:
    def __init__(self, vertex_u: int, vertex_v: int, color: int) -> None:
        self.vertex_u = vertex_u
        self.vertex_v = vertex_v
        self.color = color

    def __str__(self) -> str:
        return f"u: {self.vertex_u}, v: {self.vertex_v}, color: {self.color}"

def simulated_annealing(graph, edge_list) -> None:
    pass

if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog='Python code to solve the instances of the diversified matching problem.')

    parse.add_argument("-f", "--file-path", action="store", dest="filepath",
                       help="Define the path to the instance file (mandatory)")

    options = parse.parse_args()
    if not options.filepath:
        print('Wrong usage of script!')
        print()
        parse.print_help()
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

        print(n_vertices)
