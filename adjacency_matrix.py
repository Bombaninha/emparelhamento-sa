class Graph:

    def __init__(self, vertexes):
        self.vertexes = vertexes
        self.graph = [[0] * self.vertexes for i in range(self.vertexes)]

    def add_edge(self, u, v):
        self.graph[u - 1][v - 1] = 1 # += 1
        self.graph[v - 1][u - 1] = 1

    def show_matrix(self):
        for i in range(self.vertexes):
            print(self.graph[i])

'''
Exemplo de uso:

g = Graph(4)

g.add_edge(1, 2)
g.add_edge(3, 4)
g.add_edge(2, 3)

g.show_matrix()
'''