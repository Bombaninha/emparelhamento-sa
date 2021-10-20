class Graph:

    def __init__(self, vertexes: int) -> None:
        self.vertexes = vertexes
        self.graph = [[] for i in range(self.vertexes)]

    def add_edge(self, u: int, v: int) -> None:
        self.graph[u - 1].append(v)
        self.graph[v - 1].append(u)

    def show_list(self) -> None:
        for i in range(self.vertexes):
            print(f'{ i + 1}: ', end='  ')
            text_to_be_printed = ''
            for j in self.graph[i]:
                text_to_be_printed += str(j) + '  ->  '
            text_to_be_printed += 'None'
            print(text_to_be_printed)

    def get_deggre(self, vertex: int) -> int:
        return len(self.graph[vertex - 1])

    def has_edge_between(self, vertex_u: int, vertex_v: int) -> bool:
        return (vertex_v) in self.graph[vertex_u - 1]

    def get_edge_deggre(self, vertex_u: int, vertex_v: int) -> float or bool:
        if(self.has_edge_between(vertex_u, vertex_v)):
            return (self.get_deggre(vertex_u) + self.get_deggre(vertex_v)) / 2
        else:
            return False

'''
#Exemplo de uso:

g = Graph(4)

g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 3)

#print(g.get_deggre(1))
#print(g.has_edge_between(1, 1))
#print(g.has_edge_between(1, 2))
#print(g.get_edge_deggre(1, 1))
g.show_list()
'''