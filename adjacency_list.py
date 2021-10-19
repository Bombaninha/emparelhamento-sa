class Graph:

    def __init__(self, vertexes):
        self.vertexes = vertexes
        self.graph = [[] for i in range(self.vertexes)]

    def add_edge(self, u, v):
        self.graph[u - 1].append(v)
        self.graph[v - 1].append(u)

    def show_list(self):
        for i in range(self.vertexes):
            print(f'{ i + 1}: ', end='  ')
            text_to_be_printed = ''
            for j in self.graph[i]:
                text_to_be_printed += str(j) + '  ->  '
            text_to_be_printed += 'None'
            print(text_to_be_printed)

'''
Exemplo de uso:

g = Graph(4)

g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 3)

g.show_list()
'''