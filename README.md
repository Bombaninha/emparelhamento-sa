# Emparelhamento Diversificado com Simulated Annealing

Implementação e resultados obtidos no trabalho final da cadeira INF05010 - Otimização Combinatória da Universidade Federal do Rio Grande do Sul, ministrada pelo professor Dr. Marcus Ritt no semestre 2021/1.

## Emparelhamento Diversificado
Dado um grafo não-direcionado G = (V, A), onde cada aresta a ∈ A possui um tipo ta, deseja-se encontrar um emparelhamento M ⊆ A tal que todos m ∈ M possuem tipos diferentes e que contenha o maior número de arestas possíveis. 

## Simulated Annealing
Recozimento simulado é uma meta-heurística para otimização que consiste numa técnica de busca local probabilística, e se fundamenta numa analogia com a termodinâmica. 

### Tarefas

* Formular o problema como programa linear ou inteiro: A formulação do arquivo inteiro está presente no arquivo _glpk_optimize.jl_.
* Resolver as instâncias definidas (abaixo) com um solver genérico: O solver utilizado foi o GLPK e os resultados estão apresentados no relatório.
* Definir e implementar e meta-heurística escolhida para o problema: A implementação da meta-heurística está presente no arquivo _simulated_annealing.py_.
* Resolver as instâncias definidas com a meta-heurística: Os resultados obtidos estão presentes no relatório.
* Documentar e analisar os experimentos: Os resultados e comparações acerca do experimento estão presentes no relatório.
* Apresentar os resultados: A apresentação utilizada está localizada no arquivo _Apresentação.pdf_.
