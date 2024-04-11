import random
from random import randint
import numpy as np
import time


#########################################
#DEFINICOES ESTÁTICAS
#########################################

NR_VERTICES_GRAFO = 30
TAMANHO_POPULACAO = 40


####################################################################
#FUNÇÕES UTILIZADAS PARA O AG E PARA A COLORAÇÃO MIN
####################################################################


def criaCromossomo(maximoNumCores):
    return np.random.randint(1, maximoNumCores + 1, size=(NR_VERTICES_GRAFO))

def criaPopulacao(maximoNumCores):
    print(maximoNumCores)
    return np.array([criaCromossomo(maximoNumCores) for i in range(TAMANHO_POPULACAO)])


def welch_powell(grafo):
    # Passo 1: Ordenar os vértices por grau (do maior para o menor)
    graus = [(v, sum(grafo[v])) for v in range(len(grafo))]
    graus_ordenados = sorted(graus, key=lambda x: x[1], reverse=True)

    # Inicializar o dicionário de cores com todos os vértices do grafo
    cores = {v: None for v in range(len(grafo))}

    # Passo 2: Colorir os vértices
    for vertice, _ in graus_ordenados:
        # Conjunto de cores dos vizinhos
        cores_vizinhos = {cores[vizinho] for vizinho, adjacente in enumerate(grafo[vertice]) if
                          adjacente and cores[vizinho] is not None}

        # Encontrar a menor cor disponível
        cor = 1
        while cor in cores_vizinhos:
            cor += 1

        # Atribuir a cor ao vértice
        cores[vertice] = cor

    # Retornar as cores atribuídas aos vértices
    return cores


def aplicaHeuristicaGulosa(populacao, grafo, maximoNumCores):
    novaPopulacao = []
    for cromossomo in populacao:
        # Aplica a heurística de coloração gulosa a cada cromossomo da população
        cromossomo_guloso = coloracao_gulosa(grafo, maximoNumCores)
        novaPopulacao.append(cromossomo_guloso)
    return np.array(novaPopulacao)

def coloracao_gulosa(grafo, maximoNumCores):
    num_vertices = len(grafo)
    cores = [0] * num_vertices  # Inicializa vetor de cores com 0

    for vertice in range(num_vertices):
        cores_disponiveis = list(range(1, maximoNumCores + 1))  # Lista de cores disponíveis para o vértice atual
        random.shuffle(cores_disponiveis)  # Embaralhar a ordem das cores disponíveis

        for vizinho in range(num_vertices):
            if grafo[vertice][vizinho] and cores[vizinho] in cores_disponiveis:
                cores_disponiveis.remove(cores[vizinho])  # Remove cores já usadas pelos vizinhos

        if not cores_disponiveis:  # Se não houver cores disponíveis, adiciona uma nova cor
            cores_disponiveis.append(max(cores) + 1)

        # Atribui a menor cor disponível ao vértice atual
        cores[vertice] = cores_disponiveis[0]

    return cores
#
#
#
# def criaPopulacaoComHeuristica(grafo, maximoNumCores):
#     populacao = []
#     for i in range(TAMANHO_POPULACAO // 2):  # Metade da população gerada aleatoriamente
#         cromossomo = criaCromossomo(maximoNumCores)
#         print(cromossomo)
#         populacao.append(cromossomo)
#     for _ in range(TAMANHO_POPULACAO // 2):
#         # Embaralhar a ordem dos vértices antes de aplicar a heurística de coloração gulosa
#         vertices_embaralhados = list(range(NR_VERTICES_GRAFO))
#         random.shuffle(vertices_embaralhados)
#
#         grafo_embaralhado = np.zeros((NR_VERTICES_GRAFO, NR_VERTICES_GRAFO))
#         for i in vertices_embaralhados:
#             for j in vertices_embaralhados:
#                 grafo_embaralhado[i][j] = grafo[i][j]
#
#         cromossomo = coloracao_gulosa(grafo_embaralhado, maximoNumCores)
#         populacao.append(cromossomo)
#         print(cromossomo)
#     return np.array(populacao)


# def criaPopulacaoComHeuristica(maximoNumCores):
#     print(maximoNumCores)
#     populacao = []
#     print("Populacao aleatoria")
#     for i in range(TAMANHO_POPULACAO // 2): # Metade da população gerada aleatoriamente
#         cromossomo = criaCromossomo(maximoNumCores)
#         print(cromossomo)
#         populacao.append(cromossomo)
#     print("Populacao heuristica")
#     # Metade da população gerada pela heurística gulosa
#     for j in range(TAMANHO_POPULACAO // 2):
#         cromossomo = heuristica_gulosa_coloracao()
#         print(cromossomo)
#         populacao.append(cromossomo)
#
#     # Convertendo a lista de cromossomos para um array NumPy
#     populacao = np.array(populacao)
#
#     return populacao
#
# def heuristica_gulosa_coloracao():
#     n = len(grafo)
#     grau = np.sum(grafo, axis=1)  # Calcular o grau de cada vértice
#     ordem_vertices = np.argsort(grau)  # Ordenar os vértices com base no grau
#     cores = np.full(n, -1, dtype=int)  # -1 indica que o vértice ainda não foi colorido
#     disponivel = np.full(n, False, dtype=bool)
#
#     cores[ordem_vertices[0]] = 0  # Colorir o primeiro vértice com a cor 0
#     disponivel[0] = True  # Marcar a cor 0 como disponível
#
#     for u in ordem_vertices[1:]:
#         disponivel.fill(True)  # Resetar informações de cores disponíveis
#
#         # Verificar cores dos vizinhos e marcar suas cores como não disponíveis
#         for i in range(n):
#             if grafo[u][i] == 1 and cores[i] != -1:
#                 disponivel[cores[i]] = False
#
#         # Encontrar a primeira cor disponível
#         cor = 0
#         while cor < n and not disponivel[cor]:
#             cor += 1
#
#         cores[u] = cor  # Atribuir a primeira cor disponível ao vértice
#         disponivel[cor] = True  # Marcar a cor atribuída como disponível
#
#     return cores
#


def calculaFitness(grafo, cromossomo):
    custo = 0
    for vertice1 in range(NR_VERTICES_GRAFO):
        for vertice2 in range(vertice1, NR_VERTICES_GRAFO):
            if grafo[vertice1][vertice2] == 1 and cromossomo[vertice1] == cromossomo[vertice2]:
                custo += 1
    return custo


def selecaoPorTorneio(populacao):
    novaPopulacao = []
    for _ in range(2):
        random.shuffle(populacao)
        for i in range(0, TAMANHO_POPULACAO - 1, 2):
            if calculaFitness(grafo, populacao[i]) < calculaFitness(grafo, populacao[i + 1]):
                novaPopulacao.append(populacao[i])
            else:
                novaPopulacao.append(populacao[i + 1])
    return novaPopulacao


def crossover(pai1, pai2):
    splitPoint = randint(2, NR_VERTICES_GRAFO - 2)
    filho1 = np.concatenate((pai1[:splitPoint], pai2[splitPoint:]))
    filho2 = np.concatenate((pai2[:splitPoint], pai1[splitPoint:]))
    return filho1, filho2


def mutacao(cromossomo, chance):
    probabilidade = random.uniform(0, 1)
    vertices_conflitantes = []
    if chance <= probabilidade:
        for vertice1 in range(NR_VERTICES_GRAFO):
            for vertice2 in range(vertice1, NR_VERTICES_GRAFO):
                if grafo[vertice1][vertice2] == 1 and cromossomo[vertice1] == cromossomo[vertice2]:
                    vertices_conflitantes.append(vertice1)
                    break
        for vertice in vertices_conflitantes:
            cores_possiveis = set(range(1, maximoNumCores + 1))
            for vizinho in range(NR_VERTICES_GRAFO):
                if grafo[vertice][vizinho] == 1 and cromossomo[vizinho] in cores_possiveis:
                    cores_possiveis.remove(cromossomo[vizinho])
            if cores_possiveis:
                cromossomo[vertice] = min(cores_possiveis)
    return cromossomo

# def mutacao(cromossomo, chance):
#     probabilidade = random.uniform(0, 1)
#     if chance <= probabilidade:
#         for vertice1 in range(NR_VERTICES_GRAFO):
#             for vertice2 in range(vertice1, NR_VERTICES_GRAFO):
#                 if grafo[vertice1][vertice2] == 1 and cromossomo[vertice1] == cromossomo[vertice2]:
#                     cromossomo[vertice1] = randint(1, maximoNumCores)
#     return cromossomo


def criaGrafo():
    grafo = np.random.randint(0, 2, size=(NR_VERTICES_GRAFO, NR_VERTICES_GRAFO))
    for i in range(NR_VERTICES_GRAFO):
        for j in range(NR_VERTICES_GRAFO):
            if i == j:
                grafo[i][j] = 0
                continue
            grafo[i][j] = grafo[j][i]

    return grafo

def printGrafo(grafo):
    print([row for row in grafo])


def geraGrafoTeste():
    global grafo
    result = {}

    for i in range(NR_VERTICES_GRAFO):
        result[i] = []
        for j in range(NR_VERTICES_GRAFO):
            if grafo[i][j] == 1:
                result[i].append(j)

    return result

def coloracaoMax(colorDict):
    maxCores = max(colorDict.values())
    return maxCores

def maximoCores(grafo):
    maximoNumCores = 0
    for row in grafo:
        cM = sum(row)
        if cM > maximoNumCores:
            maximoNumCores = cM
    return maximoNumCores


def testaColoracao():
  grafo = geraGrafoTeste()
  vertices = sorted((list(grafo.keys())))
  grafo_coloravel = {}

  for vertex in vertices:
    cor_nao_usada = len(vertices) * [True]

    for vizinho in grafo[vertex]:
      if vizinho in grafo_coloravel:
        colour = grafo_coloravel[vizinho]
        cor_nao_usada[colour] = False
    for colour, unused in enumerate(cor_nao_usada):
        if unused:
            grafo_coloravel[vertex] = colour
            break

  return coloracaoMax(grafo_coloravel)


def verifica_casos_base(n, grafo):
    if n == 1:
        print('Grafo possui coloração 1')
        exit(0)
    elif n == 2:
        if 1 in grafo[0]:
            print('Grafo possui coloração 2')
            exit(0)
        else:
            print('Grafo possui coloração 1')
            exit(0)

def ehPar(populacao):
    return len(populacao) % 2 != 0

def gera_filhos_por_crossover(crossover, populacao, populacao_filha, i):
    filho1, filho2 = crossover(populacao[i], populacao[i + 1])
    populacao_filha.append(filho1)
    populacao_filha.append(filho2)

def verifica_geracao_e_gera_mutacao(mutacao, geracao, cromossomo):
    if geracao < 200:
        cromossomo = mutacao(cromossomo, 0.65)
    elif geracao < 400:
        cromossomo = mutacao(cromossomo, 0.5)
    else:
        cromossomo = mutacao(cromossomo, 0.15)

def cenario_melhor_fitness_0(maximoNumCores):
    return

def cenario_count_0(maximoNumCores):
    return

def cenario_maximo_num_maior_1(countCoresFalhas):
    print(f'O grafo é {countCoresFalhas + 1}-coloravel')

def outros_cenarios(maximoNumCores):
    print(f'O grafo é {maximoNumCores + 1}-coloravel')

def printa_solucao_correta_da_coloracao(testaColoracao):
    print('Solucão correta da coloração seria :', testaColoracao())


#####################################################################################################
#ALGORITMO NA FUNÇÃO MAIN
#####################################################################################################

if __name__ == '__main__':

    timeStart = time.time()

    grafo = criaGrafo()

    verifica_casos_base(NR_VERTICES_GRAFO, grafo)

    maximoNumCores = maximoCores(grafo)

    count = 0
    countCoresFalhas = 0

    print(f'Tentativa de colorir o grafo com: {maximoNumCores} cores')

    for _ in iter(int, 1):
        populacao = criaPopulacao(maximoNumCores)
        #populacao = aplicaHeuristicaGulosa(populacao,grafo,maximoNumCores)
        # for cromossomo in populacao:
        #     cores_refinadas = welch_powell(grafo)
        #     # Atualizar o cromossomo com as cores refinadas
        #     for vertice, cor in cores_refinadas.items():
        #         cromossomo[vertice] = cor
        melhorFitness = calculaFitness(grafo, populacao[0])
        fittest = populacao[0]

        geracao = 0
        numGeracoes = 50

        if NR_VERTICES_GRAFO > 5:
            numGeracoes = NR_VERTICES_GRAFO * 15

            while melhorFitness != 0 and geracao != numGeracoes:
                geracao += 1

                populacao = selecaoPorTorneio(populacao)

                if ehPar(populacao):
                    populacao.pop()

                populacao_filha = []
                random.shuffle(populacao)
                for i in range(0, len(populacao) - 1, 2):
                    gera_filhos_por_crossover(crossover, populacao, populacao_filha, i)

                for cromossomo in populacao_filha:
                    verifica_geracao_e_gera_mutacao(mutacao, geracao, cromossomo)

                for i in range(len(populacao), TAMANHO_POPULACAO):
                    populacao.append(criaCromossomo(maximoNumCores))


                populacao = populacao_filha
                melhorFitness = calculaFitness(grafo, populacao[0])
                fittest = populacao[0]
                for individuo in populacao:
                    if(calculaFitness(grafo, individuo) < melhorFitness):
                        melhorFitness = calculaFitness(grafo, individuo)
                        fittest = individuo
                if melhorFitness == 0:
                    break

            if melhorFitness == 0:
                cenario_melhor_fitness_0(maximoNumCores)
                maximoNumCores -= 1
                count = 0
            else:
                if count != 2 and maximoNumCores > 1:
                    countCoresFalhas = maximoNumCores
                    
                    if count == 0:
                        cenario_count_0(maximoNumCores)
                    if count == 1:
                        maximoNumCores -= 1

                    count += 1
                    continue
                if maximoNumCores > 1:
                    cenario_maximo_num_maior_1(countCoresFalhas)
                else:
                    outros_cenarios(maximoNumCores)
                printa_solucao_correta_da_coloracao(testaColoracao)
                timeEnd = time.time()
                executionTime = timeEnd - timeStart
                print(f'Tempo de execução: {executionTime} segundos')
                break