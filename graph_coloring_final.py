import random
from random import randint
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import networkx as nx


#########################################
# DEFINICOES ESTÁTICAS
#########################################

NR_VERTICES_GRAFO = 8
TAMANHO_POPULACAO = 50

####################################################################
# FUNÇÕES UTILIZADAS PARA O AG E PARA A COLORAÇÃO MIN
####################################################################


def criaCromossomo(maximoNumCores):

    return np.random.randint(1, maximoNumCores + 1, size=(NR_VERTICES_GRAFO))


def criaPopulacao(maximoNumCores):
    return np.array([criaCromossomo(maximoNumCores) for i in range(TAMANHO_POPULACAO)])


velocidades = []
qtdPSO = 0


def atualizarVelocidade():
    global velocidades
    velocidades.clear()
    numeroTrocas = 3
    for i in range(numeroTrocas):
        r1 = random.randint(0, NR_VERTICES_GRAFO-1)
        r2 = random.randint(0, NR_VERTICES_GRAFO-1)
        velocidades.append((r1, r2))


def atualizarPosicao(grafo, cromossomo):
    global velocidades, qtdPSO
    for p1, p2 in velocidades:

        fitnessAnterior = calculaFitness(grafo, cromossomo)

        novoCromossomo = copy.deepcopy(cromossomo)
        novoCromossomo[p1], novoCromossomo[p2] = novoCromossomo[p2], novoCromossomo[p1]
        novoFitness = calculaFitness(grafo, novoCromossomo)

        if novoFitness < fitnessAnterior:
            qtdPSO = qtdPSO + 1
            return novoCromossomo
        else:
            return cromossomo


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
    if chance <= probabilidade:
        for vertice1 in range(NR_VERTICES_GRAFO):
            for vertice2 in range(vertice1, NR_VERTICES_GRAFO):
                if grafo[vertice1][vertice2] == 1 and cromossomo[vertice1] == cromossomo[vertice2]:
                    cromossomo[vertice1] = randint(1, maximoNumCores)
    return cromossomo


def criaGrafo():
    grafo = np.random.randint(0, 2, size=(
        NR_VERTICES_GRAFO, NR_VERTICES_GRAFO))
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
    print(f'O grafo é {maximoNumCores}-coloravel')
    return


def cenario_count_0(maximoNumCores):
    return


def cenario_maximo_num_maior_1(maximoNumCores):
    print(f'O grafo é {maximoNumCores}-coloravel')


def outros_cenarios(maximoNumCores):
    print(f'O grafo é {maximoNumCores}-coloravel')


def printa_solucao_correta_da_coloracao(testaColoracao):
    print('Solucão correta da coloração seria :', testaColoracao())


#####################################################################################################
# ALGORITMO NA FUNÇÃO MAIN
#####################################################################################################

if __name__ == '__main__':

    start_time = time.time()

    grafo = criaGrafo()

    G = nx.Graph()

    for i in range(len(grafo)):
        G.add_node(i)

    for i in range(len(grafo)):
        for j in range(len(grafo[i])):
            if grafo[i][j] == 1:
                G.add_edge(i, j)

    verifica_casos_base(NR_VERTICES_GRAFO, grafo)

    maximoNumCores = maximoCores(grafo)

    count = 0
    countCoresFalhas = 0

    print(f'Tentativa de colorir o grafo com: {maximoNumCores} cores')

    for _ in iter(int, 1):
        populacao = criaPopulacao(maximoNumCores)

        melhorFitness = calculaFitness(grafo, populacao[0])
        fittest = populacao[0]

        geracao = 0
        numGeracoes = 50

        if NR_VERTICES_GRAFO > 5:
            numGeracoes = NR_VERTICES_GRAFO * 15

            while melhorFitness != 0 and geracao != numGeracoes:
                geracao += 1

                # print(f'Geração: {geracao}')

                populacao = selecaoPorTorneio(populacao)

                if ehPar(populacao):
                    populacao.pop()

                populacao_filha = []
                random.shuffle(populacao)
                for i in range(0, len(populacao) - 1, 2):
                    gera_filhos_por_crossover(
                        crossover, populacao, populacao_filha, i)

                for cromossomo in populacao_filha:
                    verifica_geracao_e_gera_mutacao(
                        mutacao, geracao, cromossomo)

                for i in range(len(populacao), TAMANHO_POPULACAO):
                    populacao.append(criaCromossomo(maximoNumCores))

                for cromossomo in populacao_filha:
                    atualizarVelocidade()
                    atualizarPosicao(grafo, cromossomo)

                populacao = populacao_filha
                melhorFitness = calculaFitness(grafo, populacao[0])
                fittest = populacao[0]
                for individuo in populacao:
                    if (calculaFitness(grafo, individuo) < melhorFitness):
                        melhorFitness = calculaFitness(grafo, individuo)
                        fittest = individuo
                        # print(f"Melhor fitness: {melhorFitness}")
                        # print(f"maximoNumCores: {maximoNumCores}")

                if melhorFitness == 0:
                    # print("Encontrou")
                    fittest_final = fittest
                    maximoNumCores_final = maximoNumCores
                    break

            # print(f"Melhor fitness: {melhorFitness}")
            if melhorFitness == 0:
                cenario_melhor_fitness_0(maximoNumCores)
                maximoNumCores -= 1
                count = 0
            else:
                if count != 2 and maximoNumCores > 1:
                    countCoresFalhas = maximoNumCores

                    if count == 0:
                        print(
                            f'{maximoNumCores} falhou. Testando denovo com {maximoNumCores} cores')
                        cenario_count_0(maximoNumCores)

                    elif count == 1:
                        print(
                            f'{maximoNumCores} falhou. Testando denovo com {maximoNumCores - 1} cores')
                        maximoNumCores -= 1
                    else:
                        print(
                            f'{maximoNumCores} falhou. Testando denovo com {maximoNumCores} cores')

                    count += 1
                    continue
                if maximoNumCores > 1:
                    cenario_maximo_num_maior_1(maximoNumCores_final)
                else:
                    outros_cenarios(maximoNumCores_final)
                printa_solucao_correta_da_coloracao(testaColoracao)

                print(f"Quantidade PSO: {qtdPSO}")

                tempoTotal = time.time() - start_time

                print(f"Tempo total: {tempoTotal}")

                print(f"Melhor indivíduo: {fittest_final}")

                for i, cor in enumerate(fittest_final):
                    print(f"Coloração: vértice {i} - {cor}")
                    G.nodes[i]['color'] = cor

                pos = nx.spring_layout(G)
                nx.draw(G, pos, node_color=[G.nodes[i]['color']
                        for i in G.nodes()], with_labels=True)
                plt.show()

                break
