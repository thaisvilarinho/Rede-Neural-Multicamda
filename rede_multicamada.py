# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(soma):
    return 1/(1+np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = 2*np.random.random((2,3))-1
pesos1 = 2*np.random.random((3,1))-1

epocas = 1000000
taxaAprendizagem = 0.5
momento = 1

for j in range(epocas):

    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    deltaSaidaxPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaxPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)








