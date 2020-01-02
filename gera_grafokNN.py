# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:13:49 2019

@author: Samuel
"""
import numpy as np
from sklearn.neighbors import kneighbors_graph
import pandas as pd 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy.spatial import distance as dt
import warnings

def buildGraph(X,neigh,distance):
    #####################################################
    # Graph construction
    A = kneighbors_graph(X, n_neighbors = neigh, include_self = False,metric=distance, mode='connectivity')
    A = A.toarray()
    conteudo = []
    result = np.where(A == 1)
    if len(result) > 0:
        listOfCoordinates= list(zip(result[0], result[1]))
    else:
        result = np.where(A==0)
        listOfCoordinates= list(zip(result[0], result[1]))
        
    for cord in listOfCoordinates:
        conteudo.append(cord)
    return conteudo
    

def main(nome_dataset,neigh,distance,mode):
    warnings.filterwarnings("ignore")
    
    adjList = buildGraph(X,neigh,distance)
    
    return adjList
    
