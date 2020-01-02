# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:19:22 2019
@author: Samuel
"""
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import lil_matrix
from label_propagation import LGC, HMN, CAMLP, OMNI, PARW
import gera_grafokNN as gg
import warnings
import time 
import sys
#Leitura do dataset
def openData(arch):
    #This function readas paths and returns dataFrames
    df = pd.read_csv(arch,sep=',',header=0)
    return df.dropna()
   
def computeCentrality(G):
    #This function computes the centrality measures for the graphs
    
    #Degree
    degrees = [val for (node, val) in G.degree()]
    print('Degree Values: {}.'.format(np.mean(degrees)))
    
    #Betweenness
    bts = [val for (node, val) in nx.betweenness_centrality(G).items()]
    print('Betweenness Values: {}.'.format(np.mean(bts)))
    
    #Closeness
    cls =  [val for (node, val) in nx.closeness_centrality(G).items()]
    print('Closeness Values: {}.'.format(np.mean(cls)))
    
    #Eigenvector
    eig = [val for (node, val) in nx.eigenvector_centrality(G).items()]
    print('Eigenvector Values: {}.'.format(np.mean(eig)))
    
    #Pagerank
    pgr = [val for (node, val) in nx.pagerank(G).items()]
    print('PageRank Values: {}.'.format(np.mean(pgr)))
    
    #Clustering
    clt = [val for (node, val) in nx.clustering(G).items()]
    print('Clustering Values: {}.'.format(np.mean(clt)))
    
    #Structural holes
    sho = [val for (node, val) in nx.effective_size(G).items()]
    print('Structural Holes Values: {}.'.format(np.mean(sho)))
    
    print('-----------------------------------------------------------------------')    


def main():
    #Time variable
    t1 = time.time()
    
    #Output array 
    output = []
    
    warnings.filterwarnings("ignore")
    
    #INPUTS
    trainData = sys.argv[1]
    testData = sys.argv[2]
    model = sys.argv[3]
    print(model)
    mode = sys.argv[4]
    neighborsCode = sys.argv[5]
    distancesCode = sys.argv[6]
    
    if neighborsCode == 'low':
        neighbors = [i for i in range(1,11)]
    elif neighborsCode == 'high':
        neighbors = [15,20,25,50,75,100]
    
    
    if distancesCode == 'ch':
        distances = ['chebyshev']
    elif distancesCode == 'all':
        distances = ['chebyshev','cosine','euclidean','manhattan','jaccard']     
        
    train = openData(trainData)   
    train = openData(testData)   
    
    #Build the graph
    
    #Run the models
    
    #Time variable
    t2 = time.time()
    
    print('Time to run test tree: {} seconds.'.format(t2-t1))
    
    
    
if __name__=='__main__':
    main()
