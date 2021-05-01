import numpy as np
from numba import njit
import math

@njit
def bfs(Gdeg,edgeListG):
    #simple breadth first search algorithm, from each vertex
    N = Gdeg.size
    
    distMat1 = np.zeros((N,N))
    conn = True
    for s in range(N):
        visited = np.zeros(N,dtype=np.int8)
     
        # Create a queue for BFS. Queues are not suported with njit yet so do it manually
        myQueue = np.zeros(N,dtype=np.int8)
        dist = np.zeros(N,dtype=np.int8)
        startInd = 0
        endInd = 0

        # Mark the source node as visited and enqueue it 
        myQueue[endInd] = s
        endInd += 1
        visited[s] = 1

        while endInd > startInd:
            pivot = myQueue[startInd]
            startInd += 1
            
            for i in range(Gdeg[pivot]):
                if visited[edgeListG[pivot][i]] == 0:
                    myQueue[endInd] = edgeListG[pivot][i]
                    dist[edgeListG[pivot][i]] = dist[pivot] + 1
                    endInd += 1
                    visited[edgeListG[pivot][i]] = 1
        if endInd < N:
            conn = False #not connected
        
        for i in range(N):
            distMat1[s][i] = dist[i]
        
    return distMat1, conn


@njit
def score_graph(adjMatG, edgeListG, Gdeg):
    """
    Reward function for Conjecture 2.3, using numba
    """
    N = Gdeg.size
    INF = 100000
            
    distMat, conn = bfs(Gdeg,edgeListG)
    #G has to be connected
    if not conn:
        return -INF
        
    diam = np.amax(distMat)
    sumLengths = np.zeros(N,dtype=np.int8)
    sumLengths = np.sum(distMat,axis=0)        
    evals =  np.linalg.eigvalsh(distMat)
    evals = -np.sort(-evals)
    proximity = np.amin(sumLengths)/(N-1.0)

    ans = -(proximity + evals[math.floor(2*diam/3) - 1])

    return ans

