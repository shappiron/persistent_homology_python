#SCRIPT
import numpy as np
from numba import njit

import itertools
from itertools import combinations
from functools import reduce
from copy import deepcopy

###FUNCTIONS
#function for merging tuples
def tup_comb(el_prev, el):
    return el_prev + el

@njit
def self_distance_matrix(X):
    n = X.shape[0]
    dist_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            d = np.linalg.norm(X[i,:] - X[j,:]) 
            dist_matrix[i,j] = d
            dist_matrix[j,i] = d
    return dist_matrix


def filtered_complex(D, max_dimension):
    unique_distances = np.unique(D)
    n = D.shape[0]
    simplices = [{(s,):0.0 for s in range(n)}]
    for d in range(max_dimension):
        simplices.append({})

    #Create simplicial complexes
    for step in unique_distances:
        for d in range(max_dimension):
            if d == 0:
                for i in range(0, n):
                    for j in range(i+1, n):
                        if step >= D[i,j]:
                            simplices[1][(i,j)] = D[i,j]
            else:
                combos = list(combinations(list(simplices[d].keys()), d+2))
                for c in combos:
                    border_set = np.unique(reduce(tup_comb, c))
                    if (len(border_set) == d+2) and (tuple(border_set) not in simplices[d+1]):
                        simplices[d+1][tuple(border_set)] = step
    #Sorting
    simplices_list = list(itertools.chain.from_iterable([list(d.items()) for d in simplices]))
    sorted_simplices_list = sorted(simplices_list, key = lambda x: (x[1], len(x[0]), x[0]))
    simplices_indices = {s[0]:i for i,s in enumerate(sorted_simplices_list)}
    
    return sorted_simplices_list, simplices_indices


def construct_boundary_matrix(sorted_simplices_list, simplices_indices):
    boundary_matrix_d = len(sorted_simplices_list)
    boundary_matrix = np.zeros((boundary_matrix_d, boundary_matrix_d), dtype=np.int)
    for j, simp in enumerate(sorted_simplices_list):
        if len(simp[0]) == 1:
            continue
        else:
            d = len(simp[0])
            combos = list(combinations(simp[0], d-1))
            for c in combos:
                boundary_matrix[simplices_indices[c], j] = 1
    return boundary_matrix
        
@njit
def low(vec, ind):
    while ind != -1:
        if vec[ind] == 1:
            return ind
        else:
            ind -= 1
    return None

@njit
def reduce_matrix(R):
    for j in range(R.shape[1]):
        j_low = low(R[:,j], j)
        if j_low is not None:
            all_k = np.ones(j)
            while all_k.any() and (j_low is not None):
                for k in range(j):
                    k_low = low(R[:,k], k)
                    if (k_low is None) or (j_low is None):
                        all_k[k] = False
                    else:
                        all_k[k] = j_low == k_low
            
                    if (k_low is not None) and (j_low is not None) and (k_low == j_low):
                        R[:,j] = np.logical_xor(R[:,j], R[:,k])
                        j_low = low(R[:,j], j)
    return R

@njit
def interpretation(R):
    #Persistence pairs and Homology classes
    P, E = [], []
    for i in range(R.shape[0]):
        flag = np.count_nonzero(R[:,i]) == 0
        if flag:
            check = []
        for j in range(R.shape[1]):
            if R[:,j].any() and (i == low(R[:,j], j)):
                P.append((i,j))

            if low(R[:,j], j) is None:
                continue
            if (i!=low(R[:,j], j)) and flag:
                check.append(True)
            else:
                check.append(False)    
        if np.array(check).all():
            E.append(i)        
    return P, E


#MAIN
def run(X, max_dimension=2):
    assert X.shape[0] > max_dimension, 'Number of points must be greater than maximum simplex dimensionality'
    D = self_distance_matrix(X)
    sorted_simplices_list, simplices_indices = filtered_complex(D, max_dimension)
    boundary_matrix = construct_boundary_matrix(sorted_simplices_list, simplices_indices)
    ##deepcopy matrix
    R = deepcopy(boundary_matrix)
    R = reduce_matrix(R)
    P, E = interpretation(R)
    homologies = [[sorted_simplices_list[p[0]], sorted_simplices_list[p[1]]] for p in P]
    homologies = homologies + [[sorted_simplices_list[e], (None, D.max())] for e in E]
    return homologies
