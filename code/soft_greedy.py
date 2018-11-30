# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:34:44 2018

@author: david.saltiel
"""

#%%
import os
folder_path = "C:\\Users\\david.saltiel\\Documents\\AlphaBetaMatrix\\OCA\\code"
os.chdir(folder_path)


#%%
import numpy as np
from functions import init, compute_X, step0


#%%

#%%
def soft_greedy(df, n, algo, reload) :


    list_score = []
    if reload == False :
        print('Computation of features importance to start the greedy algorithm :')    
        step0(df, algo)
    else :
        print('We use the previous computation of features importance to start the greedy algorithm :')
    
    list_var = ['Block1', 
                'Block2',
                'Block3',
                'Block4',
                'Block5',
                'Block6']
    
    iteration = 0
    max_iter = 20
    
    min_iter = 10
    index = 0
    tol = 1e-7
    # To check if we 've made a complete tour over the vector
    loop = 0
    count = 0
    # index_best_k = 1 -> we take the 1st best k (if 2 -> we take the 2nd best) ...
    index_best_k = 1
    
    # computation of the vector X0
    print('##')
    print('Initialization')
    print('##')
    Xold, best_k, score_test_old, l_scores = init(df, n, index_best_k, algo)
    list_score.extend(l_scores)
    print(Xold) 
    print('Score ', score_test_old)
    
    X_init =  Xold.copy()
    condition = True
    
    while condition :
        
        index_best_k = 1
        
        print('##')
        print('Iteration : ', iteration)
        print('Index : ', index)
        print('##')
        
        print('xold ',Xold) 
        
        # To repeat the computation over the vector X
        if index == len(list_var) :
            index =0
            loop += 1
            print('')
            print("We restart the loop over the vector")
            print('Index : ', index)
            print('')
            
        if loop >0 :
            count +=1
        l_test_j, l_scores_compute = compute_X(df, n, index, best_k, Xold,
                                               algo, list_score)
        list_score = l_scores_compute
        best_j = np.argmax(l_test_j) + index_best_k
        print('best j ',best_j)
        score_test_new = max(l_test_j)
        Xnew = Xold.copy()
        Xnew[index] = best_j
        
        print(Xnew)
        print('Score : ',score_test_new)
        
        
        
        
        while Xnew == X_init and iteration < min_iter and loop != 0 :
            index_best_k += 1
            iteration += 1
            print('')
            print('Local minimum :')
            print("We need to take the ",index_best_k, " best score")
            print('')
            
            print('##')
            print('Iteration : ', iteration)
            
            print('##')
            
            
            best_j = np.argmax(l_test_j) + index_best_k 
            score_new = np.sort(l_test_j)[-index_best_k]
            list_score.extend(score_new)
            Xnew[index] = best_j
            print(Xnew)
            print('Score : ',score_new)
            
        condition = (iteration < max_iter and Xnew !=Xold and
                     abs(score_test_old - score_test_new)>= tol) or count < len(list_var)
        print('Xnew !=Xold',Xnew !=Xold) 
        print('loop', loop)
        print('abs(score_test_old - score_test_new)>= tol)',abs(score_test_old - score_test_new)>= tol)  
        print('count < len(list_var)',count < len(list_var))             
        print(condition)    
        Xold = Xnew.copy()
        score_test_old = score_test_new
        # To explore the vector X :
        index += 1
        
        iteration += 1
    return Xnew, list_score
