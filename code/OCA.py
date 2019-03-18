# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:25:38 2018

@author: david.saltiel
"""
#%%
import pandas as pd
import numpy as np
from functions import compute_feature_importance, compute_k_best
from functions import compute_j_best, get_k_j_best_list
from BCA import BCAFunction

#%%

'''
    Generic OCAMethod for feature selection
'''
class OCAMethod:
    def __init__(self, df, n_min = 3, verbose = True, reload_feature_importance = False, 
                 method = 'XGB' , split = 'temporal'):
        self.df = df
        self.n_min = n_min
        self.verbose = verbose
        self.reload_feature_importance = reload_feature_importance
        self.method = method
        self.split = split

    '''
        First loop of the OCA method
    '''        
    def __get_k_best_j_best(self):
        list_score = []
        if self.reload_feature_importance == False:
            if self.verbose:
                print('Computation of features importance to start the greedy algorithm :')    
            compute_feature_importance(self.df, self.method ,self.split)
        else :
            if self.verbose:
                print('We use the previous computation of features importance to start the greedy algorithm :')
        
        list_var = ['Block1',  'Block2', 'Block3', 'Block4', 'Block5', 'Block6']
    
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
        if self.verbose:
            print('##\nInitialization\n##')
        Xold, best_k, score_test_old, l_scores = compute_k_best(self.df, self.n_min, index_best_k, self.method, self.split)
        list_score.extend(l_scores)
        if self.verbose:
            print('Xold : {0}\nScore : {1}'.format( Xold, score_test_old) )
        
        X_init =  Xold.copy()
        condition = True
            
        while condition :
            index_best_k = 1
            
            if self.verbose:
                print( '##\nIteration : {0}\nIndex : {1}\n##\nXold :{2}'.format(iteration, index, Xold))
                
            # To repeat the computation over the vector X
            if index == len(list_var) :
                index =0
                loop += 1
                if self.verbose:
                    print('We restart the loop over the vector\nIndex : {0}\n'.format(index))
                
            if loop >0 :
                count +=1
            l_test_j, l_scores_compute = compute_j_best(self.df, self.n_min, index, best_k, Xold,
                                                   self.method, list_score,self.split)
            list_score = l_scores_compute
            best_j = np.argmax(l_test_j) + index_best_k
            if self.verbose:
                print('best j ',best_j)
            score_test_new = max(l_test_j)
            Xnew = Xold.copy()
            Xnew[index] = best_j
            
            if self.verbose:
                print('Xnew : {0}\nScore : {1}'.format(Xnew, score_test_new))
            
            while Xnew == X_init and iteration < min_iter and loop != 0 :
                index_best_k += 1
                iteration += 1
                if self.verbose:
                    print('\nLocal minimum :\nWe need to take the {0} best score'.format(index_best_k))
                    print('##\nIteration : {0}\n##'.format(iteration))
                
                
                best_j = np.argmax(l_test_j) + index_best_k 
#                score_new = np.sort(l_test_j)[-index_best_k]
#                list_score.extend([score_new])
                Xnew[index] = best_j
               
                
            condition = (iteration < max_iter and Xnew !=Xold and
                         abs(score_test_old - score_test_new)>= tol) or count < len(list_var)
            if self.verbose:
                print(condition)    
            Xold = Xnew.copy()
            score_test_old = score_test_new
            # To explore the vector X :
            index += 1
            
            iteration += 1
            self.X = Xnew
            self.list_score = list_score
            
    ''' This is the second step of the method
    '''
    def __binary_coordinate_ascent(self):
        df2 = pd.read_csv('features_importance_' + self.method + '.csv')
        list_col = list(df2['Variable'])
        step_1_solution = get_k_j_best_list(list_col, self.X[0], 'Block1', self.X)
        tol = 1e-10
        a, b, c, d_x, d_y, l_step2 = BCAFunction(self.df, tol, step_1_solution,
                                                 self.list_score, self.verbose,self.split)
        self.selected_features = c
        self.list_score = l_step2
        
    
    ''' 
        select features 
        return 
            - the subset of initial features kept by the method
            - the scores at each iterations
        
    '''
    def select_features(self):
        # step 1: j best optimization
        self.__get_k_best_j_best()
        
        # step 2: full coordinate ascent binary optimization
        self.__binary_coordinate_ascent()
        return self.selected_features, self.list_score