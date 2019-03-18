# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:08:44 2018

@author: david.saltiel
"""

'''
Binary coordinate ascent algorithm :
The algorithm works as follows:
we represent the N features by a vector x in (0,1)^n 
x(i) = 1 -> features i in the model 
'''

import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from functions import score, selec_feat

#%%
''' This is a generic function
    for BCA, given a tolerance tol
    a list of features to treat
    and a list of score

    This function is called both in BCAMethod
    and in step 2 of OCAMEthod
'''
def BCAFunction(df, tol, list_treat, list_score, verbose, split):
    start = datetime.now()
    step = 0
    if verbose:
        print('step : ', step)
    list_treat = list(df.columns.values)
    list_delete = list(set(list(df.columns.values))-set(list_treat))
    	
    df = df.drop(list_delete  ,axis=1)
    features = list(df.columns.values)
    features.remove('Label')
    df_X = df[features]
    N = df_X.shape[1]
    df_X_init = df_X.copy()
    df_Y = df['Label']
    condition = True
    X_opt = np.zeros(N)

    y_opt = score(X_opt, df_X_init, df_Y, split)
    while condition :
        step +=1
        for i in range(N):            
            X = X_opt.copy()
            X[i] = int(not X_opt[i])  
            score_X = score(X, df_X_init, df_Y, split) 
            score_X_opt = score(X_opt, df_X_init, df_Y, split)
            if  score_X >= score_X_opt :
                if score_X > list_score[-1] :
                    print('score : ', score(X, df_X_init, df_Y, split))
                    list_score.append(score_X)
                else :
                    list_score.append(list_score[-1])
                    
                X_opt = X.copy()
            else :
                if score_X > list_score[-1] :   
                    list_score.append(score_X)
                else :
                    list_score.append(list_score[-1])
        
        y = score(X_opt, df_X_init, df_Y, split)
        if verbose:
            print('step : {0}\ny_opt : {1}\ny :{2}'.format(step, y_opt, y))
        if abs(y - y_opt)<tol or step > 10 :
            condition = False
        y_opt = y
    list_features = selec_feat(X_opt, df_X_init, False)
    if verbose:
        print('It took : ',datetime.now()-start)
    return X_opt, y_opt, list_features, df_X_init, df_Y, list_score
   


'''
    Generic BCAMethod for feature selection
'''
class BCAMethod:
    def __init__(self, df, tol =  1e-10, verbose = True, split = 'temporal'):
        self.df = df
        self.verbose = verbose
        self.tol = tol
        self.split = split
        
    ''' 
        select features 
        return 
            - the subset of initial features kept by the method
            - the scores at each iterations
        
    '''
    def select_features(self):
        X_opt, y_opt, list_features, df_X_init, df_Y, list_score = \
            BCAFunction(self.df, self.tol, list(self.df.columns), [0],
                        self.verbose, self.split)
        self.list_score = list_score
        self.selected_features = list_features
        return self.selected_features, self.list_score        
        
        

