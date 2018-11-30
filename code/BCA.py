# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:08:44 2018

@author: david.saltiel
"""

#%%
'''

Binary coordinate ascent algorithm :
we represent the N features by a vector in x in (0,1)^n 
x(i) = 1 -> features i in the model 
'''

import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from functions import score, selec_feat

#%%

def BCAFunction(df, tol, list_treat, list_score):
    
    start = datetime.now()
    step = 0
    print('step : ', step)
#    df = treat_data(df)
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

    y_opt = score(X_opt, df_X_init, df_Y)
    while condition :
        step +=1
        for i in range(N):            
            X = X_opt.copy()
            X[i] = int(not X_opt[i])  
            score_X = score(X, df_X_init, df_Y) 
            score_X_opt = score(X_opt, df_X_init, df_Y)
            if  score_X >= score_X_opt :
                if score_X > list_score[-1] :
                    print('score ',score(X, df_X_init, df_Y))
                    list_score.append(score_X)
                else :
                    list_score.append(list_score[-1])
                    
                X_opt = X.copy()
            else :
                if score_X > list_score[-1] :   
                    list_score.append(score_X)
                else :
                    list_score.append(list_score[-1])
        
        y = score(X_opt, df_X_init, df_Y)
        print('step : ', step)        
        print('y_opt : ', y_opt)
        print('y : ', y)
        if abs(y - y_opt)<tol or step > 10 :
            condition = False
        y_opt = y
    list_features = selec_feat(X_opt, df_X_init, False)
    print('It took : ',datetime.now()-start)
    return X_opt, y_opt, list_features, df_X_init, df_Y, list_score


class BCAMethod:
    def __init__(self, data, tol =  1e-10):
        self.data = data
        
    ''' 
        select features 
        return 
            - the subset of initial features kept by the method
            - the scores at each iterations
        
    '''
    def select_features(self):
        X_opt, y_opt, list_features, df_X_init, df_Y, list_score = \
            BCAFunction(self.data.df, self.tol, list(self.data.df.columns), [0])
        return list_features, list_score
        

