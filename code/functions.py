# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:57:09 2018

@author: david.saltiel
"""

#%%

import numpy as np
from datetime import datetime
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

#%%
'''
    A generic file to contain various basic functions
    for our feature selection
'''


'''
    compute accuracy score
'''
def compute_accuracy_score(X, Y, split):
    random.seed(0)

    if split == 'temporal' :
        split_data = int(0.67*X.shape[0])
        x_train, x_test, y_train, y_test = X.iloc[:split_data,:], X.iloc[split_data:,:],\
                                            Y.iloc[:split_data], Y.iloc[split_data:]
    else :
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,
                                                        random_state = 0)
    

    clf_xgb = XGBClassifier(random_state=0)
    dic_param = { 'n_estimators':[100, 150, 200, 250] , 
          'max_depth' : [3, 4, 5, 6, 7, 9, 10,11,12] ,
          'learning_rate' : [0.01, 0.02, 0.05, 0.07, 0.1] , 
          'gamma' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 10] , 
          'colsample_bylevel' : [1, 0.7],
          'colsample_bytree' : [1, 0.7],
          'subsample' : [1, 0.7],
          'reg_lambda' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 5, 8, 10],
          'min_child_weight' : [1,2,3,5,6]
					}

    OPT = RandomizedSearchCV( clf_xgb, 
              param_distributions = dic_param, 
                                cv = 3 , scoring = 'roc_auc',
                                     n_iter = 60, n_jobs=-1,
                                     random_state = 0 )
    OPT.fit(x_train , y_train)
    model = OPT.best_estimator_
    return model.score(x_test,y_test)

def selec_feat(binary_list, df, drop) :
    l_drop = []
    l_keep = []
    assert len(binary_list) == len(list(df.columns))
    for i in range(len(list(df.columns))):
        if binary_list[i] == 0 :
            l_drop.append(list(df.columns)[i])
        else :
            l_keep.append(list(df.columns)[i])
    df = df.drop(l_drop, axis=1)
    if drop :
        return df
    else :
        return l_keep



def score(X, df_X, df_Y, split):
    if X.sum() != 0:
        return compute_accuracy_score(selec_feat(X, df_X, True), df_Y, split)
    else :
        return 0

#################################################################


def compute_feature_importance(df, algo, split):
    
    random.seed(0)
    features_ = list(df.columns.values)
    features_.remove('Label')
    
    X = df[features_]
    Y = df['Label']
    
    if split == 'temporal' :
        split_data = int(0.67*df.shape[0])
        x_train, x_test, y_train, y_test = X.iloc[:split_data,:], X.iloc[split_data:,:],\
                                            Y.iloc[:split_data], Y.iloc[split_data:]
    else :
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,
                                                        random_state = 0)
    if algo == 'RF' :
        
        clf_rf = RandomForestClassifier(random_state=0)
        
        dic_param = { 'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)] ,
                     'max_depth' : [int(x) for x in np.linspace(5, 100, num = 20)] ,
                     'min_samples_split' : [ 2, 3, 4, 5, 7, 9, 15] , 
                     'min_samples_leaf' : [ 2, 3, 4, 5, 7, 9, 15] , 
                     'criterion' : ['gini', 'entropy']
                     }
        
        OPT = RandomizedSearchCV( clf_rf , 
                                  param_distributions = dic_param, 
                                                    cv = 3 , scoring = 'roc_auc',
                                                         n_iter = 60, n_jobs=-1,
                                                         random_state = 0 )
        
        OPT.fit(x_train , y_train)
        
        model=OPT.best_estimator_
        model.fit(x_train , y_train)
        
        feature_importance=pd.DataFrame(model.feature_importances_)
        feature_importance = pd.concat([pd.DataFrame(
            x_train.columns.values), feature_importance], axis=1, ignore_index=True)
        feature_importance.columns = ['Variable', 'Importance']
        feature_importance = feature_importance.sort_values(
            by='Importance', axis=0, ascending=False)
        
        feature_importance.to_csv(
            'features_importance_RF.csv',
            sep=',',
            index=False,encoding='latin-1')
    else :
        clf_xgb = XGBClassifier(random_state=0)
        dic_param = { 'n_estimators':[100, 150, 200, 250] , 
              'max_depth' : [3, 4, 5, 6, 7, 9, 10,11,12] ,
              'learning_rate' : [0.01, 0.02, 0.05, 0.07, 0.1] , 
              'gamma' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 10] , 
              'colsample_bylevel' : [1, 0.7],
              'colsample_bytree' : [1, 0.7],
              'subsample' : [1, 0.7],
              'reg_lambda' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 5, 8, 10],
              'min_child_weight' : [1,2,3,5,6]
					}
        
        OPT = RandomizedSearchCV( clf_xgb , 
                                  param_distributions = dic_param, 
                                                    cv = 3 , scoring = 'roc_auc',
                                                         n_iter = 60, n_jobs=-1,
                                                         random_state = 0 )
        
        OPT.fit(x_train , y_train)
        
        model=OPT.best_estimator_
        model.fit(x_train , y_train)
        
        feature_importance=pd.DataFrame(model.feature_importances_)
        feature_importance = pd.concat([pd.DataFrame(
            x_train.columns.values), feature_importance], axis=1, ignore_index=True)
        feature_importance.columns = ['Variable', 'Importance']
        feature_importance = feature_importance.sort_values(
            by='Importance', axis=0, ascending=False)
        
        feature_importance.to_csv(
            'features_importance_XGB.csv',
            sep=',',
            index=False,encoding='latin-1')


#################################################################

def analysis(df, j, other_arguments, algo, split) :
    

    if other_arguments is not None and len(other_arguments) == 2:
        var, Xold = other_arguments[0], other_arguments[1]
    else :
        Xold = "Initialization"

    random.seed(0)
    
    
    
#    df = treat_data(df)
   
    if algo == 'RF' :
        # read file with features importance :
        df2 = pd.read_csv('features_importance_RF.csv')
    else :
        df2 = pd.read_csv('features_importance_XGB.csv')
        
    list_col = list(df2['Variable'])

    
    if other_arguments is not None:
        list_to_keep = get_k_j_best_list(list_col, j, var, Xold)
    else:
        list_to_keep = get_k_best_features(list_col, j) 
    
    
    list_delete = list(set(list(df.columns.values))-set(list_to_keep))
    	
    df = df.drop(list_delete  ,axis=1)
    			
    		
    features_ = list(df.columns.values)
    features_.remove('Label')
    
    X = df[features_]
    Y = df['Label']
    
    if split =='temporal' :
        split_data = int(0.67*df.shape[0])
        x_train, x_test, y_train, y_test = X.iloc[:split_data,:], X.iloc[split_data:,:],\
                                            Y.iloc[:split_data], Y.iloc[split_data:]
    else :
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,
                                                        random_state = 0)
    
    if algo == 'RF' :
        clf_rf = RandomForestClassifier(random_state=0)
        
        dic_param = { 'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)] ,
                     'max_depth' : [int(x) for x in np.linspace(5, 100, num = 20)] ,
                     'min_samples_split' : [ 2, 3, 4, 5, 7, 9, 15] , 
                     'min_samples_leaf' : [ 2, 3, 4, 5, 7, 9, 15] , 
                     'criterion' : ['gini', 'entropy']
                     }
        
        OPT = RandomizedSearchCV( clf_rf , 
                                  param_distributions = dic_param, 
                                                    cv = 3 , scoring = 'roc_auc',
                                                         n_iter = 60, n_jobs=-1,
                                                         random_state = 0 )
        OPT.fit(x_train , y_train)
    
        filename = "RF_optimization_trace_{0}.txt".format(datetime.now().strftime("%Y-%m-%d %H"))
        file = open(filename, mode = 'a')
        file.write("{0}: Xold={1}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"), Xold) )
        file.write("Best Params : %s" % str (OPT.best_params_))
        file.write("\n\n")
        file.close()
        
    else :
        clf_xgb = XGBClassifier(random_state=0)
    
        dic_param = { 'n_estimators':[100, 150, 200, 250] , 
                  'max_depth' : [3, 4, 5, 6, 7, 9, 10,11,12] ,
                  'learning_rate' : [0.01, 0.02, 0.05, 0.07, 0.1] , 
                  'gamma' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 10] , 
                  'colsample_bylevel' : [1, 0.7],
                  'colsample_bytree' : [1, 0.7],
                  'subsample' : [1, 0.7],
                  'reg_lambda' : [ 0.01, 0.05, 0.1, 0.5, 0.7, 1, 5, 8, 10],
                  'min_child_weight' : [1,2,3,5,6]
					}
    
        OPT = RandomizedSearchCV( clf_xgb , 
                              param_distributions = dic_param, 
                                                cv = 3 , scoring = 'roc_auc',
                                                     n_iter = 60, n_jobs=-1,
                                                     random_state = 0 )
        
        OPT.fit(x_train , y_train)
   
        filename = "XGB_optimization_trace_{0}.txt".format(datetime.now().strftime("%Y-%m-%d %H"))
        file = open(filename, mode = 'a')
        file.write("{0}: Xold={1}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"), Xold) )
        file.write("Best Params : %s" % str (OPT.best_params_))
        file.write("\n\n")
        file.close()
        
        
    model = OPT.best_estimator_
    return model.score(x_test,y_test)


#################################################################
''' returns from the long key:"F_Close Algo Bars_0"
          the short key = "F_Close Algo Bars"
'''
def get_short_key_from_long_key( long_key):
    pos  = long_key.find('_', len(long_key)-4)
    return long_key[:pos] 


#################################################################
def get_k_best_features(list_col, k):
    best_features_keys = {'key': 1}
    list_to_keep = []
    for feature in list_col:
        short_key = get_short_key_from_long_key(feature)
        if short_key not in best_features_keys:
            best_features_keys[short_key ] = 1
            list_to_keep.append(feature)
        elif best_features_keys[short_key] < k:
            best_features_keys[short_key ] += 1
            list_to_keep.append(feature)
    list_to_keep.extend(['Label'])

    return list_to_keep

def get_k_j_best_list(list_col, j, var, Xold) :
    
    list_var = ['Block1', 
                'Block2',
                'Block3',
                'Block4',
                'Block5',
                'Block6']
    
    best_features_keys = {'key': 1}
    list_to_keep = []

    # found the first key that can vary    
    short_key_visited  = [ get_short_key_from_long_key(list_col[0]) ]

    for feature in list_col:
        short_key = get_short_key_from_long_key(feature)
       
        if short_key not in short_key_visited and short_key not in best_features_keys :
            short_key_visited.append(short_key)
              
        if short_key not in best_features_keys:
            best_features_keys[short_key ] = 1
            list_to_keep.append(feature)
        elif var == short_key:
            if best_features_keys[short_key] < j :
                best_features_keys[short_key ] += 1
                list_to_keep.append(feature)
        elif var != short_key:
            if best_features_keys[short_key] < Xold[list_var.index(short_key)]: 
                best_features_keys[short_key ] += 1
                list_to_keep.append(feature)
        
    list_to_keep.extend(['Label'])
    return list_to_keep

#################################################################



def compute_k_best(df, n, index_best_k, algo, split):
    
    '''
    Initialisation :
    '''
    
    list_var = ['Block1', 
                'Block2',
                'Block3',
                'Block4',
                'Block5',
                'Block6']
    
    
    
    '''
    
    Computation of the vector X0 :
    '''
    

    l_scores = []
    score_step = [0]
    for k in range(1,n) :       
        sc_test =  analysis(df, k, None, algo, split)
        l_scores.append(sc_test)
        if sc_test < score_step[-1] :
            score_step.append(score_step[-1])
        else :
            score_step.append(sc_test)
    print(l_scores)
    best_k = np.argmax(l_scores) + index_best_k 
    
    Xold = [best_k]*len(list_var)
    
    score_test_old = max(l_scores)
    del score_step[0]
    return Xold, best_k, score_test_old, score_step

#################################################################

'''
Step 1 :
'''

def compute_j_best(df, n, index, k_max, Xold, algo, score_step, split) :
    
    list_var = ['Block1', 
                'Block2',
                'Block3',
                'Block4',
                'Block5',
                'Block6']
    
    var = list_var[index]
    
    
    l_scores = []
    for j in range(1,n):
  
        sc_test_j = analysis(df, j, [var, Xold], algo, split)
        l_scores.append(sc_test_j)

        if sc_test_j < score_step[-1] :
            score_step.append(score_step[-1])
        else :
            score_step.append(sc_test_j)

    print(l_scores)

    
    return l_scores, score_step
