# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:58:55 2018

@author: david.saltiel
"""

''' 
    All the import for this file
'''
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFE
#from sklearn.ensemble import RandomForestClassifier
from functions import compute_accuracy_score
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
#%%
'''
    Generic RFEMEthod for feature selection
'''
class RFEMethod:
    '''
        df an objec that contains a dataframe contained in df.df 
        the method uses the RFE method from sklearn
        to compute RFE for the df
    '''
    def __init__(self, df, verbose = False):
        self.df = df
        self.verbose = verbose

    '''
        get score and feature for a given number of features
        using the RFE method from sklearn
    '''
    def get_score_and_features(self, n_features):
        features = list(self.df.columns.values)
        features.remove('Label')
        
        df_X = self.df[features]
        df_Y = self.df['Label']
        estimator = XGBClassifier(random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.33,
                                                        random_state = 0)
        
        selector = RFE(estimator, n_features, step=1)
        selector = selector.fit(x_train, y_train)
        
        features_bool = np.array(selector.support_)
        features = np.array(df_X.columns)
        list_feat = list(features[features_bool])
#        list_to_keep = []
#        for i in range(len(list(selector.support_))):
#        	if list(selector.support_)[i] :
#        		list_to_keep.append(i)
#        list_feat = list(df_X.columns[list_to_keep])
#        list_delete = list(set(list(df_X.columns.values))-set(df_X.columns[list_to_keep]))
#        x_test = x_test.drop(list_delete  ,axis=1)
        
#        clf_xgb = XGBClassifier(random_state=0)
#        clf_xgb = clf_xgb.fit(x_train , y_train)
  
        score = selector.score(x_test,y_test)
#        score = compute_accuracy_score(x_test, y_test)
        self.selected_features = list_feat
        self.list_score = score
        return self.selected_features, self.list_score
        
    
    '''
        loop over all features between nMin
        and nMax and generate the corresponding score
        if save_pickle == True, we also save as 
        a pickle the result
    '''
    def save_all_score(self, nMin = 1, nMax = 20, save_pickle = False):
        dic_score_RFE ={}
        for n_features in range(nMin, nMax):
            list_feat, score = self.get_score_and_features(n_features)
            dic_score_RFE[n_features] = [list_feat, score]
            if self.verbose :
                print('n_features : {0} score : {1}'.format(n_features, score))
        if save_pickle:
            pickle.dump( dic_score_RFE, open( "dic_score_RFE_v2.p", "wb" ) )
        return dic_score_RFE

    ''' 
        select features 
        return the subset of initial features kept by the method
        
    '''
    def select_features(self, n_features= 6):
        return self.get_score_and_features(n_features)
