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
from sklearn.ensemble import RandomForestClassifier
from functions import compute_accuracy_score
import pickle

#%%
'''
    Generic RFEMEthod for feature selection
'''
class RFEMethod:
    '''
        data an objec that contains a dataframe contained in data.df 
        the method uses the RFE method from sklearn
        to compute RFE for the data
    '''
    def __init__(self, data, verbose = False):
        self.data = data
        self.verbose = verbose

    '''
        get score and feature for a given number of features
        using the RFE method from sklearn
    '''
    def get_score_and_features(self, n_features):
        features = list(self.data.df.columns.values)
        features.remove('Label')
        
        df_X = self.data.df[features]
        df_Y = self.data.df['Label']
        estimator = RandomForestClassifier(random_state=0)
        
        selector = RFE(estimator, n_features, step=1)
        selector = selector.fit(df_X, df_Y)
        list_to_keep = []
        for i in range(len(list(selector.support_))):
        	if list(selector.support_)[i] :
        		list_to_keep.append(i)
        list_feat = list(df_X.columns[list_to_keep])
        list_delete = list(set(list(df_X.columns.values))-set(df_X.columns[list_to_keep]))
        df_X = df_X.drop(list_delete  ,axis=1)  
        score = compute_accuracy_score(df_X)
        return list_feat, score

    
    '''
        loop over all features between nMin
        and nMax and generate the corresponding score
        if save_pickle == True, we also save as 
        a pickle the result
    '''
    def save_all_score(self, nMin = 1, nMax = 45, save_pickle = False):
        dic_score_RFE ={}
        for n_features in range(nMin, nMax):
            list_feat, score = self.get_score_and_features(n_features)
            dic_score_RFE[n_features] = [list_feat, score]
            if self.verbose :
                print(n_features, score)
        if save_pickle:
            pickle.dump( dic_score_RFE, open( "dic_score_RFE_v2.p", "wb" ) )
        return dic_score_RFE

    ''' 
        select features 
        return the subset of initial features kept by the method
        
    '''
    def select_features(self, n_features= 6):
        return self.get_score_and_features(n_features)
