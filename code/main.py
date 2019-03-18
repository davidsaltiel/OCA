# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:30:59 2018

@author: david.saltiel
"""

from data_loader import DataLoader
from OCA import OCAMethod
from RFE import RFEMethod
from BCA import BCAMethod
from graphics import create_graphic_for_method, create_figure2, create_figure3

''' This is the main function
    It calls all the 3 features selection
    method and compares them
'''
#%% reads the data
data = DataLoader(dataset_name = 'trade_selection_A229', extension = 'csv')
data.clean_data()

#%% OCA method
oca_method = OCAMethod(data.df, n_min=10, verbose = True,reload_feature_importance = False)
oca_method.select_features()

#%% RFE method
rfe_method = RFEMethod(data.df, verbose = True)
rfe_method.select_features()
rfe_dictionary = rfe_method.save_all_score()

#%% BCA method
bca_method = BCAMethod(data.df, verbose = True)
bca_method.select_features()


#%% figure 1 combines oca and bca convergence graphics created below
# graphic for oca convergence
create_graphic_for_method(oca_method, 'Convergence for OCA', 'OCAConvergence')
# graphic for bca convergence
create_graphic_for_method(bca_method, 'Convergence for BCA', 'BCAConvergence')

#%% graphic for rfe comparison figure 2 of the paper
create_figure2(rfe_dictionary, oca_method, bca_method, data.df)

#%% graphic for figure 3 of the paper
create_figure3(rfe_dictionary, oca_method, data.df)


