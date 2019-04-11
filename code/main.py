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
from impact_threshold import analysis_threshold
from score_result import test_set_result, plot_result, plot_hist

''' This is the main function
    It calls all the 3 features selection
    method and compares them
'''
#%%
# Read the data:
data = DataLoader(dataset_name = 'trade_selection', extension = 'csv')
split = int(data.df.shape[0]*0.8)

# Compute a threshold to create the label:
threshold_ = analysis_threshold(data.df, split)

# Some plots of the variable Result:
plot_result(data.df)
plot_hist(data.df)

# Clean the data:
datas = data.clean_data(threshold = threshold_)
old_df = datas.copy()
datas = datas.drop(['Result'], axis = 1)

# Split the train_test and evaluation set:
train_evaluation_set = datas.iloc[:split,:]
test_set = datas.iloc[split:,:]

# Get the block variables:
list_var = data.get_list_var_block()

#%% OCA method
oca_method = OCAMethod(train_evaluation_set, n_min=10, verbose = True,
                reload_feature_importance = False, list_var = list_var)
oca_method.select_features()

#%% RFE method
rfe_method = RFEMethod(train_evaluation_set, verbose = True)
rfe_method.select_features()
rfe_dictionary = rfe_method.save_all_score(nMax = len(oca_method.selected_features)+1)

#%% BCA method
bca_method = BCAMethod(train_evaluation_set, verbose = True)
bca_method.select_features()


#%% figure 1 combines oca and bca convergence graphics created below
# graphic for oca convergence
create_graphic_for_method(oca_method, 'Convergence for OCA', 'OCAConvergence')
# graphic for bca convergence
create_graphic_for_method(bca_method, 'Convergence for BCA', 'BCAConvergence')

#%% graphic for rfe comparison figure 2 of the paper
create_figure2(rfe_dictionary, oca_method, bca_method, train_evaluation_set)

#%% graphic for figure 3 of the paper
create_figure3(rfe_dictionary, oca_method, train_evaluation_set)

#%%
# Some results of the methods on train and test set:
test_set_result(oca_method, train_evaluation_set, test_set,old_df, data.df)
test_set_result(bca_method, train_evaluation_set, test_set,old_df, data.df)

