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


# get the argument for our class
data = DataLoader(dataset_name = 'trade_selection', extension = 'csv')
data.clean_data()

# OCA method
oca_method = OCAMethod(data, n_min=3, verbose=True)
oca_method.select_features()

# RFE method
rfe_method = RFEMethod(data)
rfe_method.select_features()
ref_dictionary = rfe_method.save_all_score(verbose = False)

# BCA method
bca_method = BCAMethod(data.df)
bca_method.select_features()


# figure 1 combines oca and bca convergence graphics created below
# graphic for oca convergence
create_graphic_for_method(oca_method, 'Convergence for OCA', 'OCAConvergence')
# graphic for bca convergence
create_graphic_for_method(bca_method, 'Convergence for BCA', 'BCAConvergence')

# graphic for rfe comparison figure 2 of the paper
create_figure2(ref_dictionary)

# graphic for figure 3 of the paper
create_figure3(ref_dictionary)


