import matplotlib.pyplot as plt
import numpy as np


''' This is the code to generate the
    graphics of the paper
    Because of time, we temporarily 
    hard coded some results
    Need to be revisited
'''


#%%
'''
    generic function for figure 1 of the paper
'''
def create_graphic_for_method(method, title, filename):
    list_score = method.list_score
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(list_score)),list_score,'-', markersize=8)
    ax.set(xlabel='Number of iterations ', ylabel='score',
           title=title)
    ax.grid()
    plt.legend()
#    plt.savefig(filename + '.PNG')
    plt.show()
    
#%%
'''
    function to create figure 2 of the paper
'''
def create_figure2(rfe_dictionary, oca_method, bca_method, df):
    features_oca = oca_method.selected_features
    scores_oca = oca_method.list_score
    features_bca = bca_method.selected_features
    scores_bca = bca_method.list_score
    rfe_values = [el[1] for el in list(rfe_dictionary.values())]
    best_score = np.max(rfe_values)
    best_key = np.argmax(rfe_values)
    features_nb = df.shape[1]
    rfe_point2 = rfe_dictionary[len(features_oca)]
    fig, ax = plt.subplots()
    ax.plot(best_key/features_nb, best_score,'+', color= 'sandybrown', label = 'RFE_'+str(best_key), mew=4, ms=12)
    ax.plot(len(features_oca)/features_nb, scores_oca[-1], 'r+', label = 'OCA', mew=4, ms=12)
    ax.plot(len(rfe_point2[0])/features_nb, rfe_point2[1],'+', color= 'darkorange', label = 'RFE_24', mew=4, ms=12)
    ax.plot(len(features_bca)/features_nb, scores_bca[-1], 'b+', label = 'BCA', mew=4, ms=12)
    
    ax.set(xlabel='% of features used', ylabel='score',
           title='Comparison of methods')
    ax.grid()
    plt.legend()
#    plt.savefig('Comparison_methods.png')
    plt.show()
    


#%%
'''
    function to create figure 3 of the paper
'''
def create_figure3(rfe_dictionary, oca_method, df):
    features_nb = df.shape[1]
    features_oca = oca_method.selected_features
    scores_oca = oca_method.list_score
    x = rfe_dictionary.keys()
    y = [el[1] for el in list(rfe_dictionary.values())]
    best_score = np.max(y)
    best_key = np.argmax(y)
    proportion_oca = (len(features_oca)/features_nb) 
    rfe_point2 = rfe_dictionary[len(features_oca)]
    fig, ax = plt.subplots()
    ax.plot(x, y,'b+' ,markersize=6, label = 'RFE')
    ax.plot([proportion_oca,proportion_oca], [0,scores_oca[-1] +0.2] ,':',linewidth=2.5,color = 'black',label='Best features set')
    ax.plot([proportion_oca - 0.1,proportion_oca +0.5], [scores_oca[-1],scores_oca[-1]],':',linewidth=2.5,color = 'black' ,label='Best score')
    ax.plot(proportion_oca, scores_oca[-1],'rP',markersize=8, label = 'OCA')
    ax.plot(best_key/features_nb, best_score,'P',color= 'darkorange' ,markersize=8, label = 'Optimal RFEs')
    ax.plot(len(rfe_point2[0])/features_nb, rfe_point2[1],'P',color= 'darkorange' ,markersize=8)
    plt.ylim(0,scores_oca[-1] +0.2)
    plt.xlim(proportion_oca - 0.1,proportion_oca +0.5)
    ax.set(xlabel='% of features used', ylabel='score',
           title='RFE vs OCA')
    ax.grid()
    plt.legend()
#    plt.savefig('RFE_vs_OCA.png')
    plt.show()