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
    plt.savefig(filename + '.PNG')
    plt.show()
    
#%%
'''
    function to create figure 2 of the paper
'''
def create_figure2(rfe_dictionary):
    fig, ax = plt.subplots()
    ax.plot(19.444,62.8,'+',color= 'sandybrown', label = 'RFE_28', mew=4, ms=12)
    ax.plot(16.666,62.8,'r+', label = 'OCA', mew=4, ms=12)
    ax.plot(16.666,62.39,'+',color= 'darkorange', label = 'RFE_24', mew=4, ms=12)
    ax.plot(27.08,62.19,'b+', label = 'BCA', mew=4, ms=12)
    
    #ax.plot([0,100], [100,0] ,label='Border 1')
    #ax.plot([0,100], [0,100] ,label='Border 2')
    ax.set(xlabel='% of features used', ylabel='score',
           title='Comparison of methods')
    ax.grid()
    plt.legend()
    plt.savefig('Comparison_methods.png')
    plt.show()
    


#%%
'''
    function to create figure 3 of the paper
'''
def create_figure3(rfe_dictionary):
    x = rfe_dictionary.keys
    y = rfe_dictionary.values
    fig, ax = plt.subplots()
    ax.plot(x, y,'b+' ,markersize=6, label = 'RFE')
    ax.plot([16.666,16.666], [0,64] ,':',linewidth=2.5,color = 'black',label='Best features set')
    ax.plot([10,30], [62.8,62.8],':',linewidth=2.5,color = 'black' ,label='Best score')
    ax.plot(16.666,62.8,'rP',markersize=8, label = 'OCA')
    ax.plot(19.444, 62.8,'P',color= 'darkorange' ,markersize=8, label = 'Optimal RFEs')
    ax.plot(16.666, 62.39,'P',color= 'darkorange' ,markersize=8)
    plt.ylim(53,64)
    plt.xlim(10,30)
    ax.set(xlabel='% of features used', ylabel='score',
           title='RFE vs OCA')
    ax.grid()
    plt.legend()
    plt.savefig('RFE_vs_OCA.png')
    plt.show()