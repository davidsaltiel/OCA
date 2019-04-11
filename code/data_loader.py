import pandas as pd
from datetime import datetime
from os.path import dirname, join
import re

'''
    A generic class to load our data
'''
class DataLoader(object):
    def __init__(self, dataset_name = 'trade_selection', extension = 'csv'):
        module_path = dirname(__file__)
        short_filename = dataset_name + '.' + extension
        filename = join(module_path, '..', 'data', short_filename)
        self.dataset_name = dataset_name
        self.extension = extension
        
        if extension == 'csv':
            try :
                self.df = pd.read_csv(filename, thousands=',')
            except : 
                self.df = pd.read_csv(filename, thousands=',',encoding='latin-1')
        elif extension == 'xlsx':
            self.df = pd.read_excel(filename, thousands=',')
        else:
            raise Exception('Unknown extension type')
        

    ''' A simple function to get the date time from
        the datetime input
    '''
    def __convert(self, u):
        if self.extension == 'csv':
            u=datetime.strptime(u, '%d/%m/%Y %H:%M').strftime('%H%M%S')
        elif self.extension == 'xlsx':
            u=u.strftime('%Y-%m-%d %H:%M:%S')
            u=datetime.strptime(u, '%d/%m/%Y %H:%M').strftime('%H%M%S')
        else:
            raise Exception('Unknown extension type')
        z = int(u)
        return z
            
    '''
    Get the list of the block variables
    '''        
    def get_list_var_block(self):
        keys = self.df.columns
        block_variables_tmp = []   
        block_variables = []
        for key in keys:
            result = re.findall("([A-z_ €\-\(\),0-9]+)_[0-9]+", key)
            if len(result) == 1:
                if result[0] not in block_variables_tmp:
                    m = 1
                    block_variables_tmp.append(result[0]) 
                else :
                    m += 1
                    if m == 2 :
                        block_variables.append(result[0])
        
        return block_variables    

    ''' clean data 
        removes zero and some useless columns
        data specific
    '''
    def clean_data(self, threshold = -160):
        # Features_treatment :
        list_rows_to_delete = []
        for i in range(self.df.shape[0]):
            if self.df.iloc[i,1:].sum() == 0:
                list_rows_to_delete.append(i)
        self.df = self.df.drop(list_rows_to_delete,axis=0)
        self.df = self.df.reset_index(drop=True)
        
        if self.dataset_name == 'trade_selection':
            self.df['Label'] = (self.df['Result']>threshold)*1
            self.df["Entry time"] =  self.df["Entry time"].apply(lambda x: self.__convert(x))
            self.df['dif_Block1_0'] = self.df['Block1_0'] - self.df['Block1_19']
             
            self.df['dif_Block2_0'] = self.df['Block2_0'] - self.df['Block2_29']
            self.df['dif_Block3_0'] = self.df['Block3_0'] - self.df['Block3_19']
            self.df['mean_Block1_0'] = self.df.iloc[:,2:22].mean(axis=1)
            self.df['mean_Block2_0'] = self.df.iloc[:,22:52].mean(axis=1)
            self.df['mean_Block3_0'] = self.df.iloc[:,52:72].mean(axis=1)
            self.df['mean_Block4_0'] = self.df.iloc[:,72:92].mean(axis=1)
            self.df['mean_Block5_0'] = self.df.iloc[:,92:112].mean(axis=1)
            self.df['mean_Block6_0'] = self.df.iloc[:,112:132].mean(axis=1)

            # Drop columns :
#            list_cols_to_delete = ['Result']
#            self.df = self.df.drop(list_cols_to_delete  ,axis=1)
            return self.df
