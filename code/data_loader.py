import pandas as pd
from datetime import datetime
from os.path import dirname, join


class DataLoader(object):
    def __init__(self, dataset_name = 'trade_selection', extension = 'csv'):
        module_path = dirname(__file__)
        short_filename = dataset_name + '.' + extension
        filename = join(module_path, '..', 'data', short_filename)
        self.dataset_name = dataset_name
        self.df = pd.read_csv(filename, thousands=',')
        
    def __convert(self, u):
        u=datetime.strptime(u, '%d/%m/%Y %H:%M').strftime('%H%M%S')
        z = int(u)
        return z
        


    def clean_data(self, threshold = -160):
        # Features_treatment :
        list_rows_to_delete = []
        for i in range(self.df.shape[0]):
            if self.df.iloc[i,1:].sum() == 0:
                list_rows_to_delete.append(i)
        self.df = self.df.drop(list_rows_to_delete,axis=0)
        
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
            list_cols_to_delete = ['Result']
            self.df = self.df.drop(list_cols_to_delete  ,axis=1)

