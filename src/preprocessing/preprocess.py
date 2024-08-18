import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def nan_values(data,null_cols):
    # Tratamiento de ausentes
    
    for column in null_cols:   
        if data[column].isna().sum()/data.shape[0] < 0.15:
            mode=data[column].mode()[0]
            data[column].fillna(value=mode,inplace=True)
        elif data[column].isna().sum()/data.shape[0] > 0.15:
            data.dropna(inplace=True)
    return data

def duplicated_values(data):
    # Tratamiento de duplicados
    if data.duplicated().sum() > 0:
            data=data.drop_duplicates()
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID''' 
        
    
    
    # Pasamos columnas a formato snake_case
    data = columns_transformer(data)
    
    data=nan_values(data,data.columns)
    
    # Boxplots
    
    numeric=['age', 'salary', 'family_members', 'insurance_benefits']
    for column in numeric:
        fig,ax=plt.subplots()
        ax.boxplot(data[column])
        ax.set_title(column)   
        fig.savefig(f'./files/modeling_output/figures/box_{column}')
    
    data=duplicated_values(data)
    
    #Borramos los valores atipicos por encima de 56 años y los salarios menores a 10000 y mayores a 67000.

    data=data[(data['age']<56)&((data['salary']<67000)&(data['salary']>10000))]
    
    path = './files/datasets/intermediate/'

    data.to_csv(path+'preprocessed_data.csv', index=False)
    return data