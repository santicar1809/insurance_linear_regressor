import pandas as pd

def load_datasets():
    '''This function will upload the necessary datasets
    to perform the project.'''
    df=pd.read_csv('./files/datasets/input/insurance_us.csv')
    return df