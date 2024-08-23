import pandas as pd

def feature_engineer(data):
   
   data.reset_index(drop=True,inplace=True)
   
   return data