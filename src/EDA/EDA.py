import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os
from sklearn.metrics import mean_squared_error

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    describe_result=data.describe()
    
    eda_path = './files/modeling_output/figures/'
    reports_path='./files/modeling_output/reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)
    
    # Exporting the file
    with open(reports_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open(reports_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
    
    fig , ax  = plt.subplots()
    ax.plot(data)
    fig.savefig(eda_path+'fig.png')
    
    df_gender=data.copy()
    df_gender['gender']=df_gender['gender'].astype('str')
    fig=px.histogram(df_gender['gender'])
    fig.write_image(eda_path+'fig.png', format='png', scale=2)
    fig.write_html(eda_path+'fig.html') 
    
    # Boxplots
    
    numeric=['age', 'salary', 'family_members', 'insurance_benefits']
    for column in numeric:
        fig,ax=plt.subplots()
        ax.boxplot(data[column])
        ax.set_title(column)   
        fig.savefig(f'./files/modeling_output/figures/box_{column}')
        
    
    # Graficos de correlacion
    g = sns.pairplot(data, kind='hist')
    g.fig.set_size_inches(12, 12)    
    g.savefig(eda_path+'figcorr.png')
    
    corr_df = data.corr(method="pearson")
    fig1,ax1=plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, ax=ax1, cmap="coolwarm", fmt=".2f")
    fig1.savefig(eda_path+'figcorr2.png')
    
    # Beneficios por edad
    fig2,ax2=plt.subplots()
    ax2.bar(data['age'],data['insurance_benefits'])
    ax2.set_title('Beneficios por edad')
    fig2.savefig(eda_path+'fig2.png')
    
    # Miembros de la familia por edad
    
    fig3=px.histogram(data,x='age',y='family_members',color='gender',barmode='group')
    fig3.write_image(eda_path+'fig3.png', format='png', scale=2)
    fig3.write_html(eda_path+'fig3.html') 
    
    # Salario por edad
    
    fig4=px.histogram(data,x='age',y='salary',color='gender',barmode='group')
    fig4.write_image(eda_path+'fig4.png', format='png', scale=2)
    fig4.write_html(eda_path+'fig4.html') 
    
    # Beneficios por miembros de la familia
    
    fig5=px.histogram(data,x='family_members',y='insurance_benefits',color='gender',barmode='group')
    fig5.write_image(eda_path+'fig5.png', format='png', scale=2)
    fig5.write_html(eda_path+'fig5.html') 