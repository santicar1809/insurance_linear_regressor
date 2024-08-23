import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler
from src.models.hyper_parameters import knn_models,knn,linear_regression_models
import joblib

def binaria(columna):
    if columna >0:
        return 1
    else:
        return 0

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = knn_models() 
    models_knn=knn()
    models_lr=linear_regression_models()
    data_esc=MaxAbsScaler().fit_transform(data)
    data_esc=pd.DataFrame(data_esc,columns=data.columns,index=data.index)
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    results = []

    # Iterating the models
    models_name = ['knn_euclidean','knn_manhattan']
    for model,i in zip(models,models_name):
        result= model_knn(data, model)
        results.append(result)
        result.to_csv(f'./files/modeling_output/reports/model_report_{i}.csv',index=True) 
    #Escaled data
    models_name = ['knn_euclidean_esc','knn_manhattan_esc']
    for model,i in zip(models,models_name):
        result= model_knn(data_esc, model)
        results.append(result)
        result.to_csv(f'./files/modeling_output/reports/model_report_{i}.csv',index=False) 
    
    results_knn=[]
    
    for model in models_knn:
        result_knn= knn_algorithm(data,model[1], model[2])
        best_estimator,best_score,score_val = result_knn
        results_knn.append([model[0],best_estimator,best_score, score_val])
        joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
    results_df = pd.DataFrame(results_knn, columns=['model','best_estimator','best_train_score','validation_score'])
    results_df.to_csv('./files/modeling_output/reports/model_report.csv',index=False)
    
    for model in models_lr:
        result_lr= linear_regression(data,model[1])
        score_val = result_lr
        results_knn.append([model[0],best_estimator,best_score, score_val])
        joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
    results_df = pd.DataFrame(results_knn, columns=['model','best_estimator','best_train_score','validation_score'])
    results_df.to_csv('./files/modeling_output/reports/model_report.csv',index=False)
    
    return results_df


def model_knn(data, model):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    features=data.drop(['insurance_benefits'],axis=1)
    result=model.get_knn(features)
    
    return result

def knn_algorithm(data, pipe,params_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    data['insurance_benefit_acepted']=data['insurance_benefits'].apply(binaria)
    #Separamos los datasets
    features=data.drop(['insurance_benefits','insurance_benefit_acepted'],axis=1)
    target=data['insurance_benefit_acepted']
    features_train,features_valid,target_train,target_valid=train_test_split(
        features,target,test_size=0.3,random_state=seed)
    
    gs = GridSearchCV(pipe, param_grid=params_grid, scoring='f1', cv=2)
    gs.fit(features_train, target_train)
    
    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    predict_valid = best_estimator.predict(features_valid)
    score_val=eval_classifier(target_valid,predict_valid)

    return best_estimator, best_score, score_val

def linear_regression(data, pipe):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    #Separamos los datasets
    features=data.drop(['insurance_benefits','insurance_benefit_acepted'],axis=1)
    features=features.to_numpy()
    target=data['insurance_benefits'].to_numpy()
    features_train,features_valid,target_train,target_valid=train_test_split(
        features,target,test_size=0.3,random_state=seed)
    
    pipe.fit(features_train, target_train)
    
    # Scores
    predict_valid = pipe.predict(features_valid)
    score_val=eval_regressor(target_valid,predict_valid)

    return score_val
    
def eval_classifier(y_true, y_pred):
    
    f1_score = metrics.f1_score(y_true, y_pred)
    # si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo    
    cm = metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print(f'Matriz de confusión: {cm}')
    return f1_score

def eval_regressor(y_true, y_pred):
    
    rmse = metrics.mean_squared_error(y_true, y_pred)**0.5
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = metrics.r2_score(y_true, y_pred)**0.5
    return rmse,r2_score

