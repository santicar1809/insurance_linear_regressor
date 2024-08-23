from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


def get_P(X):
        while True:
                try:
                        rng = np.random.default_rng(seed=42)
                        P = rng.random(size=(X.shape[1], X.shape[1]))   
                        X @ P
                        return P
                        break
                        
                except ValueError:
                        print('Vuelve a intentar')
class LinearRegressionOfuscated():
    def __init__(self, obfusc=True):
        self.obfusc=obfusc
        self.P=None
    def obfuscate(self,X):
        if self.obfusc==True:
            X2= X @ self.P
            return X2
    def fit(self, train_features, train_target):
        if self.obfusc==True:
            self.P=get_P(train_features)
            train_features=train_features @ self.P
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = w[1:]
        self.w0 = w[0]
        
    def predict(self, test_features):
        if self.obfusc==True:
            test_features=test_features @ self.P
        #X = np.concatenate((np.ones((test_features.shape[0], 1)), test_features), axis=1)
        return test_features.dot(self.w) + self.w0
    

class NearestNeighbor():
    def __init__(self, metric='euclidean', k=5):
        self.k = k
        self.metric = metric
        
        if metric not in ['euclidean', 'manhattan']:
            raise ValueError("Métrica no soportada. Usa 'euclidean' o 'manhattan'.")
        
    def _compute_distance(self, vector1, vector2):
        if self.metric == 'euclidean':
            return distance.euclidean(vector1, vector2)
        elif self.metric == 'manhattan':
            return distance.cityblock(vector1, vector2)

    def nearest_neighbor_predict(self, new_features):
        distances = []
        
        for i in range(self.features.shape[0]):
            vector = self.features.iloc[i].values
            dist = self._compute_distance(new_features, vector)
            distances.append(dist)
        
        distances = np.array(distances)
        best_indices = distances.argsort()[:self.k]
        best_indices=np.delete(best_indices,0)
        return distances[best_indices], best_indices

    def get_knn(self, features, query_point_index=0):
        self.features = features
        query_point = self.features.iloc[query_point_index].values

        nbrs_distances, nbrs_indices = self.nearest_neighbor_predict(query_point)

        df_res = pd.concat([
            self.features.iloc[nbrs_indices],
            pd.DataFrame(nbrs_distances,index=nbrs_indices,columns=['distance'])
        ], axis=1)
    
        return df_res

class LinearRegression():
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0

def knn_models():
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''
    #Creamos los pipelines
    knn_euclidean=NearestNeighbor(metric='euclidean', k=5)
    knn_manhattan=NearestNeighbor(metric='manhattan', k=5)
    models=[knn_euclidean,knn_manhattan]
    
    return models

def knn():
    # Creamos los pipelines
    pipe_knn_esc = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('knn', KNeighborsClassifier())
    ])
    pipe_knn = Pipeline([
        ('knn', KNeighborsClassifier())
    ])
    params_grid ={
        'knn__n_neighbors':np.arange(1,16),
        'knn__metric': ['euclidean', 'cityblock']
    }
    
    dummie_pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('dummie',DummyClassifier(strategy="constant",constant=1))
    ])

    dummie_param_grid = {
            'dummie__strategy': ['constant'],  # Regularización
            'dummie__constant': [1]  # Fuerza de la regularización
            }
    dummie = ['dummie',dummie_pipeline,dummie_param_grid]
    
    knn=['knn',pipe_knn,params_grid]
    knn_esc=['knn_esc',pipe_knn_esc,params_grid]
    models=[dummie,knn,knn_esc]

    
    return models

def linear_regression_models():
    pipe_linear_regression_esc=Pipeline([
        ('scaler', MaxAbsScaler()),
        ('lr',LinearRegression())  
    ])
    
    pipe_linear_regression=Pipeline([
        ('lr',LinearRegression())
    ])
    
    lr_esc=['lr_esc',pipe_linear_regression_esc]
    lr=['lr',pipe_linear_regression]
    
    models=[lr_esc,lr]
    
    return models