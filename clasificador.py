from abc import ABC, abstractmethod
from balanceador import Balanceador
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import os
import joblib

class Clasificador(ABC):
    def __init__(self, balanceador:Balanceador, parametros:dict=None):
        self.balanceador = balanceador
        self.parametros = parametros
        self.datasetEnterno = {}
        self.datasetVal = {}
        self.modelos = {}

        self.datasetEnterno["E/I"] = (np.array(self.balanceador.train_EI["Embedding"].tolist()), self.balanceador.train_EI["MBTI"].tolist())
        self.datasetVal["E/I"] = (np.array(self.balanceador.val_EI["Embedding"].tolist()), self.balanceador.val_EI["MBTI"].tolist())

        self.datasetEnterno["S/N"] = (np.array(self.balanceador.train_SN["Embedding"].tolist()), self.balanceador.train_SN["MBTI"].tolist())
        self.datasetVal["S/N"] = (np.array(self.balanceador.val_SN["Embedding"].tolist()), self.balanceador.val_SN["MBTI"].tolist())

        self.datasetEnterno["T/F"] = (np.array(self.balanceador.train_TF["Embedding"].tolist()), self.balanceador.train_TF["MBTI"].tolist())
        self.datasetVal["T/F"] = (np.array(self.balanceador.val_TF["Embedding"].tolist()), self.balanceador.val_TF["MBTI"].tolist())

        self.datasetEnterno["J/P"] = (np.array(self.balanceador.train_JP["Embedding"].tolist()), self.balanceador.train_JP["MBTI"].tolist())
        self.datasetVal["J/P"] = (np.array(self.balanceador.val_JP["Embedding"].tolist()), self.balanceador.val_JP["MBTI"].tolist())
        
    
    @abstractmethod
    def busqueda_hiperparametros(self) -> dict:
        '''
        Funcion que mediante un diccionario de hiperparametros correspondientes al modelo clasificador concreto, predefinido por defecto, 
        devuelve la mejor combinacion para cada dimension.
        
        :return: Diccionario con la combinacion de hiperparametros en general
        '''
        pass
    
    @abstractmethod
    def entrenar_dimension(self, parametros:dict=None) -> None:
        '''
        Funcion que sirve para el entreno de un modelo de clasificacion de una dimension concreta.
        
        :param parametros: Diccionario que contiene el nombre de la dimension "dimension" y el conjunto de hiperparametros
        :return: None
        '''
        pass

    def entrenar_modelo(self, parametros:dict=None) -> None:
        '''
        Funcion que sirve para entrenar a todos los modelos de todas las dimensiones dado un diccionario de parametros, compartidos para todas las dimensiones
        
        :param parametros: Diccionario que contiene los hiperparametros para todos los modelos, en caso de no pasarse ninguno se usarán los hiperparametros por defecto de la clase
        :return: None
        '''
        if not parametros:
            parametros = self.parametros
        for dimension in ["E/I","S/N","T/F","J/P"]:
            self.entrenar_dimension(parametros|{"dimension":dimension})

    def guardar_dimension(self,modelo, dimension:str, carpeta:str, sufijo_archivo:str):
        '''
        Funcion que , dada una dimension, guarda el modelo en la carpeta correspondiente con el nombre dimension_sufijo
        
        :param modelo: Modelo clasificador a guardar
        :param dimension: Dimension a la que corresponde el modelo (ej: E/I)
        :param carpeta: Carpeta local en la que se guardará el modelo
        :param sufijo_archivo: Sufijo del archivo
        :return: None
        '''
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        joblib.dump(modelo, os.path.join(carpeta, f"{dimension}_{sufijo_archivo}"))

    def guardar_modelo(self, carpeta:str, sufijo_archivo:str):
        '''
        Funcion que guarda el modelo completo (todas las dimensiones) en la carpeta especificada, con el sufijo correspondiente
        
        :param carpeta: Carpeta local en la que se guardará el modelo
        :param sufijo_archivo: Sufijo del archivo
        :return: None
        '''
        for dimension in ["E/I","S/N","T/F","J/P"]:
            self.guardar_dimension(self.modelos[dimension], dimension, carpeta, sufijo_archivo)

    def getModelos(self) -> dict:
        '''
        Funcion que devuelve un diccionario con cada dimension y su modelo entrenado

        :return: Diccionario con cada dimension y su modelo
        '''
        return self.modelos
    
    def getParametros(self) -> dict:
        '''
        Funcion que devuelve un diccionario con todos los hiperparametros de la clase

        :return: Diccionario con todos los hiperparametros de la clase
        '''
        
        return self.parametros

    @abstractmethod
    def __str__(self):
        pass

class XGB(Clasificador):
    def __init__(self, balanceador, 
                 parametros = {'n_estimators': 352, 'max_depth': 8, 
                               'learning_rate': 0.041694500859353036, 'subsample': 0.6688508428769014, 
                               'colsample_bytree': 0.6246885857257476, 'gamma': 0.6844100651211652, 
                               'tree_method': 'hist',  
                                'device': 'cuda',       
                                'random_state': 42,
                                'n_jobs': 1
                                }
                                ):
        super().__init__(balanceador, parametros)
    
    def busqueda_hiperparametros(self):
        configuracion = {
            'tree_method': 'hist',  
            'device': 'cuda',       
            'random_state': 42,
            'n_jobs': 1
        }

        def objective(trial):
            
            hiperparametros = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),#Eligira un numero entre 100 y 800
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = XGBClassifier(**parametros)

            X_train, y_train = self.datasetEnterno["E/I"]

            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=1)
            return score.mean()

        print("[XGBoost][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=1)

        return estudio.best_params | configuracion
       
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]
        X_val, y_val = self.datasetVal[dimension]

        self.modelos[dimension] = XGBClassifier(**parametros).fit(X_train, y_train, eval_set=[(X_val, y_val)])
        print(f"\t[XGB][INFO] Modelo {dimension} entrenado.")
    
    def __str__(self):
        return "XGB"
    
class LSVC(Clasificador):
    def __init__(self, balanceador, 
                 parametros = {'loss': 'squared_hinge', 'C': 9.605448156429867,
                                'tol': 0.0004119653918147926, 'fit_intercept': False,
                                'random_state': 42,
                                'class_weight':'balanced',
                                'max_iter': 5000
                                }):
        super().__init__(balanceador, parametros)

    def busqueda_hiperparametros(self):
        configuracion = {
            'random_state': 42,
            'class_weight':'balanced',
            'max_iter': 5000,
        }

        def objective(trial):
            
            hiperparametros = {
                'penalty': "l2",
                'loss': trial.suggest_categorical("loss",['hinge', 'squared_hinge']),
                'C': trial.suggest_float("C", 1e-4, 10.0, log=True),
                'tol' : trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                'fit_intercept' : trial.suggest_categorical("fit_intercept", [True, False]),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LinearSVC(**parametros)
            X_train, y_train = self.datasetEnterno["E/I"]
            
            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[LinearSVC][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        return estudio.best_params | configuracion
    
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]

        self.modelos[dimension] = LinearSVC(**parametros).fit(X_train, y_train)
        print(f"\t[LinearSVC][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "LSVC"

class LR(Clasificador):
    def __init__(self, balanceador,
                  parametros = {'penalty': None, 'C': 9.732719859279047, 
                                'solver': 'lbfgs', 'tol': 0.00010564097578389837, 
                                'class_weight': 'balanced',
                                'random_state': 42,
                                'n_jobs': -1,
                                'max_iter': 5000}
                                ):
        super().__init__(balanceador, parametros)

    def busqueda_hiperparametros(self):
        configuracion = {
           'random_state': 42,
           'n_jobs': -1,
           'max_iter': 5000
        }

        def objective(trial):
            
            hiperparametros = {
                'penalty': trial.suggest_categorical("penalty",["l1","l2",None]),
                'C': trial.suggest_float("C", 1.0, 50.0, log=True),
                'solver': trial.suggest_categorical("solver", ['lbfgs', 'liblinear', 'newton-cg']),
                'tol' : trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                'class_weight': trial.suggest_categorical("class_weight", [None, 'balanced']),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LogisticRegression(**parametros)
            X_train, y_train = self.datasetEnterno["E/I"]
            
            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[LogisticRegression][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        return estudio.best_params | configuracion
    
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]

        self.modelos[dimension] = LogisticRegression(**parametros).fit(X_train, y_train)
        print(f"\t[LinearSVC][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "LR"

class KNC(Clasificador):
    def __init__(self, balanceador, 
                 parametros = {'n_neighbors': 2, 'weights': 'distance', 
                                'algorithm': 'brute', 'leaf_size': 14, 
                                'metric': 'cosine',
                                'n_jobs': -1 }
                                ):
        super().__init__(balanceador, parametros)
    
    def busqueda_hiperparametros(self):
        configuracion = {
           'n_jobs': -1 
        }

        def objective(trial):
            
            hiperparametros = {
                'n_neighbors': trial.suggest_int("n_neighbors", 1, 15),
                'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int("leaf_size", 5, 40),
                'metric': trial.suggest_categorical("metric", ['minkowski', 'euclidean', 'cosine']),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = KNeighborsClassifier(**parametros)
            X_train, y_train = self.datasetEnterno["E/I"]
            
            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[KNeighborsClassifier][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        return estudio.best_params | configuracion
    
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]

        self.modelos[dimension] = KNeighborsClassifier(**parametros).fit(X_train, y_train)
        print(f"\t[KNeighborsClassifier][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "KNC"

class DTC(Clasificador):
    def __init__(self, balanceador, 
                 parametros = {'criterion': 'gini', 'max_depth': 15, 
                               'min_samples_split': 9, 'min_samples_leaf': 4, 
                               'max_features': None, 'class_weight': None,
                                'random_state': 42}
                                ):
        super().__init__(balanceador, parametros)
    
    def busqueda_hiperparametros(self):
        configuracion = {
           'random_state': 42 
        }

        def objective(trial):
            
            hiperparametros = {
                'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
                'max_depth': trial.suggest_int("max_depth", 3, 20),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 50),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
                'max_features': trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                'class_weight': trial.suggest_categorical("class_weight", [None, "balanced"]),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = DecisionTreeClassifier(**parametros)
            X_train, y_train = self.datasetEnterno["E/I"]
            
            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[DecisionTreeClassifier][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        return estudio.best_params | configuracion
    
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]

        self.modelos[dimension] = DecisionTreeClassifier(**parametros).fit(X_train, y_train)
        print(f"\t[DecisionTreeClassifier][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "DTC"

class MLPC(Clasificador):
    def __init__(self, balanceador, 
                 parametros = {'hidden_layer_sizes': (256, 128, 64), 'activation': 'tanh', 
                               'solver': 'adam', 'learning_rate': 'constant', 
                               'learning_rate_init': 0.001,
                                'early_stopping': True,
                                'max_iter':6000,
                                'random_state':42}
                                ):
        super().__init__(balanceador, parametros)
    
    def busqueda_hiperparametros(self):
        configuracion = {
           'early_stopping': True,
           'max_iter':6000,
           'random_state':42 
        }

        def objective(trial):
            
            hiperparametros = {
                "hidden_layer_sizes" : trial.suggest_categorical("hidden_layer_sizes", [(256,256,128), (128,128,64), (256,128,64)]),
                "activation" : trial.suggest_categorical("activation", ['logistic', 'tanh', 'relu']),
                "solver" : trial.suggest_categorical("solver", ['sgd', 'adam']),
                "learning_rate" : trial.suggest_categorical("learning_rate", ['constant', 'invscaling']),
                "learning_rate_init" : trial.suggest_categorical("learning_rate_init", [0.001, 0.0005]),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = MLPClassifier(**parametros)
            X_train, y_train = self.datasetEnterno["E/I"]
            
            score = cross_val_score(modelo, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[MLPClassifier][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        return estudio.best_params | configuracion
    
    def entrenar_dimension(self, parametros = None):
        dimension = parametros["dimension"]
        del parametros["dimension"]

        X_train, y_train = self.datasetEnterno[dimension]

        self.modelos[dimension] = MLPClassifier(**parametros).fit(X_train, y_train)
        print(f"\t[MLPClassifier][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "MLPC"
    