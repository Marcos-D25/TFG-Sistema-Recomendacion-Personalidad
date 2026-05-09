from abc import ABC, abstractmethod
from balanceador import Balanceador
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
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

        self.datasetEnterno["E-I"] = (np.array(self.balanceador.train_EI["Embedding"].tolist()), self.balanceador.train_EI["MBTI"].tolist())
        self.datasetVal["E-I"] = (np.array(self.balanceador.val_EI["Embedding"].tolist()), self.balanceador.val_EI["MBTI"].tolist())

        self.datasetEnterno["S-N"] = (np.array(self.balanceador.train_SN["Embedding"].tolist()), self.balanceador.train_SN["MBTI"].tolist())
        self.datasetVal["S-N"] = (np.array(self.balanceador.val_SN["Embedding"].tolist()), self.balanceador.val_SN["MBTI"].tolist())

        self.datasetEnterno["T-F"] = (np.array(self.balanceador.train_TF["Embedding"].tolist()), self.balanceador.train_TF["MBTI"].tolist())
        self.datasetVal["T-F"] = (np.array(self.balanceador.val_TF["Embedding"].tolist()), self.balanceador.val_TF["MBTI"].tolist())

        self.datasetEnterno["J-P"] = (np.array(self.balanceador.train_JP["Embedding"].tolist()), self.balanceador.train_JP["MBTI"].tolist())
        self.datasetVal["J-P"] = (np.array(self.balanceador.val_JP["Embedding"].tolist()), self.balanceador.val_JP["MBTI"].tolist())
        
    
    @abstractmethod
    def busqueda_hiperparametros(self,dimension:str) -> dict:
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
        for dimension in ["E-I","S-N","T-F","J-P"]:
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
        for dimension in ["E-I","S-N","T-F","J-P"]:
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
                 parametros = {'n_estimators': 650, 'max_depth': 9, 'learning_rate': 0.010288662924278852,
                                'min_child_weight': 10, 'subsample': 0.829092618497977, 'colsample_bytree': 0.5881832414574534, 
                                'gamma': 4.140467354786688, 'reg_alpha': 0.054217127083094424, 'reg_lambda': 0.13120231425895848, 
                                'grow_policy': 'depthwise', 'multi_strategy': 'multi_output_tree', 'objective': 'binary:hinge',
                                  'tree_method': 'hist', 
                                  'device': 'cuda', 
                                  'random_state': 42, 
                                  'n_jobs': 1, 
                                  'verbosity': 0}
                                ):
        super().__init__(balanceador, parametros)
    
    def busqueda_hiperparametros(self, dimension:str):
        configuracion = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',  
            'device': 'cuda',       
            'random_state': 42,
            'verbosity':0,
            "n_jobs": 8
        }

        def objective(trial):
            
            hiperparametros = {
                "scale_pos_weight": trial.suggest_float("scale_pos_weight",1,10, step=0.01),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 8),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = XGBClassifier(**parametros)

            X_train, y_train = self.datasetEnterno[dimension] #Depende de la dimension

            score = cross_val_score(modelo, X_train, y_train, cv=4, scoring="f1_macro", n_jobs=1)
            return score.mean()

        print("[XGBoost][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=50, gc_after_trial=True, n_jobs=6)

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
                                #'class_weight':'balanced',
                                'max_iter': 5000
                                }):
        super().__init__(balanceador, parametros)

    def busqueda_hiperparametros(self,dimension:str):
        configuracion = {
            'random_state': 42,
            'max_iter': 15000, # Los SVM tardan más en converger con datos muy dimensionales
            'dual': "auto"     # Auto-decisión optima en versiones nuevas de Sklearn
        }

        def objective(trial):
            
            # Determinamos primero la penalización para evitar incompatibilidades
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            
            hiperparametros = {
                'penalty': penalty,
                'C': trial.suggest_float("C", 1e-4, 100.0, log=True),
                'tol': trial.suggest_float("tol", 1e-5, 1e-1, log=True),
                #'class_weight': trial.suggest_categorical("class_weight", [None, 'balanced']),
                'fit_intercept': trial.suggest_categorical("fit_intercept", [True, False]),
            } 
            
            # Si penalty es L1, solo soporta squared_hinge
            if penalty == "l1":
                hiperparametros['loss'] = "squared_hinge"
                hiperparametros['dual'] = False # Requisito matemático estricto de Sklearn
            else:
                hiperparametros['loss'] = trial.suggest_categorical("loss", ['hinge', 'squared_hinge'])

            # Combinamos con config base pisando 'dual' si es necesario
            parametros = configuracion.copy()
            parametros.update(hiperparametros)

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LinearSVC(**parametros)
            X_train, y_train = self.datasetEnterno[dimension]
            
            score = cross_val_score(modelo, X_train, y_train, cv=4, scoring="f1_macro", n_jobs=1)
            return score.mean()

        print("[LinearSVC][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=50, gc_after_trial=True, n_jobs=6)
        # Re-construimos el diccionario final
        best_params = estudio.best_params.copy()
        if best_params.get("penalty") == "l1":
            best_params["loss"] = "squared_hinge"
            best_params["dual"] = False

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
                                #'class_weight': 'balanced',
                                'random_state': 42,
                                'n_jobs': -1,
                                'max_iter': 5000}
                                ):
        super().__init__(balanceador, parametros)

    def busqueda_hiperparametros(self,dimension:str):
        configuracion = {
           'random_state': 42,
           'n_jobs': -1,
           'max_iter': 10000, # Aumentado para asegurar convergencia con L1/Saga
           'solver': 'saga'   # Saga soporta l1, l2 y elasticnet
        }

        def objective(trial):
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
            
            hiperparametros = {
                'penalty': penalty,
                'C': trial.suggest_float("C", 1e-4, 100.0, log=True), # Inverso de lambda/alpha en XGBoost
                'tol': trial.suggest_float("tol", 1e-5, 1e-1, log=True),
                #'class_weight': trial.suggest_categorical("class_weight", [None, 'balanced']), # Equivalente a scale_pos_weight
            } 
            
            # Si usamos ElasticNet, Optuna debe buscar el ratio exacto entre L1 y L2
            if penalty == "elasticnet":
                hiperparametros['l1_ratio'] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            
            # Si penalty es None, C no tiene efecto y tira error en sklearn
            if penalty is None:
                del hiperparametros['C']

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LogisticRegression(**parametros)
            X_train, y_train = self.datasetEnterno[dimension]
            
            score = cross_val_score(modelo, X_train, y_train, cv=4, scoring="f1_macro", n_jobs=1)
            return score.mean()

        print("[LogisticRegression][EJECUCION] Iniciando Estudio Optuna...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=50, gc_after_trial=True, n_jobs=6)

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
    
    def busqueda_hiperparametros(self,dimension:str):
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
            X_train, y_train = self.datasetEnterno[dimension]
            
            score = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
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
    
    def busqueda_hiperparametros(self,dimension:str):
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
                #'class_weight': trial.suggest_categorical("class_weight", [None, "balanced"]),
            } 

            parametros = hiperparametros | configuracion

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = DecisionTreeClassifier(**parametros)
            X_train, y_train = self.datasetEnterno[dimension]
            
            score = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
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

# --- DEFINICIÓN DE LA RED EN PYTORCH ---
class MBTINet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2):
        super(MBTINet, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, x):
        return self.red(x)

class MLPC(Clasificador):
    def __init__(self, balanceador, parametros=None):
        super().__init__(balanceador, parametros)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for dim in ["E-I", "S-N", "T-F", "J-P"]:
            X, y = self.datasetEnterno[dim]
            self.datasetEnterno[dim] = (X.astype(np.float32), np.array(y).astype(np.int64))

    def busqueda_hiperparametros(self, dimension: str):
        def objective(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            
            net = NeuralNetClassifier(
                module=MBTINet,
                module__input_dim=768,
                module__hidden_dim=hidden_dim,
                criterion=nn.CrossEntropyLoss,
                optimizer=optim.Adam,
                lr=lr,
                max_epochs=20,
                batch_size=batch_size,
                device=self.device, 
                train_split=None,  
                verbose=0          
            )


            X_train, y_train = self.datasetEnterno[dimension]

            score = cross_val_score(net, X_train, y_train, cv=4, scoring="f1_macro", n_jobs=1)
            return score.mean()

        print(f"[MLPC-GPU][EJECUCION] Iniciando Optuna en {self.device.upper()} para {dimension}...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=50, gc_after_trial=True, n_jobs=1)

        return estudio.best_params

    def entrenar_dimension(self, parametros=None):
        dimension = parametros["dimension"]
        
        h_dim = parametros.get("hidden_dim", 256)
        lr = parametros.get("lr", 0.001)
        b_size = parametros.get("batch_size", 256)

        X_train, y_train = self.datasetEnterno[dimension]

        net = NeuralNetClassifier(
            module=MBTINet,
            module__input_dim=768,
            module__hidden_dim=h_dim,
            lr=lr,
            batch_size=b_size,
            device=self.device,
            max_epochs=50,
            verbose=0
        )
        
        self.modelos[dimension] = net.fit(X_train, y_train)
        print(f"\t[MLPC][INFO] Modelo {dimension} entrenado.")

    def __str__(self):
        return "MLPC"
    