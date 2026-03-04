import os
import pandas as pd
import numpy as np
import joblib
from openpyxl import load_workbook
import optuna

from balanceador import *
from procesador import Preprocesador

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class Pipeline:
    def __init__(self, nombre_modelo, balanceador:Balanceador=None):
        self.nombre_modelo = nombre_modelo
        self.nombre_balanceador = balanceador.__str__() if balanceador else None
        self.balanceador = balanceador

    def xgboost(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros = {'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 2000, 'subsample': 0.6, 'eval_metric': 'aucpr', 'early_stopping_rounds': 15}
                        
        X_train_EI = np.array(self.balanceador.train_bal_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_bal_EI["MBTI"].tolist()
        X_val_EI = np.array(self.balanceador.val_EI["Embedding"].tolist())
        y_val_EI = self.balanceador.val_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_bal_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_bal_SN["MBTI"].tolist()
        X_val_SN = np.array(self.balanceador.val_SN["Embedding"].tolist())
        y_val_SN = self.balanceador.val_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_bal_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_bal_TF["MBTI"].tolist()
        X_val_TF = np.array(self.balanceador.val_TF["Embedding"].tolist())
        y_val_TF = self.balanceador.val_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_bal_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_bal_JP["MBTI"].tolist()
        X_val_JP = np.array(self.balanceador.val_JP["Embedding"].tolist())
        y_val_JP = self.balanceador.val_JP["MBTI"].tolist()
        
        
        def objective(trial):
            hiperparametros = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),#Eligira un numero entre 100 y 800
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                
                'tree_method': 'hist',  
                'device': 'cuda',       
                'random_state': 42,
                'n_jobs': 1
            }

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = XGBClassifier(**hiperparametros)

            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para XGBoost...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_XGBoost.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros XGBoost - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")

        '''
        
        xgb_EI = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric=parametros['eval_metric'], early_stopping_rounds=parametros['early_stopping_rounds'], verbosity=0)
        xgb_EI.fit(X_train_EI, y_train_EI, eval_set=[(X_val_EI, y_val_EI)])
        print("[INFO] Modelo E/I entrenado.")
        xgb_SN = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric=parametros['eval_metric'], early_stopping_rounds=parametros['early_stopping_rounds'], verbosity=0)
        xgb_SN.fit(X_train_SN, y_train_SN, eval_set=[(X_val_SN, y_val_SN)])
        print("[INFO] Modelo S/N entrenado.")
        xgb_TF = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric=parametros['eval_metric'], early_stopping_rounds=parametros['early_stopping_rounds'], verbosity=0)
        xgb_TF.fit(X_train_TF, y_train_TF, eval_set=[(X_val_TF, y_val_TF)])
        print("[INFO] Modelo T/F entrenado.")
        xgb_JP = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric=parametros['eval_metric'], early_stopping_rounds=parametros['early_stopping_rounds'], verbosity=0)
        xgb_JP.fit(X_train_JP, y_train_JP, eval_set=[(X_val_JP, y_val_JP)])
        print("[INFO] Modelo J/P entrenado.")

        self.modelos = { 
            'EI': xgb_EI,
            'SN': xgb_SN,
            'TF': xgb_TF,
            'JP': xgb_JP
        }
        

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="XGBoost")
        os.makedirs("modelos_XGB", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_XGB", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        '''       
      
    def LinearSVM(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros = {'C': 10, 'class_weight': None, 'loss': 'squared_hinge', 'penalty': 'l2', 'max_iter':10000}

        X_train_EI = np.array(self.balanceador.train_bal_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_bal_EI["MBTI"].tolist()
        X_train_SN = np.array(self.balanceador.train_bal_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_bal_SN["MBTI"].tolist()
        X_train_TF = np.array(self.balanceador.train_bal_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_bal_TF["MBTI"].tolist()
        X_train_JP = np.array(self.balanceador.train_bal_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_bal_JP["MBTI"].tolist()
        
        
        def objective(trial):
            hiperparametros = {
                'penalty': "l2",
                'loss': trial.suggest_categorical("loss",['hinge', 'squared_hinge']),
                'C': trial.suggest_float("C", 1e-4, 10.0, log=True),
                'tol' : trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                'fit_intercept' : trial.suggest_categorical("fit_intercept", [True, False]),
                
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': trial.suggest_categorical("class_weight", [None, 'balanced']),
                'max_iter': 5000,
                'dual': False
            }

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LinearSVC(**hiperparametros)

            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para LinearSVC...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_LinearSVC.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros LinearSVC - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        
        '''

        lSVM_EI = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'],class_weight=parametros["class_weight"], random_state=42, dual=False).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        lSVM_SN = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'],class_weight=parametros["class_weight"], random_state=42,  dual=False).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        lSVM_TF = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'],class_weight=parametros["class_weight"], random_state=42,  dual=False).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        lSVM_JP = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'],class_weight=parametros["class_weight"], random_state=42,  dual=False).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")

        self.modelos = { 
            'EI': lSVM_EI,
            'SN': lSVM_SN,
            'TF': lSVM_TF,
            'JP': lSVM_JP
        }
        

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="LinearSVM")
        os.makedirs("modelos_LinearSVM", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LinearSVM", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        '''

    def RL(self, nombre_Archivo, parametros = None):

        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros={'C': 10, 'class_weight': None, 'penalty': 'l2', 'solver': 'lbfgs'}

        X_train_EI = np.array(self.balanceador.train_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_JP["MBTI"].tolist()
        
        
        def objective(trial):
            hiperparametros = {
                'penalty': trial.suggest_categorical("penalty",["l1","l2",None]),
                'C': trial.suggest_float("C", 1.0, 50.0, log=True),
                'solver': trial.suggest_categorical("solver", ['lbfgs', 'liblinear', 'newton-cg']),
                'tol' : trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': trial.suggest_categorical("class_weight", [None, 'balanced']),
                'max_iter': 5000
            }

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = LogisticRegression(**hiperparametros)

            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para LogisticRegresion...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_LogisticRegresion.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros LogisticRegresion - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")

        '''
        
        lr_EI = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], class_weight=parametros["class_weight"], max_iter=5000, random_state=42, verbose=0).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        lr_SN = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], class_weight=parametros["class_weight"], max_iter=5000, random_state=42, verbose=0).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        lr_TF = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], class_weight=parametros["class_weight"], max_iter=5000, random_state=42, verbose=0).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        lr_JP = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], class_weight=parametros["class_weight"], max_iter=5000, random_state=42, verbose=0).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        #Diccionario para almacenar los modelos entrenados
        self.modelos = { 
            'EI': lr_EI,
            'SN': lr_SN,
            'TF': lr_TF,
            'JP': lr_JP
        }

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="Regresión Logística")
        os.makedirs("modelos_LR", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        '''
            
    def KNC(self, nombre_Archivo, parametros = None):
        if not parametros:
            parametros = {
                "n_neighbors": 3,
                "weights": 'distance',
                "algorithm": 'ball_tree',
                "leaf_size": 10
            }

        X_train_EI = np.array(self.balanceador.train_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_JP["MBTI"].tolist()
        
        def objective(trial):
            hiperparametros = {
                'n_neighbors': trial.suggest_int("n_neighbors", 1, 15),
                'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int("leaf_size", 5, 40),
                'metric': trial.suggest_categorical("metric", ['minkowski', 'euclidean', 'cosine']),
                'n_jobs': -1 
            }
            modelo = KNeighborsClassifier(**hiperparametros)

            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para KNeighborsClassifier...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_KNC.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros KNC - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")


        '''
        knc_EI = KNeighborsClassifier(n_neighbors=parametros["n_neighbors"], weights=parametros["weights"], algorithm=parametros["algorithm"], leaf_size=parametros["leaf_size"], n_jobs=-1).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        knc_SN = KNeighborsClassifier(n_neighbors=parametros["n_neighbors"], weights=parametros["weights"], algorithm=parametros["algorithm"], leaf_size=parametros["leaf_size"], n_jobs=-1).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        knc_TF = KNeighborsClassifier(n_neighbors=parametros["n_neighbors"], weights=parametros["weights"], algorithm=parametros["algorithm"], leaf_size=parametros["leaf_size"], n_jobs=-1).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        knc_JP = KNeighborsClassifier(n_neighbors=parametros["n_neighbors"], weights=parametros["weights"], algorithm=parametros["algorithm"], leaf_size=parametros["leaf_size"], n_jobs=-1).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        #Diccionario para almacenar los modelos entrenados
        self.modelos = { 
            'EI': knc_EI,
            'SN': knc_SN,
            'TF': knc_TF,
            'JP': knc_JP
        }

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="KNeighboursClassifier")
        os.makedirs("modelos_LR", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        '''
            
    def DTC(self, nombre_Archivo, parametros = None):
        if not parametros:
            parametros = {
                "criterion": ['gini', 'entropy', 'log_loss'],

            }

        X_train_EI = np.array(self.balanceador.train_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_JP["MBTI"].tolist()
        
        def objective(trial):
            hiperparametros = {
                'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
                'max_depth': trial.suggest_int("max_depth", 3, 20),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 50),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
                'max_features': trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                'class_weight': trial.suggest_categorical("class_weight", [None, "balanced"]),
                'random_state': 42
            }

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = DecisionTreeClassifier(**hiperparametros)

            # Ojo: DecisionTree no usa n_jobs internamente en el modelo, pero cross_val_score sí
            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1)
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para DTC...")
        estudio = optuna.create_study(direction="maximize")
        
        # El árbol simple es muy rápido de entrenar, 30 iteraciones se harán en nada
        estudio.optimize(objective, n_trials=30, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_DTC.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros DTC - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")

        '''
        dtc_EI = DecisionTreeClassifier().fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        dtc_SN = DecisionTreeClassifier().fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        dtc_TF = DecisionTreeClassifier().fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        dtc_JP = DecisionTreeClassifier().fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        #Diccionario para almacenar los modelos entrenados
        self.modelos = { 
            'EI': dtc_EI,
            'SN': dtc_SN,
            'TF': dtc_TF,
            'JP': dtc_JP
        }

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="DecisionTreeClassifier")
        os.makedirs("modelos_LR", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        '''
        
    def MLP(self, nombre_Archivo, parametros = None):
        if not parametros:
            parametros = {'hidden_layer_sizes': (256, 128, 64), 'activation': 'tanh', 'solver': 'adam', 'learning_rate': 'constant', 'learning_rate_init': 0.001}

        X_train_EI = np.array(self.balanceador.train_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_JP["MBTI"].tolist()
        
        '''
        # Funcion objetivo para optuna
        def objective(trial):
            # Optuna elige combinaciones inteligentemente en cada "intento"
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(256,256,128), (128,128,64), (256,128,64)])
            activation = trial.suggest_categorical("activation", ['logistic', 'tanh', 'relu'])
            solver = trial.suggest_categorical("solver", ['sgd', 'adam'])
            learning_rate = trial.suggest_categorical("learning_rate", ['constant', 'invscaling'])
            learning_rate_init = trial.suggest_categorical("learning_rate_init", [0.001, 0.0005])

            # Parche de seguridad: 'lbfgs' crashea si le pones early_stopping=True
            usa_early_stopping = True if solver in ['sgd', 'adam'] else False

            # Instanciamos el modelo con los parámetros sugeridos por Optuna
            modelo = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                early_stopping=usa_early_stopping,
                max_iter=6000,
                random_state=42
            )

            score = cross_val_score(modelo, X_train_EI, y_train_EI, cv=3, scoring="f1_macro", n_jobs=-1) #Usamos los datasets de entrenos de EI ya que es el que mas desbalanceado está
            return score.mean()

        print("[EJECUCION] Iniciando Estudio Optuna para MLP...")
        estudio = optuna.create_study(direction="maximize")
        
        estudio.optimize(objective, n_trials=20, gc_after_trial=True, n_jobs=-1)

        hiperparametros_info = {
            'E/I': estudio.best_params
        }

        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_MLP.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros MLP - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        '''
        
        mlp_EI = MLPClassifier(hidden_layer_sizes=parametros["hidden_layer_sizes"], activation=parametros["activation"], solver=parametros["solver"], learning_rate=parametros["learning_rate"], learning_rate_init=parametros["learning_rate_init"], max_iter=6000, random_state=42).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        mlp_SN = MLPClassifier(hidden_layer_sizes=parametros["hidden_layer_sizes"], activation=parametros["activation"], solver=parametros["solver"], learning_rate=parametros["learning_rate"], learning_rate_init=parametros["learning_rate_init"], max_iter=6000, random_state=42).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        mlp_TF = MLPClassifier(hidden_layer_sizes=parametros["hidden_layer_sizes"], activation=parametros["activation"], solver=parametros["solver"], learning_rate=parametros["learning_rate"], learning_rate_init=parametros["learning_rate_init"], max_iter=6000, random_state=42).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        mlp_JP = MLPClassifier(hidden_layer_sizes=parametros["hidden_layer_sizes"], activation=parametros["activation"], solver=parametros["solver"], learning_rate=parametros["learning_rate"], learning_rate_init=parametros["learning_rate_init"], max_iter=6000, random_state=42).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        #Diccionario para almacenar los modelos entrenados
        self.modelos = { 
            'EI': mlp_EI,
            'SN': mlp_SN,
            'TF': mlp_TF,
            'JP': mlp_JP
        }

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="MLP")
        os.makedirs("modelos_LR", exist_ok=True)
        for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")
        
    def obtener_metricas(self, modelo, df_test, nombre_modelo):
        X_test = np.array(df_test["Embedding"].tolist(), dtype=np.float32)
        y_test = df_test["MBTI"].tolist()
        y_pred = modelo.predict(X_test)
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "Modelos": nombre_modelo,
            "Precision": f"0: {precision[0]:.2f}\n1: {precision[1]:.2f}",
            "Recall": f"0: {recall[0]:.2f}\n1: {recall[1]:.2f}",
            "F1-Score": f"0: {f1[0]:.2f}\n1: {f1[1]:.2f}",
            "Support": f"0: {support[0]}\n1: {support[1]}",
            "Accuracy": round(acc, 2),
            "Matriz Confusion": f"[[{cm[0][0]} {cm[0][1]}]\n [{cm[1][0]} {cm[1][1]}]]"
        }

    def guardar_resultados(self, nombre_Archivo, carpeta="resultados", metodo_balanceo=None, parametros_str=None, modelo_clasificacion=None):
        print(f"[INFO] Exportando resultados a Excel (Pestaña: {metodo_balanceo})...")
    
        filas = []
        filas.append(self.obtener_metricas(self.modelos['EI'], self.balanceador.test_EI, f"{modelo_clasificacion} E/I"))
        filas.append(self.obtener_metricas(self.modelos['SN'], self.balanceador.test_SN, f"{modelo_clasificacion} S/N"))
        filas.append(self.obtener_metricas(self.modelos['TF'], self.balanceador.test_TF, f"{modelo_clasificacion} T/F"))
        filas.append(self.obtener_metricas(self.modelos['JP'], self.balanceador.test_JP, f"{modelo_clasificacion} J/P"))
        
        df_resultados = pd.DataFrame(filas)
        
        archivo_excel = os.path.join(carpeta, nombre_Archivo)
        
        modo = 'a' if os.path.exists(archivo_excel) else 'w'
        motor = 'openpyxl'
        fila_inicio = 0
        escribir_cabecera = True
        if modo == 'a':
            try:
                wb = load_workbook(archivo_excel)
                if metodo_balanceo in wb.sheetnames:
                    # La pestaña existe. Buscamos la última fila escrita y le sumamos 2 de margen
                    fila_inicio = wb[metodo_balanceo].max_row + 2
                    escribir_cabecera = False # Ya hay cabeceras arriba, no las repetimos
            except Exception as e:
                print(f"Aviso al leer el Excel: {e}")    


        with pd.ExcelWriter(archivo_excel, engine=motor, mode=modo, if_sheet_exists='overlay' if modo == 'a' else None) as writer:
            df_params = pd.DataFrame([["Parametros", parametros_str]])
            df_params.to_excel(writer, sheet_name=metodo_balanceo, startrow=fila_inicio, index=False, header=False)
            df_resultados.to_excel(writer, sheet_name=metodo_balanceo, startrow=fila_inicio + 2, index=False, header=escribir_cabecera)

        print(f"[INFO] Resultados guardados en {archivo_excel}")

    def guardar_modelo(self, carpeta, modelo, nombre_archivo):
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        joblib.dump(modelo, os.path.join(carpeta, nombre_archivo))

    #Pipeline que sirve para comprobar la mejor combinacion de hiperparametros para un modelo de clasificacion concreto
    def ejecutar_pipeline_entreno(self, preprocesar=True, parametros=None, algotitmo=None, nombre_Archivo="ResultadosRegresion.xlsx"):
        print("[EJECUCION] Ejecutando pipeline completo ...")

        #Paso 1: Preprocesamiento del dataset (opcional, se puede saltar si ya se ha preprocesado antes y se tiene el parquet guardado)

        if preprocesar:
            archivo = os.path.join("datasets","MBTI_sinProcesar.csv")
            procesador = Preprocesador(archivo, self.nombre_modelo)
            procesador.procesar_dataset()

        print("[EJECUCION] Dividiendo y balanceando dataset ...")
        #Paso 2: División y balanceo del dataset
        self.balanceador.dividir_balancear()

        #Paso 3: Dependiendo del modelo, entrenamiento y evaluación y exportación de resultados
        match algotitmo:
            case "RL": 
                print(f"[EJECUCION] Entrenando modelo de Regresión Logística con {self.nombre_balanceador} ...")
                self.RL(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "XGBoost":
                print(f"[EJECUCION] Entrenando modelo XGBoost con {self.nombre_balanceador} ...")
                self.xgboost(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "LinearSVM":
                print(f"[EJECUCION] Entrenando modelo LinearSVM con {self.nombre_balanceador} ...")
                self.LinearSVM(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "KNC":
                print(f"[EJECUCION] Entrenando modelo KNeighborsClassifier con {self.nombre_balanceador}...")
                self.KNC(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "DTC":
                print(f"[EJECUCION] Entrenando modelo DecisionTreeClassifier con {self.nombre_balanceador}...")
                self.DTC(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "MLP":
                print(f"[EJECUCION] Entrenando modelo MultiLayerPerceptron con {self.nombre_balanceador}...")
                self.MLP(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case _ : 
                print("[ERROR] Modelo de clasificación no reconocido")

        print("[EJECUCION] Fin pipeline completo ...")

#Funcion que sirve para sacar las metricas de entrenamiento para un modelo de embedding concreto
def pipeline_modelo_entreno(modelo:str):
    nombre_dataset = f"{modelo.replace('/', '_')}_dataset.parquet"
    balSMOTE = BalanceadorSMOTE(nombre_dataset=nombre_dataset)
    #balBORSMOTE = BalanceadorBorderlineSMOTE(nombre_dataset=nombre_dataset)
    #balADASYN = BalanceadorADASYN(nombre_dataset=nombre_dataset)
    #balENN = BalanceadorENN(nombre_dataset=nombre_dataset)
    #balAKNN = BalanceadorAKNN(nombre_dataset=nombre_dataset)
    
    pipelineSMOTE = Pipeline(nombre_modelo=modelo, balanceador=balSMOTE)
    #pipelineBORSMOTE = Pipeline(nombre_modelo=modelo, balanceador=balBORSMOTE)
    #pipelineADASYN = Pipeline(nombre_modelo=modelo, balanceador=balADASYN)
    #pipelineENN = Pipeline(nombre_modelo=modelo, balanceador=balENN)
    #pipelineAKNN = Pipeline(nombre_modelo=modelo, balanceador=balAKNN)

    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="RL", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="XGBoost", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="LinearSVM", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="MLP", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="KNC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="DTC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="MLP", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    

def ejecutar_pipelines(pipelines:list, preprocesar=False, algoritmo=None, nombre_Archivo="Resultados.xlsx"):
    for pipeline in pipelines:
        pipeline.ejecutar_pipeline_entreno(preprocesar=preprocesar, algotitmo=algoritmo, nombre_Archivo=nombre_Archivo)

if __name__ == "__main__":
   
    #EJECUCION PIPELINE ROBERTA BASE
    print("="*50)
    print("[INICIO] Ejecución pipeline con Roberta Base ...")
    nombre_modelo = "FacebookAI/roberta-base"
    pipeline_modelo_entreno(nombre_modelo)
    print("[FIN] Ejecución pipeline con Roberta Base ...")
    print("="*50)
    '''
    #EJECUCION PIPELINE XLM-ROBERTA-BASE
    print("="*50)
    print("[INICIO] Ejecución pipeline con XLM Roberta Base ...")
    nombre_modelo = "FacebookAI/xlm-roberta-base"
    pipeline_modelo_entreno(nombre_modelo)
    print("[FIN] Ejecución pipeline con XLM Roberta Base ...")
    print("="*50)

    #EJECUCION PIPELINE XLM-ROBERTA-LARGE
    print("="*50)
    print("[INICIO] Ejecución pipeline con XLM Roberta Large ...")
    nombre_modelo = "FacebookAI/xlm-roberta-large"
    pipeline_modelo_entreno(nombre_modelo)
    print("[FIN] Ejecución pipeline con XLM Roberta Large ...")
    print("="*50)
    '''
    
