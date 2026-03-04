import os
import pandas as pd
import numpy as np
from balanceador import *
from procesador import Preprocesador
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from openpyxl import load_workbook
from xgboost import XGBClassifier
from torch import nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch.optim import Adam

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
        
        '''
        hiperparametros = {
            'n_estimators': [2000],#Aunque sea un numero muy grande, el early stopping hace que no se llegue a ese numero si el modelo no mejora en 15 iteraciones seguidas
            'max_depth': [5,6,7],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.3,0.4,0.6],
        }

        xgb = XGBClassifier(device = "cuda", tree_method="hist",random_state=42, eval_metric="aucpr", early_stopping_rounds=15)
        #(Maximizar)Usamos aucpr porque es una metrica que se centra en el rendimiento del modelo en la clase minoritaria, lo cual es crucial en nuestro caso de clasificación binaria con clases desbalanceadas.
        #(Maximizar)F1_macro realiza la media aritmetica de calcular el F1 para cada clase y luego promediarlo.
        
        xgb_EI = GridSearchCV(xgb, hiperparametros, scoring="f1_macro", cv=3, n_jobs=1, verbose=1).fit(X_train_EI, y_train_EI, eval_set=[(X_val_EI, y_val_EI)])
        #xgb_SN = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_SN, y_train_SN)
        #xgb_TF = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_TF, y_train_TF)
        #xgb_JP = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_JP, y_train_JP)
        
        hiperparametros_info = {
            'E/I': xgb_EI.best_params_,
            #'S/N': xgb_SN.best_params_,
            #'T/F': xgb_TF.best_params_,
            #'J/P': xgb_JP.best_params_
        }
        
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
        
        '''
        hiperparametros = {
            'penalty': ['l1','l2'],
            'loss': ['squared_hinge'],
            'C': [0.01, 0.1, 1, 10],
            'class_weight' : ["balanced"]
        }

        lSVM = LinearSVC(random_state=42, dual=False, max_iter=10000)

        lSVM_EI = GridSearchCV(lSVM, hiperparametros, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1).fit(X_train_EI, y_train_EI)
        #lSVM_SN = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_SN, y_train_SN)
        #lSVM_TF = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_TF, y_train_TF)
        #lSVM_JP = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_JP, y_train_JP)

        hiperparametros_info = {
            'E/I': lSVM_EI.best_params_,
            #'S/N': lSVM_SN.best_params_,
            #'T/F': lSVM_TF.best_params_,
            #'J/P': lSVM_JP.best_params_
        }
        
        nombre_archivo = "hiperparametros_LinearSVM.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros LinearSVM - {self.nombre_modelo}\n")
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
        
        '''
        hiperparametros = {
            'C': [0.1, 1, 10, 50, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs','liblinear', 'saga'],
            'class_weight': ['balanced']
        }
        lr_EI = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42), hiperparametros, scoring="f1_macro", cv=3, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)
        #lr_SN = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_SN, y_train_SN)
        #lr_TF = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_TF, y_train_TF)
        #lr_JP = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_JP, y_train_JP)
        
        hiperparametros_info = {
            'E/I': lr_EI.best_params_,
            #'S/N': lr_SN.best_params_,
            #'T/F': lr_TF.best_params_,
            #'J/P': lr_JP.best_params_
        }
        print(f"[INFO] Mejor combinación de hiperparámetros: {hiperparametros_info}")
        nombre_archivo = "hiperparametros_LogisticRegression.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros Regresión Logística - {self.nombre_modelo}\n")
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
        '''
        hiperparametros = {
            "n_neighbors": [2,3,5],
            "weights": [ 'distance'],
            "algorithm": ['ball_tree'],
            "leaf_size": [5,10,15],
        }

        knc_EI = GridSearchCV(KNeighborsClassifier(), hiperparametros, scoring="f1_macro", cv=3, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)

        hiperparametros_info = {
            'E/I': knc_EI.best_params_
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
        '''
        hiperparametros = {
            "n_neighbors": [2,3,5],
            "weights": [ 'distance'],
            "algorithm": ['ball_tree'],
            "leaf_size": [5,10,15],
        }

        knc_EI = GridSearchCV(KNeighborsClassifier(), hiperparametros, scoring="f1_macro", cv=3, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)

        hiperparametros_info = {
            'E/I': knc_EI.best_params_
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

    def MLP(self, nombre_Archivo, parametros = None):
        if not parametros:
            parametros = {
            }

        X_train_EI = np.array(self.balanceador.train_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_EI["MBTI"].tolist()

        X_train_SN = np.array(self.balanceador.train_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_SN["MBTI"].tolist()
        
        X_train_TF = np.array(self.balanceador.train_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_TF["MBTI"].tolist()
        
        X_train_JP = np.array(self.balanceador.train_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_JP["MBTI"].tolist()
        '''
        hiperparametros = {
            "n_neighbors": [2,3,5],
            "weights": [ 'distance'],
            "algorithm": ['ball_tree'],
            "leaf_size": [5,10,15],
        }

        knc_EI = GridSearchCV(KNeighborsClassifier(), hiperparametros, scoring="f1_macro", cv=3, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)

        hiperparametros_info = {
            'E/I': knc_EI.best_params_
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
        mlp_EI = MLPClassifier().fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        mlp_SN = MLPClassifier().fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        mlp_TF = MLPClassifier().fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        mlp_JP = MLPClassifier().fit(X_train_JP, y_train_JP)
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

    #ejecutar_pipelines([pipelineAKNN], preprocesar=False, algoritmo="RL", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    #ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN], preprocesar=False, algoritmo="XGBoost", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    #ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN], preprocesar=False, algoritmo="LinearSVM", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    #ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN], preprocesar=False, algoritmo="MLP", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    #ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="KNC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    #ejecutar_pipelines([pipelineSMOTE], preprocesar=False, algoritmo="DTC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
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
    
