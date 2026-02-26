import os
import pandas as pd
import numpy as np
from balanceador import Balanceador, BalanceadorSMOTE, BalanceadorBorderlineSMOTE, BalanceadorADASYN
from procesador import Preprocesador
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
from openpyxl import load_workbook
from xgboost import XGBClassifier

class Pipeline:
    def __init__(self, nombre_modelo, balanceador:Balanceador=None):
        self.nombre_modelo = nombre_modelo
        self.nombre_balanceador = balanceador.__str__() if balanceador else None
        self.balanceador = balanceador

    def xgboost(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.7}

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
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1.0]
        }

        xgb = XGBClassifier(device = "cuda", tree_method="hist",random_state=42, eval_metric="logloss")

        xgb_EI = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_EI, y_train_EI)
        xgb_SN = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_SN, y_train_SN)
        xgb_TF = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_TF, y_train_TF)
        xgb_JP = GridSearchCV(xgb, hiperparametros, scoring="accuracy", cv=5, n_jobs=1, verbose=2).fit(X_train_JP, y_train_JP)

        hiperparametros_info = {
            'E/I': xgb_EI.best_params_,
            'S/N': xgb_SN.best_params_,
            'T/F': xgb_TF.best_params_,
            'J/P': xgb_JP.best_params_
        }
        
        nombre_archivo = "hiperparametros_XGBoost.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros XGBoost - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        '''
        xgb_EI = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric="logloss").fit(X_train_EI, y_train_EI)
        xgb_SN = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric="logloss").fit(X_train_SN, y_train_SN)
        xgb_TF = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric="logloss").fit(X_train_TF, y_train_TF)
        xgb_JP = XGBClassifier(learning_rate=parametros['learning_rate'], max_depth=parametros['max_depth'], n_estimators=parametros['n_estimators'], subsample=parametros['subsample'], device = "cuda", tree_method="hist",random_state=42, eval_metric="logloss").fit(X_train_JP, y_train_JP)

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
            parametros = {'C': 10, 'loss': 'squared_hinge', 'max_iter': 1000, 'penalty': 'l2'}

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
            'penalty': ['l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [1000, 2000, 3000]
        }

        lSVM = LinearSVC(random_state=42)

        lSVM_EI = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_EI, y_train_EI)
        lSVM_SN = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_SN, y_train_SN)
        lSVM_TF = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_TF, y_train_TF)
        lSVM_JP = GridSearchCV(lSVM, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_JP, y_train_JP)

        hiperparametros_info = {
            'E/I': lSVM_EI.best_params_,
            'S/N': lSVM_SN.best_params_,
            'T/F': lSVM_TF.best_params_,
            'J/P': lSVM_JP.best_params_
        }
        
        nombre_archivo = "hiperparametros_LinearSVM.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros LinearSVM - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        
        '''

        lSVM_EI = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'], random_state=42).fit(X_train_EI, y_train_EI)
        lSVM_SN = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'], random_state=42).fit(X_train_SN, y_train_SN)
        lSVM_TF = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'], random_state=42).fit(X_train_TF, y_train_TF)
        lSVM_JP = LinearSVC(C=parametros['C'], loss=parametros['loss'], max_iter=parametros['max_iter'], penalty=parametros['penalty'], random_state=42).fit(X_train_JP, y_train_JP)

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

    def MLP(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros = {
                'hidden_layer_sizes': (100,),
                'activation': "relu",
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 200
            }

        X_train_EI = np.array(self.balanceador.train_bal_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_bal_EI["MBTI"].tolist()
        X_train_SN = np.array(self.balanceador.train_bal_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_bal_SN["MBTI"].tolist()
        X_train_TF = np.array(self.balanceador.train_bal_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_bal_TF["MBTI"].tolist()
        X_train_JP = np.array(self.balanceador.train_bal_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_bal_JP["MBTI"].tolist()

        hiperparametros = {
            'hidden_layer_sizes': [(100,), (100, 50), (50,)],
            'activation': ["relu", "tanh", "logistic"],
            'solver': ['adam', 'sgd',  'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000, 1500, 2000]
        }

        mlp = MLPClassifier(random_state=42)

        mlp_EI = GridSearchCV(mlp, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_EI, y_train_EI)
        #mlp_SN = GridSearchCV(mlp, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_SN, y_train_SN)
        #mlp_TF = GridSearchCV(mlp, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_TF, y_train_TF)
        #mlp_JP = GridSearchCV(mlp, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_JP, y_train_JP)

        hiperparametros_info = {
            'E/I': mlp_EI.best_params_,
            #'S/N': mlp_SN.best_params_,
            #'T/F': mlp_TF.best_params_,
            #'J/P': mlp_JP.best_params_
        }
        
        nombre_archivo = "hiperparametros_MLP.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros MLP - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        
        '''
        self.modelos = { 
            'EI': mlp_EI,
            'SN': mlp_SN,
            'TF': mlp_TF,
            'JP': mlp_JP
        }
        '''

        #self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="XGBoost")
        #os.makedirs("modelos_LR", exist_ok=True)
        #for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
        #    self.guardar_modelo("modelos_XGB", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")

    def KNN(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 30
            }

        X_train_EI = np.array(self.balanceador.train_bal_EI["Embedding"].tolist())
        y_train_EI = self.balanceador.train_bal_EI["MBTI"].tolist()
        X_train_SN = np.array(self.balanceador.train_bal_SN["Embedding"].tolist())
        y_train_SN = self.balanceador.train_bal_SN["MBTI"].tolist()
        X_train_TF = np.array(self.balanceador.train_bal_TF["Embedding"].tolist())
        y_train_TF = self.balanceador.train_bal_TF["MBTI"].tolist()
        X_train_JP = np.array(self.balanceador.train_bal_JP["Embedding"].tolist())
        y_train_JP = self.balanceador.train_bal_JP["MBTI"].tolist()

        hiperparametros = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [20, 30, 40],
            'metric': ['euclidean', 'manhattan', 'minkowski']

        }

        knn = KNeighborsClassifier(random_state=42)

        knn_EI = GridSearchCV(knn, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_EI, y_train_EI)
        #knn_SN = GridSearchCV(knn, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_SN, y_train_SN)
        #knn_TF = GridSearchCV(knn, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_TF, y_train_TF)
        #knn_JP = GridSearchCV(knn, hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=1).fit(X_train_JP, y_train_JP)
        
        hiperparametros_info = {
            'E/I': knn_EI.best_params_,
            #'S/N': knn_SN.best_params_,
            #'T/F': knn_TF.best_params_,
            #'J/P': knn_JP.best_params_
        }
        
        nombre_archivo = "hiperparametros_KNN.txt"
        with open(nombre_archivo, 'w') as f:
            f.write(f"Hiperparámetros KNN - {self.nombre_modelo}\n")
            f.write("="*50 + "\n\n")
            for dimension, params in hiperparametros_info.items():
                f.write(f"{dimension}: {params}\n")
        
        print(f"[INFO] Hiperparámetros guardados en {nombre_archivo}")
        
        '''
        self.modelos = { 
            'EI': knn_EI,
            'SN': knn_SN,
            'TF': knn_TF,
            'JP': knn_JP
        }
        '''

        #self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="XGBoost")
        #os.makedirs("modelos_LR", exist_ok=True)
        #for modelo, nombre in zip(self.modelos.values(), ["E-I", "S-N", "T-F", "J-P"]):   
        #    self.guardar_modelo("modelos_XGB", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")

    def regresion_logistica(self, nombre_Archivo, parametros = None):
        if not parametros: # En el caso de que no pasen parametros, se usan unos por defecto
            parametros={"C": 10, "penalty": 'l2', "solver": 'liblinear'}

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
            'C': [ 0.01, 0.1, 1, 10],
            'penalty': ['l1','l2'],
            'solver': ['liblinear']
        }
        self.lr_EI = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)
        self.lr_SN = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_SN, y_train_SN)
        self.lr_TF = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_TF, y_train_TF)
        self.lr_JP = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_JP, y_train_JP)
        '''
        lr_EI = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], max_iter=2000, random_state=42, verbose=3).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        lr_SN = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], max_iter=2000, random_state=42, verbose=3).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        lr_TF = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], max_iter=2000, random_state=42, verbose=3).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        lr_JP = LogisticRegression(C=parametros["C"], penalty=parametros["penalty"], solver=parametros["solver"], max_iter=2000, random_state=42, verbose=3).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        #Diccionario para almacenar los modelos entrenados
        self.modelos = { 
            'EI': lr_EI,
            'SN': lr_SN,
            'TF': lr_TF,
            'JP': lr_JP
        }

        self.guardar_resultados(nombre_Archivo=nombre_Archivo, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion="Regresión Logística")
        #os.makedirs("modelos_LR", exist_ok=True)
        #for modelo, nombre in zip([self.lr_EI, self.lr_SN, self.lr_TF, self.lr_JP], ["E-I", "S-N", "T-F", "J-P"]):   
        #    self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")

    def obtener_metricas(self, modelo, df_test, nombre_modelo):
        X_test = np.array(df_test["Embedding"].tolist())
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
    def ejecutar_pipeline_test(self, preprocesar=True, parametros=None, modelo=None, nombre_Archivo="ResultadosRegresion.xlsx"):
        print("[EJECUCION] Ejecutando pipeline completo ...]")

        #Paso 1: Preprocesamiento del dataset (opcional, se puede saltar si ya se ha preprocesado antes y se tiene el parquet guardado)
        if preprocesar:
            archivo = os.path.join("datasets","MBTI_sinProcesar.csv")
            procesador = Preprocesador(archivo, self.nombre_modelo)
            procesador.procesar_dataset()

        print("[EJECUCION] Dividiendo y balanceando dataset ...]")
        #Paso 2: División y balanceo del dataset
        self.balanceador.dividir_balancear()

        #Paso 3: Dependiendo del modelo, entrenamiento y evaluación y exportación de resultados
        match modelo:
            case "RL": 
                print(f"[EJECUCION] Entrenando modelo de Regresión Logística con {self.nombre_balanceador} ...")
                self.regresion_logistica(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "XGBoost":
                print(f"[EJECUCION] Entrenando modelo XGBoost con {self.nombre_balanceador} ...")
                self.xgboost(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "LinearSVM":
                print(f"[EJECUCION] Entrenando modelo LinearSVM con {self.nombre_balanceador} ...")
                self.LinearSVM(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "MLP":
                print(f"[EJECUCION] Entrenando modelo MLP con {self.nombre_balanceador} ...")
                self.MLP(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case "KNN":
                print(f"[EJECUCION] Entrenando modelo KNN con {self.nombre_balanceador} ...")
                self.KNN(nombre_Archivo=nombre_Archivo, parametros=parametros)
            case _ : 
                print("[ERROR] Modelo de clasificación no reconocido")

        print("[EJECUCION] Fin pipeline completo ...]")


if __name__ == "__main__":

    #EJECUCION PIPELINE ROBERTA BASE
    nombre_modelo = "FacebookAI/roberta-base"
    nombre_dataset = f"{nombre_modelo.replace('/', '_')}_dataset.parquet"
    
    balSMOTE = BalanceadorSMOTE(nombre_dataset=nombre_dataset)
    balBORSMOTE = BalanceadorBorderlineSMOTE(nombre_dataset=nombre_dataset)
    balADASYN = BalanceadorADASYN(nombre_dataset=nombre_dataset)

    pipelineSMOTE = Pipeline(nombre_modelo=nombre_modelo, balanceador=balSMOTE)
    pipelineBORSMOTE = Pipeline(nombre_modelo=nombre_modelo, balanceador=balBORSMOTE)
    pipelineADASYN = Pipeline(nombre_modelo=nombre_modelo, balanceador=balADASYN)

    #pipelineSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="RL", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineBORSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="RL", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineADASYN.ejecutar_pipeline_test(preprocesar=False, modelo="RL", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    
    #pipelineSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="XGBoost", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineBORSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="XGBoost", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineADASYN.ejecutar_pipeline_test(preprocesar=False, modelo="XGBoost", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")

    #pipelineSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="LinearSVM", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineBORSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="LinearSVM", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineADASYN.ejecutar_pipeline_test(preprocesar=False, modelo="LinearSVM", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")

    pipelineSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="MLP", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineBORSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="MLP", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineADASYN.ejecutar_pipeline_test(preprocesar=False, modelo="MLP", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")

    pipelineSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="KNN", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineBORSMOTE.ejecutar_pipeline_test(preprocesar=False, modelo="KNN", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")
    #pipelineADASYN.ejecutar_pipeline_test(preprocesar=False, modelo="KNN", nombre_Archivo=f"Resultados_{nombre_modelo.replace('/', '_')}.xlsx")