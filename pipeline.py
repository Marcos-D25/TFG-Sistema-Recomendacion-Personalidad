import os
import pandas as pd
import numpy as np
from balanceador import Balanceador, BalanceadorSMOTE, BalanceadorBorderlineSMOTE, BalanceadorADASYN
from procesador import Preprocesador
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class Pipeline:
    def __init__(self, nombre_modelo):
        self.nombre_modelo = nombre_modelo

    def dividir_y_balancear(self, balanceador:Balanceador):
        self.dataset = pd.read_parquet(os.path.join("datasets", f"{self.nombre_modelo.replace('/', '_')}_dataset.parquet"))
        self.dataset_EI = self.dataset[["Embedding", "E/I"]].rename(columns={"E/I": "MBTI"})
        self.dataset_SN = self.dataset[["Embedding", "S/N"]].rename(columns={"S/N": "MBTI"})
        self.dataset_TF = self.dataset[["Embedding", "T/F"]].rename(columns={"T/F": "MBTI"})
        self.dataset_JP = self.dataset[["Embedding", "J/P"]].rename(columns={"J/P": "MBTI"})
    
        train_EI, self.test_EI, self.val_EI = np.split(self.dataset_EI.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_EI)), int(0.8*len(self.dataset_EI))])
        train_SN, self.test_SN, self.val_SN = np.split(self.dataset_SN.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_SN)), int(0.8*len(self.dataset_SN))])
        train_TF, self.test_TF, self.val_TF = np.split(self.dataset_TF.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_TF)), int(0.8*len(self.dataset_TF))])
        train_JP, self.test_JP, self.val_JP = np.split(self.dataset_JP.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_JP)), int(0.8*len(self.dataset_JP))])

        self.balanceador_EI = balanceador.balancear("MBTI")
        self.balanceador_SN = balanceador.balancear("MBTI")
        self.balanceador_TF = balanceador.balancear("MBTI")
        self.balanceador_JP = balanceador.balancear("MBTI")
    
    def regresion_logistica(self, nombreResultados):
        '''
        hiperparametros = {
            'C': [ 0.01, 0.1, 1, 10],
            'penalty': ['l1','l2'],
            'solver': ['liblinear']
        }
        '''
        #Mejor combinacion de hiperparametros: C=10, penalty='l1', solver='liblinear'
        X_train_EI = np.array(self.balanceador_EI["Embedding"].tolist())
        y_train_EI = self.balanceador_EI["MBTI"].tolist()
        X_train_SN = np.array(self.balanceador_SN["Embedding"].tolist())
        y_train_SN = self.balanceador_SN["MBTI"].tolist()
        X_train_TF = np.array(self.balanceador_TF["Embedding"].tolist())
        y_train_TF = self.balanceador_TF["MBTI"].tolist()
        X_train_JP = np.array(self.balanceador_JP["Embedding"].tolist())
        y_train_JP = self.balanceador_JP["MBTI"].tolist()

        '''
        self.lr_EI = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_EI, y_train_EI)
        print(self.lr_EI.best_params_)
        self.lr_SN = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_SN, y_train_SN)
        self.lr_TF = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_TF, y_train_TF)
        self.lr_JP = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), hiperparametros, scoring="accuracy", cv=5, n_jobs=-1, verbose=3).fit(X_train_JP, y_train_JP)
        '''
        print("[INFO] Entrenando modelos de regresión logística...")
        self.lr_EI = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=2000, random_state=42, verbose=3).fit(X_train_EI, y_train_EI)
        print("[INFO] Modelo E/I entrenado.")
        self.lr_SN = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=2000, random_state=42, verbose=3).fit(X_train_SN, y_train_SN)
        print("[INFO] Modelo S/N entrenado.")
        self.lr_TF = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=2000, random_state=42, verbose=3).fit(X_train_TF, y_train_TF)
        print("[INFO] Modelo T/F entrenado.")
        self.lr_JP = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=2000, random_state=42, verbose=3).fit(X_train_JP, y_train_JP)
        print("[INFO] Modelo J/P entrenado.")
        
        self.guardar_resultados(nombreResultados)
        os.makedirs("modelos_LR", exist_ok=True)
        for modelo, nombre in zip([self.lr_EI, self.lr_SN, self.lr_TF, self.lr_JP], ["E-I", "S-N", "T-F", "J-P"]):   
            self.guardar_modelo("modelos_LR", modelo, f"{nombre}_{self.nombre_modelo.replace('/', '_')}.pkl")

    def evaluar_y_reportar(self, modelo, df_test, nombre_eje, archivo_txt):
        X_test = np.array(df_test["Embedding"].tolist())
        y_test = df_test["MBTI"].tolist()

        y_pred = modelo.predict(X_test)
        
        cabecera = f"\n--- Métricas para el eje: {nombre_eje} ---\n"
        archivo_txt.write(cabecera)
        
        # Matriz de Confusión
        matriz_str = f"\nMatriz de Confusión:\n{confusion_matrix(y_test, y_pred)}\n"
        archivo_txt.write(matriz_str)
        
        # Reporte de Clasificación
        reporte_str = f"\nReporte de Clasificación:\n{classification_report(y_test, y_pred)}\n"
        archivo_txt.write(reporte_str)
        archivo_txt.write("-" * 50 + "\n")

    def guardar_resultados(self, nombre_archivo, carpeta="resultados"):
        with open(os.path.join(carpeta, nombre_archivo), "w", encoding="utf-8") as f:
            f.write("==========================================================\n")
            f.write(f"REPORTE DE ENTRENAMIENTO - {nombre_archivo}\n")
            f.write("==========================================================\n")
            
            self.evaluar_y_reportar(self.lr_EI, self.test_EI, "Introversión / Extroversión (E/I)", f)
            self.evaluar_y_reportar(self.lr_SN, self.test_SN, "Sensación / Intuición (S/N)", f)
            self.evaluar_y_reportar(self.lr_TF, self.test_TF, "Pensamiento / Sentimiento (T/F)", f)
            self.evaluar_y_reportar(self.lr_JP, self.test_JP, "Juicio / Percepción (J/P)", f)

    def guardar_modelo(self, carpeta, modelo, nombre_archivo):
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        joblib.dump(modelo, os.path.join(carpeta, nombre_archivo))

    #Pipeline que sirve para comprobar la mejor combinacion de hiperparametros para un modelo de clasificacion concreto
    def ejecutar_pipeline_test(self, preprocesar=True, modelo="RL"):
        print("[EJECUCION] Ejecutando pipeline completo ...]")

        #Paso 1: Preprocesamiento del dataset
        if preprocesar:
            archivo = os.path.join("datasets","MBTI_sinProcesar.csv")
            procesador = Preprocesador(archivo, self.nombre_modelo)
            procesador.procesar_dataset()

        print("[EJECUCION] Dividiendo y balanceando dataset ...]")
        #Paso 2: División y balanceo del dataset
        self.dividir_y_balancear()


        #Paso 3: Regresión logística para cada dimensión del MBTI
        self.regresion_logistica("resultados_RegLog.txt")
        print("[EJECUCION] Fin pipeline completo ...]")


if __name__ == "__main__":
    pipeline = Pipeline("FacebookAI/roberta-base")
    pipeline.ejecutar_pipeline_test(preprocesar=False)