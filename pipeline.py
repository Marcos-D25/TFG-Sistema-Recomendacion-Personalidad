import os
import pandas as pd
import numpy as np
import joblib
from openpyxl import load_workbook
import optuna
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from balanceador import Balanceador, BalanceadorSMOTE, BalanceadorBorderlineSMOTE, BalanceadorADASYN, BalanceadorENN, BalanceadorAKNN
from procesador import Preprocesador
from clasificador import Clasificador, XGB, LR, MLPC,LSVC, DTC, KNC

class Pipeline:
    def __init__(self, nombre_modelo, balanceador:Balanceador=None):
        self.nombre_modelo = nombre_modelo
        self.nombre_balanceador = balanceador.__str__() if balanceador else None
        self.balanceador = balanceador

    def entreno_clasificador(self, buscar_hiper:bool=False,parametros:dict=None, guardar:bool=True, nomExcel:str="resultados.xlsx") -> None:
        '''
        Funcion que entrena el clasificador elegido.

        :param buscar_hiper: Bool que decide si buscar la mejor combinacion de hiperparametros antes de realizar el entreno. Si True se ignorarán los parametros recibidos en la función
        :param parametros: Diccionario que contiene los parametros especificos para el entreno. En caso de no indicar ninguno, se usarán los parametros predefinidos de la clase
        :param guardar: Bool que indica si se guardan los modelos en una carpeta local con el nombre del clasificador
        :param nomExcel: Nombre del archivo excel en el que guardar los resultados de los modelos entrenados
        :return: None
        '''

        if buscar_hiper:
            parametros = self.clasificador.busqueda_hiperparametros()
        elif not parametros: 
            parametros = self.clasificador.getParametros()
        
        self.clasificador.entrenar_modelo()

        self.modelos = self.clasificador.getModelos()
        
        if guardar:
            self.clasificador.guardar_modelo(f"modelos_{self.clasificador.__str__()}", self.nombre_modelo.replace("/","_"))

        self.guardar_resultados(nombre_Archivo=nomExcel, metodo_balanceo=self.nombre_balanceador, parametros_str=str(parametros), modelo_clasificacion=self.clasificador.__str__())

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
        filas.append(self.obtener_metricas(self.modelos['E/I'], self.balanceador.test_EI, f"{modelo_clasificacion} E/I"))
        filas.append(self.obtener_metricas(self.modelos['S/N'], self.balanceador.test_SN, f"{modelo_clasificacion} S/N"))
        filas.append(self.obtener_metricas(self.modelos['T/F'], self.balanceador.test_TF, f"{modelo_clasificacion} T/F"))
        filas.append(self.obtener_metricas(self.modelos['J/P'], self.balanceador.test_JP, f"{modelo_clasificacion} J/P"))
        
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


    #Pipeline que sirve para comprobar la mejor combinacion de hiperparametros para un modelo de clasificacion concreto
    def ejecutar_pipeline_entreno(self,preprocesar=True, parametros=None, algotitmo=None, nombre_Archivo="resultados.xlsx"):
        print("[EJECUCION] Ejecutando pipeline completo ...")
        
        print("[EJECUCION] Dividiendo y balanceando dataset ...")
        self.balanceador.dividir_balancear()

        self.clasificador:Clasificador = None

        match algotitmo:
            case "RL": 
                print(f"[EJECUCION] Entrenando modelo de Regresión Logística con {self.nombre_balanceador} ...")
                self.clasificador = LR(balanceador=self.balanceador)
            case "XGBoost":
                print(f"[EJECUCION] Entrenando modelo XGBoost con {self.nombre_balanceador} ...")
                self.clasificador = XGB(balanceador=self.balanceador)
            case "LinearSVC":
                print(f"[EJECUCION] Entrenando modelo LinearSVC con {self.nombre_balanceador} ...")
                self.clasificador = LSVC(balanceador=self.balanceador)
            case "KNC":
                print(f"[EJECUCION] Entrenando modelo KNeighborsClassifier con {self.nombre_balanceador}...")
                self.clasificador = KNC(balanceador=self.balanceador)
            case "DTC":
                print(f"[EJECUCION] Entrenando modelo DecisionTreeClassifier con {self.nombre_balanceador}...")
                self.clasificador = DTC(balanceador=self.balanceador)
            case "MLP":
                print(f"[EJECUCION] Entrenando modelo MultiLayerPerceptron con {self.nombre_balanceador}...")
                self.clasificador = MLPC(balanceador=self.balanceador)
            case _ : 
                print("[ERROR] Modelo de clasificación no reconocido")
                return None        
        
        self.entreno_clasificador(parametros=parametros)

        print("[EJECUCION] Fin pipeline completo ...")

#Funcion que sirve para sacar las metricas de entrenamiento para un modelo de embedding concreto
def pipeline_modelo_entreno(modelo:str, preprocesar:bool = False, nomCarpeta:str="dataset9K"):

    nombre_dataset = f"{modelo.replace('/', '_')}_dataset.parquet"
    
    print("[EJECUCION] Preprocesando dataset ...")
    if preprocesar:
        procesador = Preprocesador(nomCarpeta, modelo)
        procesador.procesar_dataset()
    
    balSMOTE = BalanceadorSMOTE(nombre_dataset=nombre_dataset)
    balBORSMOTE = BalanceadorBorderlineSMOTE(nombre_dataset=nombre_dataset)
    balADASYN = BalanceadorADASYN(nombre_dataset=nombre_dataset)
    balENN = BalanceadorENN(nombre_dataset=nombre_dataset)
    balAKNN = BalanceadorAKNN(nombre_dataset=nombre_dataset)
    
    pipelineSMOTE = Pipeline(nombre_modelo=modelo, balanceador=balSMOTE)
    pipelineBORSMOTE = Pipeline(nombre_modelo=modelo, balanceador=balBORSMOTE)
    pipelineADASYN = Pipeline(nombre_modelo=modelo, balanceador=balADASYN)
    pipelineENN = Pipeline(nombre_modelo=modelo, balanceador=balENN)
    pipelineAKNN = Pipeline(nombre_modelo=modelo, balanceador=balAKNN)

    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="RL", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="XGBoost", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="LinearSVC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="MLP", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="KNC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    ejecutar_pipelines([pipelineSMOTE, pipelineBORSMOTE, pipelineADASYN, pipelineENN, pipelineAKNN], algoritmo="DTC", nombre_Archivo=f"Resultados_{modelo.replace('/', '_')}.xlsx")
    

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
    
    #EJECUCION PIPELINE XLM-ROBERTA-BASE
    print("="*50)
    print("[INICIO] Ejecución pipeline con XLM Roberta Base ...")
    nombre_modelo = "FacebookAI/xlm-roberta-base"
    pipeline_modelo_entreno(nombre_modelo,preprocesar=True)
    print("[FIN] Ejecución pipeline con XLM Roberta Base ...")
    print("="*50)

    #EJECUCION PIPELINE XLM-ROBERTA-LARGE
    print("="*50)
    print("[INICIO] Ejecución pipeline con XLM Roberta Large ...")
    nombre_modelo = "FacebookAI/xlm-roberta-large"
    pipeline_modelo_entreno(nombre_modelo,preprocesar=True)
    print("[FIN] Ejecución pipeline con XLM Roberta Large ...")
    print("="*50)
    '''
    '''
