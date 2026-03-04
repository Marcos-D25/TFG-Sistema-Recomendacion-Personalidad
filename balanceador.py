import os
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, AllKNN
from abc import ABC, abstractmethod

class Balanceador(ABC):

    def __init__(self, nombre_dataset:str):
        '''
        Inicializador de clase.
        
        :param nombre_dataset: Nombre del archivo del dataset, se espera que esté en la carpeta local "datasets"
        '''
        self.dataset = pd.read_parquet(os.path.join("datasets", nombre_dataset))

    def balancear(self, dataset:pd.DataFrame, columnas:dict={"c1":"Embedding", "c2": "MBTI"}) -> pd.DataFrame:
        '''
        Funcion que sirve para ejecutar el balanceo propio de la clase al dataset pasado por parametro.
        Si el dataset pasado como parametro no cuenta con desbalanceo (la clase minoritaria es más del 50% que la mayoritaria) se devuelve una copia del dataset

        :param dataset: Dataset al cual se le va a aplicar el balanceo. Se espera que tenga 2 columnas solamente
        :param columnas: Diccionario con las columnas del dataset ej. {"c1":"Embedding", "c2": "MBTI"}  
        :return: Devuelve el dataset
        '''
        
        X = dataset[columnas["c1"]].tolist()
        y = dataset[columnas["c2"]].tolist()
        
        #Ratio real de desbalanceo
        conteo = Counter(y)
        clases_ordenadas = conteo.most_common()
        mayoritaria_count = clases_ordenadas[0][1]
        minoritaria_count = clases_ordenadas[1][1]
        
        ratio_actual = minoritaria_count / mayoritaria_count
        
        
        if ratio_actual >= 0.5:
            print(f"   [INFO] Eje naturalmente equilibrado (Ratio: {ratio_actual:.2f}). Se omite balanceo sintético.")
            return dataset.copy()
            

        try:
            X_resampled, y_resampled = self.balanceador.fit_resample(X, y)
            df_balanceado = pd.DataFrame({
                "MBTI": y_resampled,
                "Embedding": X_resampled
            })
            return df_balanceado
            
        except Exception as e:
            print(f"⚠️ [AVISO] {self.balanceador.__str__()} falló: {e}")
            return dataset.copy()

    def dividir_balancear(self, balancear:bool = True) -> None:
        '''
        Funcion que genera un dataset por cada subclase E/I S/N T/F J/P, de cada dataset se subdivide en sets de entrenamiento.
            80% Entrenamiento - 10% Validacion - 10% Test

        :param balancear: Indica si se quiere aplicar una tecnica de balanceo o no
        :return: None
        '''
        dataset_EI = self.dataset[["Embedding", "E/I"]].rename(columns={"E/I": "MBTI"})
        dataset_SN = self.dataset[["Embedding", "S/N"]].rename(columns={"S/N": "MBTI"})
        dataset_TF = self.dataset[["Embedding", "T/F"]].rename(columns={"T/F": "MBTI"})
        dataset_JP = self.dataset[["Embedding", "J/P"]].rename(columns={"J/P": "MBTI"})
    
        
        def procesar_eje(df)-> tuple [pd.DataFrame,pd.DataFrame,pd.DataFrame]:
            '''
            Funcion interna que realiza la division train, test, eval segun el dataset pasado como párametro.

            :param df: Datasets a dividir
            :return: Los 3 datasets divididos, (Train, Test, Val)
            '''


            X = df.drop(columns=["MBTI"])
            y = df["MBTI"]
            
            #80% para Train, 10% Validacion, 10% Test 
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
            
            X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.50, random_state=42, stratify=y_eval)
            
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            # Devolvemos el train balanceado, y los otros dos intactos (mundo real)
            return train_df, test_df, val_df

        # 3. Aplicamos el proceso a los 4 ejes de personalidad
        self.train_EI, self.test_EI, self.val_EI = procesar_eje(dataset_EI)
        self.train_SN, self.test_SN, self.val_SN = procesar_eje(dataset_SN)
        self.train_TF, self.test_TF, self.val_TF = procesar_eje(dataset_TF)
        self.train_JP, self.test_JP, self.val_JP = procesar_eje(dataset_JP)

        if balancear:
            self.train_EI = self.balancear(self.train_EI)
            self.train_SN = self.balancear(self.train_SN)
            self.train_TF = self.balancear(self.train_TF)
            self.train_JP = self.balancear(self.train_JP)

    @abstractmethod
    def __str__(self):
        pass

class BalanceadorSMOTE(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = SMOTE(random_state=42)

    def __str__(self):
        return "SMOTE"

class BalanceadorBorderlineSMOTE(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = BorderlineSMOTE(random_state=42)

    def __str__(self):
        return "BorderlineSMOTE"

class BalanceadorADASYN(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = ADASYN(random_state=42)
    
    def __str__(self):
        return "ADASYN"

class BalanceadorENN(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = EditedNearestNeighbours()
    
    def __str__(self):
        return "EditedNearestNeighbours"

class BalanceadorAKNN(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = AllKNN()
    
    def __str__(self):
        return "AllKNN"
