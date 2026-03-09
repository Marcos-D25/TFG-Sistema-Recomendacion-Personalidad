
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, AllKNN
from abc import ABC, abstractmethod

class Balanceador(ABC):

    def __init__(self, nomCarpeta: str = "datasetRBFT", balancear: bool = True):
        '''
        Inicializador de clase que carga los cuatro archivos individuales de cada eje dentro
        de la carpeta proporcionada. Cada archivo debe llamarse:
        - datasetEI.parquet
        - datasetSN.parquet
        - datasetTF.parquet
        - datasetJP.parquet

        :param nomCarpeta: Ruta a la carpeta que contiene los datasets.
        :param balancear: Si se debe aplicar balanceo sintético al conjunto de entrenamiento.
        '''
        self.nomCarpeta = nomCarpeta
        self.balancear_flag = balancear

        self.datasets = {
            'E/I': pd.read_parquet(os.path.join(nomCarpeta, "datasetEI.parquet")),
            'S/N': pd.read_parquet(os.path.join(nomCarpeta, "datasetSN.parquet")),
            'T/F': pd.read_parquet(os.path.join(nomCarpeta, "datasetTF.parquet")),
            'J/P': pd.read_parquet(os.path.join(nomCarpeta, "datasetJP.parquet")),
        }

        self.train_EI = self.test_EI = self.val_EI = None
        self.train_SN = self.test_SN = self.val_SN = None
        self.train_TF = self.test_TF = self.val_TF = None
        self.train_JP = self.test_JP = self.val_JP = None

        

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

    def dividir_balancear_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Divide un DataFrame en train/val/test (80/10/10) y aplica balanceo al conjunto de
        entrenamiento si está habilitado.

        :param df: DataFrame con columnas "Embedding" y "MBTI".
        :return: (train, test, val)
        '''
        X = df.drop(columns=["MBTI"])
        y = df["MBTI"]

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_eval, y_eval, test_size=0.50, random_state=42, stratify=y_eval
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        if self.balancear_flag:
            train_df = self.balancear(train_df)

        return train_df, test_df, val_df

    def procesar_todos_ejes(self) -> None:
        '''
        Recorre cada eje cargado y lo divide/balancea guardando los resultados en atributos
        especializados (train_EI, etc.).
        Solo se ejecutará si en el constructor se indicó que se queria balancear
        '''
        
        for eje in ['E/I', 'S/N', 'T/F', 'J/P']:
            df = self.datasets[eje].copy()
            df = df[['Embedding', eje]].rename(columns={eje: 'MBTI'})
            train_df, test_df, val_df = self.dividir_balancear_df(df)
            setattr(self, f"train_{eje.replace('/', '')}", train_df)
            setattr(self, f"test_{eje.replace('/', '')}", test_df)
            setattr(self, f"val_{eje.replace('/', '')}", val_df)

    @abstractmethod
    def __str__(self):
        pass

class BalanceadorSMOTE(Balanceador):
    def __init__(self, nomCarpeta: str = "dataset9K", balancear: bool = True):
        super().__init__(nomCarpeta, balancear=balancear)
        self.balanceador = SMOTE(random_state=42)

    def __str__(self):
        return "SMOTE"

class BalanceadorBorderlineSMOTE(Balanceador):
    def __init__(self, nomCarpeta: str = "dataset9K", balancear: bool = True):
        super().__init__(nomCarpeta, balancear=balancear)
        self.balanceador = BorderlineSMOTE(random_state=42)

    def __str__(self):
        return "BorderlineSMOTE"

class BalanceadorADASYN(Balanceador):
    def __init__(self, nomCarpeta: str = "dataset9K", balancear: bool = True):
        super().__init__(nomCarpeta, balancear=balancear)
        self.balanceador = ADASYN(random_state=42)
    
    def __str__(self):
        return "ADASYN"

class BalanceadorENN(Balanceador):
    def __init__(self, nomCarpeta: str = "dataset9K", balancear: bool = True):
        super().__init__(nomCarpeta, balancear=balancear)
        self.balanceador = EditedNearestNeighbours()
    
    def __str__(self):
        return "EditedNearestNeighbours"

class BalanceadorAKNN(Balanceador):
    def __init__(self, nomCarpeta: str = "dataset9K", balancear: bool = True):
        super().__init__(nomCarpeta, balancear=balancear)
        self.balanceador = AllKNN()
    
    def __str__(self):
        return "AllKNN"

