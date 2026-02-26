import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from abc import ABC, abstractmethod

class Balanceador(ABC):

    def __init__(self, nombre_dataset:str):
        self.dataset = pd.read_parquet(os.path.join("datasets", nombre_dataset))

    def verComparativa(self,dataset, columna = "MBTI"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        dataset_original = self.dataset[columna].value_counts()
        dataset_original.plot(kind="bar", ax=ax1)
        ax1.set_title("Antes de balanceo")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--')

        dataset_balanceado = dataset[columna].value_counts()
        dataset_balanceado.plot(kind="bar", ax=ax2)
        ax2.set_title("Después de balanceo")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle='--')

        plt.tight_layout()
        plt.show()

    def balancear(self, dataset):
        X = dataset["Embedding"].tolist()
        y = dataset["MBTI"].tolist()
        try:
            X_resampled, y_resampled = self.balanceador.fit_resample(X, y)
            df_balanceado = pd.DataFrame({
                "MBTI": y_resampled,
                "Embedding": X_resampled
            })
            return df_balanceado
        except Exception as e:
            #ADASYN puede lanzar un error si ve que las clases no estan tan desbalanceadas, en ese caso se devuelve el dataset original sin balancear
            print(f"[ERROR] Error al balancear el dataset: {e}")
            return dataset.copy()

        

    def dividir_balancear(self):
        self.dataset_EI = self.dataset[["Embedding", "E/I"]].rename(columns={"E/I": "MBTI"})
        self.dataset_SN = self.dataset[["Embedding", "S/N"]].rename(columns={"S/N": "MBTI"})
        self.dataset_TF = self.dataset[["Embedding", "T/F"]].rename(columns={"T/F": "MBTI"})
        self.dataset_JP = self.dataset[["Embedding", "J/P"]].rename(columns={"J/P": "MBTI"})
    
        train_EI, self.test_EI, self.val_EI = np.split(self.dataset_EI.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_EI)), int(0.8*len(self.dataset_EI))])
        train_SN, self.test_SN, self.val_SN = np.split(self.dataset_SN.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_SN)), int(0.8*len(self.dataset_SN))])
        train_TF, self.test_TF, self.val_TF = np.split(self.dataset_TF.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_TF)), int(0.8*len(self.dataset_TF))])
        train_JP, self.test_JP, self.val_JP = np.split(self.dataset_JP.sample(frac=1, random_state=42), [int(0.6*len(self.dataset_JP)), int(0.8*len(self.dataset_JP))])

        self.train_bal_EI = self.balancear(train_EI)
        self.train_bal_SN = self.balancear(train_SN)
        self.train_bal_TF = self.balancear(train_TF)
        self.train_bal_JP = self.balancear(train_JP)

    @abstractmethod
    def __str__(self):
        pass

class BalanceadorSMOTE(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = SMOTE(sampling_strategy="not majority", random_state=42)

    def __str__(self):
        return "SMOTE"

class BalanceadorBorderlineSMOTE(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = BorderlineSMOTE(sampling_strategy="not majority", random_state=42)

    def __str__(self):
        return "BorderlineSMOTE"

class BalanceadorADASYN(Balanceador):
    def __init__(self, nombre_dataset:str):
        super().__init__(nombre_dataset)
        self.balanceador = ADASYN(sampling_strategy="not majority", random_state=42)
    
    def __str__(self):
        return "ADASYN"

    
