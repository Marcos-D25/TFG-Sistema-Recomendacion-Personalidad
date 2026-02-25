import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from abc import ABC, abstractmethod

class Balanceador(ABC):

    def __init__(self, dataset):
        self.dataset = dataset

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

    @abstractmethod
    def balancear(self, columna):
        pass

class BalanceadorSMOTE(Balanceador):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.SMOTE = SMOTE(sampling_strategy="not majority", random_state=42)

    def balancear(self, columna):
        X = self.dataset["Embedding"].tolist()
        y = self.dataset[columna].tolist()

        X_resampled, y_resampled = self.SMOTE.fit_resample(X, y)

        df_balanceado = pd.DataFrame({
            "MBTI": y_resampled,
            "Embedding": X_resampled
        })

        return df_balanceado

class BalanceadorBorderlineSMOTE(Balanceador):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.BorderlineSMOTE = BorderlineSMOTE(sampling_strategy="not majority", random_state=42)

    def balancear(self, columna):
        X = self.dataset["Embedding"].tolist()
        y = self.dataset[columna].tolist()

        X_resampled, y_resampled = self.BorderlineSMOTE.fit_resample(X, y)

        df_balanceado = pd.DataFrame({
            "MBTI": y_resampled,
            "Embedding": X_resampled
        })

        return df_balanceado

class BalanceadorADASYN(Balanceador):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.ADASYN = ADASYN(sampling_strategy="not majority", random_state=42)

    def balancear(self, columna):
        X = self.dataset["Embedding"].tolist()
        y = self.dataset[columna].tolist()

        X_resampled, y_resampled = self.ADASYN.fit_resample(X, y)

        df_balanceado = pd.DataFrame({
            "MBTI": y_resampled,
            "Embedding": X_resampled
        })

        return df_balanceado


    
