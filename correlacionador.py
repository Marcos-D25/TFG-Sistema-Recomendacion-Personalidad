import os
import numpy as np
from pipeline import Pipeline

class Correlacionador:
    '''
        Clase que permite establecer las correlaciones:
            MBTI -> OCEAN
            OCEAN -> GENEROS AUDIOVISUALES
    '''
    def __init__(self, prediccion:dict):
        '''
            Constructor de la clase Correlacionador.

            :param prediccion: Diccionario que contiene la prediccion generada por la funcion "generar_predicciones" de la clase Pipeline
        '''
        self.prediccion = np.array([prediccion[dim][0] for dim in prediccion.keys()]).reshape(-1,1) #Genera una matriz 8x1

        self.corr_OCEAN = [ #Matriz sacada del articulo "Correlation based data unification for personality trait prediction"
            [-0.30,  0.31,  0.15, -0.14, -0.13,  0.12,  0.07, -0.07],
            [ 0.71, -0.72, -0.28,  0.27,  0.00,  0.00, -0.13,  0.16],
            [ 0.28, -0.32, -0.66,  0.64, -0.17,  0.13, -0.25,  0.26],
            [-0.02,  0.02,  0.01,  0.00, -0.41,  0.28,  0.05, -0.06],
            [ 0.13, -0.13,  0.10, -0.13,  0.22, -0.27,  0.46, -0.46]
        ]
    
    def correlacionar_OCEAN(self) -> np.ndarray:
        '''
            Funcion que genera automaticamente la correlacion de MBTI a OCEAN

            :return 
        '''
        res = self.corr_OCEAN @ self.prediccion #Multiplicacion de matrices
        max_posibles = np.sum(np.abs(self.corr_OCEAN), axis=1, keepdims=True) #Hallo el maximo posible por cada dimension de OCEAN
        res = ((res / max_posibles) + 1) / 2

        return res
    
    