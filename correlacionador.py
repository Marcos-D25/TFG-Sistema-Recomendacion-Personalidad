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
        self.prediccion = np.array([prediccion[dim][0] for dim in prediccion.keys()]).reshape(-1,1) - 0.5 #Genera una matriz 8x1 centrada para que no se incluya ruido en la conversion 

        self.corr_OCEAN = [ #Matriz ordenada sacada del articulo "Correlation based data unification for personality trait prediction", la salida coincide con O C E A N
            [ 0.28, -0.32, -0.66,  0.64, -0.17,  0.13, -0.25,  0.26],
            [ 0.13, -0.13,  0.10, -0.13,  0.22, -0.27,  0.46, -0.46],
            [ 0.71, -0.72, -0.28,  0.27,  0.00,  0.00, -0.13,  0.16],
            [-0.02,  0.02,  0.01,  0.00, -0.41,  0.28,  0.05, -0.06],
            [-0.30,  0.31,  0.15, -0.14, -0.13,  0.12,  0.07, -0.07]
        ]

        self.corr_GENEROS = [
            [0.30,  -0.01,	-0.45,  -0.28,	0.04],#Aventura
            [-0.48, -0.19,	0.50,   0.52,   0.15],#Drama
            [-0.12, 0.02,   0.07,   0.10,	0.00],#Comedia
            [-0.49, -0.41,	0.06,   0.34,   0.34],#Romance
            [0.10,  -0.10,  0.25,   0.10,   0.20],#Horror
            [0.40,  0.10,   -0.10,  -0.10,  0.10]#Misterio
        ] 
        self.OCEAN = None
        self.GENEROS = None
    
    def correlacionar_OCEAN(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de MBTI a OCEAN

            :return: Diccionario con las % normalizadas para cada dimension de OCEAN
        '''
        res = self.corr_OCEAN @ self.prediccion #Multiplicacion de matrices
        #Normalizacion Z-Score + Sigmoide
        max_posibles = np.sum(np.abs(self.corr_OCEAN), axis=1, keepdims=True) #Hallo el maximo posible por cada dimension de OCEAN
        res = ((res / max_posibles) + 1) / 2

        self.OCEAN = res
        personality_traits = ['Openness', 'Concientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        return dict(zip(personality_traits, res.reshape(1,-1)[0]*100))

    def correlacionar_GENEROS(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de OCEAN a GENEROS (generos audiovisuales)
            
            :return: Diccionario con los % normalizados para cada género
        '''

        if(self.OCEAN == None): #Se debe de tener el array OCEAN para continuar
            self.correlacionar_OCEAN()
    
        res = self.corr_GENEROS @ self.OCEAN.reshape(1,-1)[0]
        #Normalizacion Softmax
        t = 0.5 #Varia la "distancia" de los resultados incrementandola
        res = np.exp(res/t)
        res = res / np.sum(res)
        return  dict(zip(["Aventura", "Drama", "Comedia", 'Romance','Horror','Misterio'],list(res)))