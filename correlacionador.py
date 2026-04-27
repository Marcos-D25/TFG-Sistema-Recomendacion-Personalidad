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
             #   O      C       E      A       N
            [0.30,  -0.01,	-0.45,  -0.28,	0.04],#Aventura
            [-0.48, -0.19,	0.50,   0.52,   0.15],#Drama
            [-0.12, 0.02,   0.07,   0.10,	0.00],#Comedia
            [-0.49, -0.41,	0.06,   0.34,   0.34],#Romance
            [0.10,  -0.10,  0.25,   0.10,   0.20],#Horror
            [0.40,  0.10,   -0.10,  -0.10,  0.10]#Misterio
        ] 

        #Indicar en la memoria que los r^2 de los generos son muy bajos, representando menos del 10% de la influencia a elegir un videojuego
        #Puntuacion que devolveria es 0, 100
        self.corr_JUEGOS = [
            #   O     C     E      A     N
            [-.136, .054, .260, -.181, .158], #Juegos de disparos +43.740
            [-.050, .158, .298, -.141, .185], #Accion sin disparos (Como mario bros) +31.820
            [-.097, .074, .249, -.156, .170], #Juegos de lucha +39.573
            [.179, -.012, .004, -.063, .029], #Juegos estrategia por turnos 56.411
            [.057,  .007, .126, -.110, .075], #Juegos estrategia en tiempo real +54.150
            [.087, -.017, .044,  .016, .150], #Juegos RPG +49.764
            [-.155, .056, .233, -.107, .101], #Juegos de deportes +44.918
            [.083, -.161, .133,  .011, .124], #Juegos de carreras +43.424
            #Juegos de simulacion de construccion (No es lo suficiente relevante, <1%)
            [.256, -.084,-.068, -.047,-.079], #Juegos de simulacion como Spore o los Sims (Sandbox) +50.746
            [.246, .090, -.057,  .106, .065], #Juegos de aventura (enfocados en la historia) +40.297
            [.315, .180, -.041, .065, -.037], #Juegos de puzzles +25.799
            [-.357, .010, .454, -.239, .153], #Juegos multijugador online +58.935
        ]

        self.corr_MUSICA = [

        ]
        self.OCEAN = None
        self.GENEROS = None
    
    def correlacionar_OCEAN(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de MBTI a OCEAN

            :return: Diccionario con las % normalizadas para cada dimension de OCEAN
        '''
        res = self.corr_OCEAN @ self.prediccion #Multiplicacion de matrices
        '''
        #Normalizacion Z-Score + Sigmoide
        max_posibles = np.sum(np.abs(self.corr_OCEAN), axis=1, keepdims=True) #Hallo el maximo posible por cada dimension de OCEAN
        res = ((res / max_posibles) + 1) / 2
        '''
        #Normailzacion sigmoide
        t = 0.8 # Ajusta este valor empíricamente
        res = 1 / (1 + np.exp(-res / t))

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
        '''
        #Normalizacion Softmax
        t = 0.5 #Varia la "distancia" de los resultados incrementandola
        res = np.exp(res/t)
        res = res / np.sum(res)
        '''
        #Sigmoide
        res = 1 / (1 + np.exp(-res))
        
        return  dict(zip(["Aventura", "Drama", "Comedia", 'Romance','Horror','Misterio'],list(res)))