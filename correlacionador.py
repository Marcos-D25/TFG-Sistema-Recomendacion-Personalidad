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
            #   O     C     E      A     N
            [ 0.327,  0.190,  0.119,  0.173,  0.008], #Thrillers 
            [ 0.026,  0.090,  0.153,  0.262,  0.111], #Romance
            [ 0.012, -0.127, -0.003, -0.137, -0.070], #Westerns
            [ 0.117,  0.170,  0.112,  0.250, -0.047], #Comedy
            [ 0.091,  0.025,  0.089,  0.047, -0.080], #Action
            [ 0.313,  0.141,  0.249,  0.277,  0.064], #Drama
            [ 0.193, -0.093, -0.016,  0.022, -0.063], #Science Fiction
            [ 0.157,  0.042,  0.073,  0.167,  0.012], #Crime
            [ 0.033, -0.097,  0.057, -0.121,  0.030], #Horror
            [ 0.208, -0.044,  0.018,  0.045,  0.027], #Fantasy
            [ 0.115, -0.012,  0.171,  0.206,  0.112]  #Musicals
        ]

        self.generos_cine = [
            "Thrillers", "Romance", "Westerns", "Comedy", "Action", 
            "Drama", "Science Fiction", "Crime", "Horrors", "Fantasy", "Musicals"
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

        self.corr_MUSICA_1 = [
            #   O     C     E      A     N
            [-.002, .026, .103, .021, -.012], # R&B
            [-.019, -.017, .129, .008, -.049], #Rap
            [.077, -.029, .034, -.033, -.002], #Electronic
            [-.055, -.016, -.071, -.072, .057],  #Rock
            [.101, .008, -.067, -.019, -.031], #New Age
            [.136, -.037, -.064,  -.032, 0], #Classical
            [.017, -.042, .061, .009, -.041], #Reggae
            [.120, -.011, .023, -.011, -.044], #Blues
            [.106, -.049, -.002, .104, -.012],  #Country
            [.134, -.021, -.006, -.028, -.020], #World
            [.214, -.115, -.044, .104, .002], #Folk
            [.041, .010, .018, -.027, -.012], #Easy Listening
            [.139, -.007, .042, .031, -.061], #Jazz
            [.120, -.020, .006, -.021, .006], #Vocal (a capella)
            [.002, -.061, -.020, .001, .030], #Punk
            [.115, -.104, -.031, .060, .101], #Alternative
            [-.034, .035, .056, .056, -.030], #Pop
            [-.031, -.023, -.076, -.069, -.001] #Heavy Metal
        ]

        self.corr_MUSICA_2 = [
            #   O     C     E      A     N
            [.41, -.06, -.02, .03, .04], #Classical, Jazz, Blues, Folk (Reflective & Complex)
            [.15, -.03, .08, .01, -.01], #Alternative, Rock, Heavy Metal (Intense & Rebellious)
            [-.08, .18, .15, .24, -.04], #Country, Pop, Religious, Sound Tracks
            [.04, -.03, .19, .09, -.01]  #Rap & Hip-Hop, Soul & Funk, Electronic & Dance
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