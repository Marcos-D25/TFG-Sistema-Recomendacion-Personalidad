import os
import numpy as np
from pipeline import Pipeline

class Correlacionador:
    '''
        Clase que permite establecer las correlaciones:
            MBTI -> OCEAN
            OCEAN -> GENEROS AUDIOVISUALES / VIDEOJUEGOS / MUSICA
    '''
    def __init__(self, prediccion:dict):
        '''
            Constructor de la clase Correlacionador.

            :param prediccion: Diccionario que contiene la prediccion generada por la funcion "generar_predicciones" de la clase Pipeline
        '''
        self.prediccion = np.array([prediccion[dim] for dim in prediccion.keys()]).reshape(-1,1) - 0.5 #Genera una matriz 8x1 centrada para que no se incluya ruido en la conversion 

        self.corr_OCEAN = [ #Matriz ordenada sacada del articulo "Correlation based data unification for personality trait prediction", la salida coincide con O C E A N
            [ 0.28, -0.32, -0.66,  0.64, -0.17,  0.13, -0.25,  0.26],
            [ 0.13, -0.13,  0.10, -0.13,  0.22, -0.27,  0.46, -0.46],
            [ 0.71, -0.72, -0.28,  0.27,  0.00,  0.00, -0.13,  0.16],
            [-0.02,  0.02,  0.01,  0.00, -0.41,  0.28,  0.05, -0.06],
            [-0.30,  0.31,  0.15, -0.14, -0.13,  0.12,  0.07, -0.07]
        ]

        self.corr_CINE = [
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
            [ 0.208, -0.044,  0.018,  0.045,  0.027], #Adventure
            [ 0.115, -0.012,  0.171,  0.206,  0.112]  #Musicals
        ]
        self.nombres_generos = ["Thriller", "Romance", "Western", "Comedy", "Action", "Drama", "Science Fiction", "Crime", "Horror", "Fantasy","Adventure", "Musical"]
        
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
        self.bases_juegos = np.array([43.740, 31.820, 39.573, 56.411, 54.150, 49.764, 44.918, 43.424, 50.746, 40.297, 25.799, 58.935])
        self.nombres_juegos = ["Shooting", "Action", "Fighting", "Turn-based", "Real Time Strategy", "RPG", "Sports", "Races", "Simulation/Sandbox", "Adventure", "Puzzles", "Multiplayer Online"]

        self.corr_MUSICA_1 = [
            #   O     C     E      A     N
            [-.002, .026, .103, .021, -.012], # R&B
            [-.019, -.017, .129, .008, -.049], #Rap
            [.077, -.029, .034, -.033, -.002], #Electronic
            [-.055, -.016, -.071, -.072, .057],  #Rock
            [.136, -.037, -.064,  -.032, 0], #Classical
            [.120, -.011, .023, -.011, -.044], #Blues
            [.106, -.049, -.002, .104, -.012],  #Country
            [.214, -.115, -.044, .104, .002], #Folk
            [.139, -.007, .042, .031, -.061], #Jazz
            [.002, -.061, -.020, .001, .030], #Punk
            [.115, -.104, -.031, .060, .101], #Alternative
            [-.034, .035, .056, .056, -.030], #Pop
            [-.031, -.023, -.076, -.069, -.001] #Heavy Metal
        ]
        self.nombres_musica_1 = ["R&B", "Rap", "Electronic", "Rock", "Classical", "Blues", "Country", "Folk", "Jazz", "Punk", "Alternative", "Pop", "Heavy Metal"]
        self.corr_MUSICA_2 = [
            #   O     C     E      A     N
            [.41, -.06, -.02, .03, .04], #Classical, Jazz, Blues, Folk, R&B (Reflective & Complex)
            [.15, -.03, .08, .01, -.01], #Alternative, Rock, Heavy Metal, Punk (Intense & Rebellious)
            [-.08, .18, .15, .24, -.04], #Country, Pop, Religious, Sound Tracks (Upbeat & Conventional)
            [.04, -.03, .19, .09, -.01]  #Rap & Hip-Hop, Soul & Funk, Electronic & Dance (Upbeat & Conventional)
        ]

        # Mapeo: Índice del género específico (corr_MUSICA_1) -> Índice del macrogénero (corr_MUSICA_2)
        # 0: Reflective/Complex, 1: Intense/Rebellious, 2: Upbeat/Conventional, 3: Energetic/Rhythmic
        self.mapeo_musica = {
            0: 0,   # R&B -> (Reflective & Complex)
            1: 3,   # Rap -> Energetic/Rhythmic
            2: 3,   # Electronic -> Energetic/Rhythmic
            3: 1,   # Rock -> Intense/Rebellious
            4: 0,   # Classical -> Reflective/Complex
            5: 0,   # Blues -> Reflective/Complex
            6: 2,   # Country -> Upbeat/Conventional
            7: 0,  # Folk -> Reflective/Complex
            8: 0,  # Jazz -> Reflective/Complex
            9: 1,  # Punk -> Intense/Rebellious
            10: 1,  # Alternative -> Intense/Rebellious
            11: 2,  # Pop -> Upbeat/Conventional
            12: 1   # Heavy Metal -> Intense/Rebellious
        }

        self.OCEAN = None
        
    
    def correlacionar_OCEAN(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de MBTI a OCEAN

            :return: Diccionario con las % normalizadas para cada dimension de OCEAN
        '''
        res = self.corr_OCEAN @ self.prediccion #Multiplicacion de matrices

        #Normailzacion sigmoide
        t = 4 # Ajusta este valor empíricamente
        self.OCEAN = 1 / (1 + np.exp(-res * t))

        return dict(
                zip(
                    ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'], 
                    [float(v * 100) for v in self.OCEAN.reshape(1, -1)[0]]
                )
            )
    

    def correlacionar_GENEROS(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de OCEAN a GENEROS PELICULAS/SERIES
            
            :return: Diccionario con los % normalizados para cada género de peliculas/series
        '''

        if(self.OCEAN is None): #Se debe de tener el array OCEAN para continuar
            self.correlacionar_OCEAN()
        
        ocean_centrado = (self.OCEAN.flatten() - 0.5) * 2

        res = self.corr_CINE @ ocean_centrado

        temp_cine = 4.5  
        bias_cine = -.2  
        res = 1 / (1 + np.exp(-(res * temp_cine + bias_cine)))
        
        return dict(zip(self.nombres_generos, [float(np.round(x * 100, 2)) for x in res]))
    
    def correlacionar_JUEGOS(self) -> dict:
        '''
            Funcion que genera automaticamente la correlacion de OCEAN a GENEROS de VIDEOJUEGOS
            
            :return: Diccionario con los % normalizados para cada género de videojuegos
        '''
        if self.OCEAN is None:
            self.correlacionar_OCEAN()
            
        # Regresión Lineal: Base + (Matriz @ OCEAN_en_escala_100)
        ocean_100 = self.OCEAN.flatten() * 100
        amplificador = .8
        res = ((self.corr_JUEGOS @ ocean_100) * amplificador) + self.bases_juegos
        
        # Recortamos los valores para que el score nunca sobrepase 100 ni sea negativo
        res = np.clip(res, 0, 100)
        return dict(
                zip(
                    self.nombres_juegos,
                    [float(x) for x in np.round(res,2)])
                )


    def correlacionar_MUSICA(self) -> dict:
        if self.OCEAN is None:
            self.correlacionar_OCEAN()
            
        ocean_centrado = (self.OCEAN.flatten() - 0.5) * 2
        
        # 1. Calculamos las puntuaciones base (crudos, antes de la sigmoide)
        res_especifica = self.corr_MUSICA_1 @ ocean_centrado
        res_general = self.corr_MUSICA_2 @ ocean_centrado
        
        # 2. Sumamos la puntuación de apoyo (macrogénero) al género específico
        res_final = np.zeros_like(res_especifica)
        
        for i, macro_idx in self.mapeo_musica.items():
            # Sumamos: OCEAN x Género Específico + OCEAN x Macrogénero
            res_final[i] = res_especifica[i] + res_general[macro_idx]
                
                
        temp_musica = 4.5
        bias_musica = -.2 
        res_final = 1 / (1 + np.exp(-(res_final * temp_musica + bias_musica)))
        
        return dict(zip(self.nombres_musica_1, [float(np.round(x * 100, 2)) for x in res_final]))


if __name__ == "__main__":
    from pipeline import Pipeline
    texto = '''
    I honestly can't stand those endless corporate strategy meetings where everyone just talks in circles about abstract concepts, 'synergy', and long-term visions. Just give me the broken backend architecture and leave me alone in my zone for a few hours. 
    I don't need a detailed roadmap or a strict schedule to get things done; 
    I prefer to dive in, take it apart, figure out exactly why the database is crashing, and build a practical fix on the fly. When I'm not working, I'm usually in my garage tinkering with my motorcycle. 
    People think I'm antisocial because I avoid networking events, but I just prefer interacting with mechanical or code systems that actually make logical sense rather than dealing with office politics.
    '''

    lista_modelos = ["Modelos Definitivos\\XGBoost\\E-I_XGBoost_calibrado.joblib",
                     "Modelos Definitivos\\Linear_SVC\\S-N_Linear_SVC_calibrado.joblib",
                     "Modelos Definitivos\\Linear_SVC\\T-F_Linear_SVC_calibrado.joblib",
                     "Modelos Definitivos\\Regresion_Logistica\\J-P_Regresion_Logistica_calibrado.joblib"]
    
    lista_encoders = ["robertaFT\\E-I_roberta-base","robertaFT\\S-N_roberta-base","robertaFT\\T-F_roberta-base","robertaFT\\J-P_roberta-base"]
    predictor = Pipeline(texto, lista_modelos, lista_encoders)

    probabilidades, perfil = predictor.generar_predicciones()

    correlacionador = Correlacionador(prediccion=probabilidades)
    ocean = correlacionador.correlacionar_OCEAN()
    print(ocean)
    cine = correlacionador.correlacionar_GENEROS()
    print(cine)
    musica = correlacionador.correlacionar_MUSICA()
    print(musica)
    videojuegos = correlacionador.correlacionar_JUEGOS()
    print(videojuegos)