import ast
import pandas as pd

class Recomendador:
    def __init__(self):
        self.dataset_series = pd.read_csv('dataset_series.csv')
        self.dataset_peliculas = pd.read_csv('dataset_peliculas.csv')
        self.dataset_musica = pd.read_csv('dataset_musica.csv')
        self.dataset_videojuegos = pd.read_csv('dataset_juegos.csv')
        self.PESO_AFINIDAD = 0.60
        self.PESO_SCORE = 0.40
    def get_top_puntuaciones(self, tipo_contenido, top_n=5) -> list:
        '''
            Funcion que dado un diccionario de puntuaciones de géneros, devuelve el top N de géneros ordenados por puntuación.

            :param tipo_contenido: Diccionario con los géneros y sus puntuaciones
            :param top_n: Número de géneros a devolver
            :return: Lista de los top N géneros ordenados por puntuación
        '''
        return sorted(tipo_contenido, key=tipo_contenido.get, reverse=True)[:top_n]
    
    def calcular_afinidad(self, generos_juego, top_usuario):
        '''
            Función que calcula la afinidad entre los géneros de un juego y el top N de géneros del usuario.

            :param generos_juego: Lista de géneros del juego (puede ser una cadena que representa una lista)
            :param top_usuario: Lista de los top N géneros del usuario
            :return: Número de coincidencias entre los géneros del juego y el top N del usuario
        '''
        if isinstance(generos_juego, str):
            try:
                generos_juego = ast.literal_eval(generos_juego)
            except:
                generos_juego = []
                
        # Contamos cuántos géneros del juego coinciden con el Top N del usuario
        coincidencias = set(generos_juego).intersection(set(top_usuario))
        return len(coincidencias)

    def recomendar(self, tipo_contenido:str, puntuaciones_usuario:dict, top_n=5):
        """
            Funcion principal que genera recomendaciones basadas en las puntuaciones del usuario para un tipo de contenido específico.

            :param tipo_contenido: Tipo de contenido para el que se quieren generar recomendaciones ('videojuegos', 'series', 'peliculas' o 'musica')
            :param puntuaciones_usuario: Diccionario con los géneros y sus puntuaciones por parte del usuario
            :param top_n: Número de géneros principales del usuario a considerar para la afinidad
        """
        top_3_generos = self.get_top_puntuaciones(puntuaciones_usuario, top_n)

        match tipo_contenido:
            case 'videojuegos': 
                df = self.dataset_videojuegos
                col = None
            case 'series': 
                df = self.dataset_series
                col = 'overview'  
            case 'peliculas': 
                df = self.dataset_peliculas
                col = 'overview'
            case 'musica': 
                df = self.dataset_musica
                col = 'Artist'
            case _: 
                raise ValueError("Tipo de contenido no válido. Elige entre 'videojuegos', 'series', 'peliculas' o 'musica'.")
        # 3.1 Calcular la afinidad bruta (0 a 3)
        df['Afinidad_Bruta'] = df['genre'].apply(lambda x: self.calcular_afinidad(x, top_3_generos))

        # 3.2 Filtrar la basura: Solo queremos juegos que tengan al menos 1 coincidencia
        df_recomendados = df[df['Afinidad_Bruta'] > 0].copy()

        # 3.3 Normalizar la Afinidad a escala 0-100 (El máximo es 3 coincidencias)
        df_recomendados['Afinidad_Norm'] = (df_recomendados['Afinidad_Bruta'] / 3) * 100

        df_recomendados['Puntuacion_Final'] = (df_recomendados['Afinidad_Norm'] * self.PESO_AFINIDAD) + (df_recomendados['score'] * self.PESO_SCORE)

        # 3.5 Ordenar por la Puntuación Final de mayor a menor y sacar el Top 5
        df_recomendados = df_recomendados.sort_values(by='Puntuacion_Final', ascending=False).head(5)

        self.mostrar_recomendaciones(df_recomendados, tipo_contenido, col)

    
    def mostrar_recomendaciones(self, df_recomendados, tipo_contenido:str, col:str = None):
        """
            Función que muestra las recomendaciones.
            :param df_recomendados: DataFrame con las recomendaciones ya ordenadas por puntuación final
            :param tipo_contenido: Tipo de contenido para el que se están mostrando las recomendaciones (musica, series, peliculas o videojuegos)
            :param col: Columna adicional a mostrar (overview para series y películas, Artist para música)
        """
        print(f"\nRECOMENDACIONES DE {tipo_contenido.upper()}:")
        for index, fila in df_recomendados.iterrows():
            print(f"- {fila['name']}")
            if col and col in fila:
                print(f"  └ {col.capitalize()}: {fila[col]}")
            print(f"  └ Géneros: {fila['genre']} | Coincidencias: {fila['Afinidad_Bruta']}/3")
            print(f"  └ Nota: {fila['score']}/100 | Match Total: {fila['Puntuacion_Final']:.1f} pts\n")


    

