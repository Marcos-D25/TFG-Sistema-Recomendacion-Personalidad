import os.path
from tqdm import tqdm
import pandas as pd
import re
from numpy import ndarray
from transformers import AutoTokenizer, AutoModel, BatchEncoding
import torch

#Esta clase se encarga de limpiar, tokenizar y realizar el embedding del dataset a partir de un modelo pre-entrenado
class Preprocesador:
    def __init__(self, nombre_modelo,max_lenght = 512, dispositivo='cuda'):
        '''
        Constructor del Preprocesador, que se encargará de realizar las tareas de limpieza, tokenizacion y embedding

        :param nombre_modelo: Nombre del modelo completo. (Debe aparecer exactamente igual que aparece en Huggingface)
        :param max_lenght: Predefinido a 512 ya que los modelos de Roberta a usar comparten el mismo tamaño maximo de entrada
        :param dispositivo: Predefinido a 'cuda' si quiere entrenar el modelo usando una tarjeta grafica compatible. Tambien se puede usar 'cpu'
        '''
        self.nombre_modelo = nombre_modelo
        self.dataset = None
        print(f"[INFO] Cargando tokenizador: {nombre_modelo}")
        self.tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

        print(f"[INFO] Cargando modelo: {nombre_modelo}")
        self.modelo = AutoModel.from_pretrained(nombre_modelo).to(dispositivo)
        self.max_lenght = max_lenght
        self.modelo.eval()  # Establece el modelo en modo evaluación para evitar dropout y otros comportamientos de entrenamiento

        self.patron_ruido = r"\|{3}|https?://\S+|www\.\S+|~|:\S+:"
        self.patron_mbti = r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b" # \b asegura que solo borre la palabra exacta

    def limpiar_texto(self, texto) -> str:
        '''
        Dado un texto, se eliminan todos los enlaces, emoticonos y cualquier otro patron de ruido

        :param texto: Texto a limpiar
        :return: Texto limpio
        '''
        if not isinstance(texto, str): # Si esta vacio (Null)
            return ""

        #LIMPIEZA CON REGEX
        post_limpio = re.sub(self.patron_ruido, " ", texto)
        post_limpio = re.sub(self.patron_mbti, "", post_limpio, flags=re.IGNORECASE)#Ignoramos las mayusculas
        post_limpio = re.sub(r"\s+", " ", post_limpio).strip()#Quito espacios de más

        return post_limpio

    def reformular_clases(self):
        '''
        Esta funcion sirve para que, a partir del dataset cargado previamente con cargar_dataset(), se creen 4 nuevas columnas en el dataset, cada una correspondiente a cada dimension de la personalidad.
        El valor es binario para cada dimension.
        Si el dataset no esta cargado lanza un error

        :return: None
        '''
        if self.dataset is None:
            raise Exception("[ERROR]Dataset no cargado en la clase.")

        self.dataset["E/I"] = self.dataset["type"].apply(lambda x: 0 if x[0] == "E" else 1)
        self.dataset["S/N"] = self.dataset["type"].apply(lambda x: 0 if x[1] == "S" else 1)
        self.dataset["T/F"] = self.dataset["type"].apply(lambda x: 0 if x[2] == "T" else 1)
        self.dataset["J/P"] = self.dataset["type"].apply(lambda x: 0 if x[3] == "J" else 1)

    def tokenizar_texto(self, texto) -> BatchEncoding :
        '''
        Tokeniza el texto pasado como parametro usando el tokenizador cargado acorde con el modelo

        :param texto: Texto a tokenizar
        :return: Devuelve un diccionario especial de huggingface donde contiene {'inputs_ids':  ..., 'attention_mask': ..., 'overflow_to_sample_mapping': ...}
        '''
        tokens = self.tokenizer(
            texto,
            add_special_tokens=True,
            max_length=self.max_lenght,
            truncation=True,
            return_overflowing_tokens=True,
            stride=50,
            padding="max_length",
            return_tensors="pt"  # Devuelve tensores de PyTorch
        )
        return tokens

    def cargar_dataset(self, nombreCarpeta="dataset9K", nombreArchivo="MBTI.csv", columna="posts")-> None:
        '''
        Funcion que carga en la clase el dataset a limpiar.

        :param nombreCarpeta: Nombre de la carpeta local en la que se encuentra el dataset a preprocesar
        :param nombreArchivo: Nombre del archivo que contiene el csv
        :param columna: Hace referencia al nombre de la columna que contiene los textos a procesar
        :return: None
        '''

        self.nomCarpeta = nombreCarpeta
        self.dataset = pd.read_csv(os.path.join(nombreCarpeta, nombreArchivo))
        self.columna = columna

    def aplicarMeanPooling(self, token_embeddings, attention_mask) -> torch.Tensor:
        '''
        Funcion que aplica la tecnica de mean pooling a los embeddings pasados como parametro

        :param token_embedding: Matriz del embedding
        :param attention_mask: Matriz que indica que embedding hace referencia a relleno y cual no
        :return: Tensor
        '''

        #Cada post se ha dividido en fragmentos de 512 tokens, por lo que tenemos un tensor de [num_fragmentos, 512, 768]
        #512 Hace referencia al numero maximo de tokens que el modelo puede procesar, y 768 es la dimensión del embedding de cada token
        #La dimension del attention_mask es [num_fragmentos, 512]. Tiene un 1 para cada token real y un 0 para cada token de relleno (padding)
        #Para poder hacer la media de los vectores de cada fragmento, necesitamos expandir la mascara de atención para que tenga la misma dimensión que los embeddings de los tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #Aqui ampliamos la mascara de atención para que tenga la misma dimensión que los embeddings de los tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Multiplicamos los embeddings de los tokens por la mascara de atención expandida para que los tokens de relleno no contribuyan a la suma
        #La suma que realizamos es en la dimension de los tokens, por lo que obtenemos un tensor de [num_fragmentos, 768] que es la suma de los embeddings de los tokens reales de cada fragmento
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) #Averiguamos el total de tokens reales (no padding)
        return sum_embeddings / sum_mask #Aplicamos la media dividiendo la suma de los embeddings por el total de tokens reales para obtener un vector de embedding representativo de cada fragmento

    def extraer_embeddings(self, tokens) -> ndarray:
        '''
        Funcion que realiza tanto el embedding como el mean pooling a los tokens pasados como parametros

        :param tokens: Corresponde con los tokens computados del texto que se quiere procesar
        :return: Array correspondiente con el embedding final del token
        '''

        #Recuperamos los tensores de input_ids y attention_mask del dataset y los subimos a la VRAM
        input_ids = tokens['input_ids'].to(self.modelo.device)
        attention_mask = tokens['attention_mask'].to(self.modelo.device)

        with torch.no_grad(): #Solo estamos usando el modelo para inferencia, no necesitamos calcular gradientes
            salida = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
        
        token_embedding = salida.last_hidden_state #Recuperamos la salida de la última capa oculta (embedding de cada token)
        
        chunk_embeddings = self.aplicarMeanPooling(token_embedding, attention_mask) #Aplicamos mean pooling a nivel de fragmento para obtener un vector por cada fragmento

        embedding_final = torch.mean(chunk_embeddings, dim=0) #Hacemos la media de los vectores de cada fragmento para obtener un solo vector que represente todo el post

        return embedding_final.cpu().numpy().flatten() #Bajamos el vector de la VRAM a la CPU y lo convertimos de una matriz (1, 768) a un vector (768,)

    def procesar_dataset(self, carpeta_dest="dataset9K", nom_archivo="MBTI_procesado") -> None:
        '''
        Esta funcion se encarga de, a partir del texto del dataset original, generar y guardar el embedding en una nueva columna llamada "embedding".
        Lanzará un error si previamente no se ha cargado en la clase el dataset

        :param carpeta_dest: Carpeta en la que se guardará el dataset
        :param nom_archivo: Nombre con el que se guardará el dataset parquet
        :return: None
        '''

        if self.dataset is None:
            raise Exception("[ERROR]Dataset no cargado en la clase.")

        self.dataset[self.columna] = self.dataset[self.columna].apply(self.limpiar_texto)
        print("[INFO] ¡Limpieza completada!")

        print(f"[INFO] Procesando embeddings con {self.nombre_modelo}...")
        
        embeddings_finales = []
        
        for texto in tqdm(self.dataset[self.columna], desc="Extrayendo Embeddings"): #Proceesamos cada post uno por uno. Tambien visualizamos el progreso con tqdm
            tokens = self.tokenizar_texto(texto)
            vector = self.extraer_embeddings(tokens)
            embeddings_finales.append(vector)

        print("[INFO] Consolidando resultados...")
        self.reformular_clases() #Generamos las clases binarias para cada dimensión del MBTI
        df_final = pd.DataFrame({
            "Posts": self.dataset[self.columna].tolist(),
            "Embedding": [emb.tolist() for emb in embeddings_finales],
            "MBTI": self.dataset["type"].tolist(),
            "E/I": self.dataset["E/I"].tolist(),
            "S/N": self.dataset["S/N"].tolist(),
            "T/F": self.dataset["T/F"].tolist(),
            "J/P": self.dataset["J/P"].tolist()
        })

        if not os.path.exists(carpeta_dest):
            os.mkdir(carpeta_dest)
        self.guardar_dataset_parquet(df_final, carpeta_dest, nom_archivo)

    def procesar_texto(self, texto) -> ndarray:
        '''
        Funcion que sirve para procesar un simple texto. En esta funcion se limpiará, tokenizará y generará el embedding correspondiente

        :param texto: Texto a procesar
        :return: Vector ndarray con el embedding 
        '''
        print("[INFO] Realizando el embedding del texto...")

        texto_limpio = self.limpiar_texto(texto)
        texto_tokenizado = self.tokenizar_texto(texto_limpio)
        res = self.extraer_embeddings(texto_tokenizado)

        print("[INFO] Embedding generado")
        return res


    def guardar_dataset_parquet(self, dataset, carpeta_dest, nom_archivo):
        '''
        Funcion que guarda en un archivo .parquet el dataset pasado como parametro

        :param dataset: Dataset a guardar
        :param carpeta_dest: Carpeta en la que se guardará el dataset
        :param nom_archivo: Nombre que obtendrá el dataset guardado 
        '''
        ruta_archivo = os.path.join(carpeta_dest, f"{nom_archivo}.parquet")
        dataset.to_parquet(ruta_archivo, engine="pyarrow")
        print(f"[EXITO] Dataset guardado en: {ruta_archivo}")

    def guardar_dataset_csv(self, dataset, ruta):
        '''
        Funcion que guarda en un archivo .csv el dataset pasado como parametro

        :param dataset: Dataset a guardar
        :param ruta: Ruta en la que se guarda el dataset
        '''
        dataset.to_csv(ruta, index=False, encoding='utf-8')
        print(f"[EXITO] Dataset guardado en: {ruta}")



def ejecutar_preprocesador(preprocesador:Preprocesador):
    print("[EJECUCION] Ejecutando preprocesador ...]")
    preprocesador.cargar_dataset()
    preprocesador.procesar_dataset()

    print("[EJECUCION] Fin preprocesador ...]")

def main():
    
    print("[EJECUCION] Ejecutando ROBERTA BASE...]")
    robertaBase = Preprocesador("FacebookAI/roberta-base")
    robertaBase.cargar_dataset(nombreCarpeta="dataset100K")
    robertaBase.reformular_clases()
    robertaBase.dataset["posts"] = robertaBase.dataset["posts"].apply(robertaBase.limpiar_texto)

    robertaBase.guardar_dataset_csv(robertaBase.dataset, os.path.join("dataset100K", "MBTI_limpio.csv"))
    
    print("[EJECUCION] ROBERTA BASE guardado]\n\n")
    '''
    print("[EJECUCION] Ejecutando XML ROBERTA BASE...]")
    xml_robertaBase = Preprocesador(os.path.join("datasets","MBTI_sinProcesar.csv"),"FacebookAI/xlm-roberta-base")
    ejecutar_preprocesador(xml_robertaBase)
    print("[EJECUCION] XML ROBERTA BASE guardado]\n\n")

    print("[EJECUCION] Ejecutando XML ROBERTA LARGE...]")
    xml_robertaLarge = Preprocesador(os.path.join("datasets","MBTI_sinProcesar.csv"),"FacebookAI/xlm-roberta-large")
    ejecutar_preprocesador(xml_robertaLarge)
    print("[EJECUCION] XML ROBERTA LARGE guardado]\n\n")
    

    print("[EJECUCION] Ejecutando ROBERTA BASE FT...")
    robertaFTEI = Preprocesador("./robertaFT/E-I_roberta-base")
    robertaFTEI.cargar_dataset()
    robertaFTEI.procesar_dataset(carpeta_dest="datasetRBFT", nom_archivo="datasetEI")

    robertaFTJP = Preprocesador("./robertaFT/J-P_roberta-base")
    robertaFTJP.cargar_dataset()
    robertaFTJP.procesar_dataset(carpeta_dest="datasetRBFT", nom_archivo="datasetJP")

    robertaFTSN = Preprocesador("./robertaFT/S-N_roberta-base")
    robertaFTSN.cargar_dataset()
    robertaFTSN.procesar_dataset(carpeta_dest="datasetRBFT", nom_archivo="datasetSN")

    robertaFTTF = Preprocesador("./robertaFT/T-F_roberta-base")
    robertaFTTF.cargar_dataset()
    robertaFTTF.procesar_dataset(carpeta_dest="datasetRBFT", nom_archivo="datasetTF")

    print("[EJECUCION] Fin ROBERTA BASE FT...")
    '''

if __name__ == "__main__":
    main()