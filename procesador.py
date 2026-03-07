import os.path
from tqdm import tqdm
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch

#Esta clase se encarga de limpiar, tokenizar y realizar el embedding del dataset a partir de un modelo pre-entrenado
class Preprocesador:
    def __init__(self, nomCarpeta, nombre_modelo, max_length=512, dispotivo='cuda', columna="posts"):
        
        print(f"[INFO] Cargando dataset desde: {nomCarpeta}")
        self.nomCarpeta = nomCarpeta
        self.dataset = pd.read_csv(os.path.join(nomCarpeta, "MBTI.csv"))
        self.nombre_modelo = nombre_modelo
        self.columna = columna

        print(f"[INFO] Cargando tokenizador: {nombre_modelo}")
        self.tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
        self.max_length = max_length

        print(f"[INFO] Cargando modelo: {nombre_modelo}")
        self.modelo = AutoModel.from_pretrained(nombre_modelo).to(dispotivo)
        self.modelo.eval()  # Establece el modelo en modo evaluación para evitar dropout y otros comportamientos de entrenamiento

        self.patron_ruido = r"\|{3}|https?://\S+|www\.\S+|~|:\S+:"
        self.patron_mbti = r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b" # \b asegura que solo borre la palabra exacta

    def limpiar_texto(self, post):
        if not isinstance(post, str): # Si esta vacio (Null)
            return ""

        #LIMPIEZA CON REGEX
        post_limpio = re.sub(self.patron_ruido, " ", post)
        post_limpio = re.sub(self.patron_mbti, "", post_limpio, flags=re.IGNORECASE)#Ignoramos las mayusculas
        post_limpio = re.sub(r"\s+", " ", post_limpio).strip()#Quito espacios de más

        return post_limpio

    def reformular_clases(self):
        #Esta función se encarga de, a partir de las 16 personalidades, generar 4 clases binarias (E/I, S/N, T/F, J/P) para cada una de las dimensiones del MBTI.
        self.dataset["E/I"] = self.dataset["type"].apply(lambda x: 0 if x[0] == "E" else 1)
        self.dataset["S/N"] = self.dataset["type"].apply(lambda x: 0 if x[1] == "S" else 1)
        self.dataset["T/F"] = self.dataset["type"].apply(lambda x: 0 if x[2] == "T" else 1)
        self.dataset["J/P"] = self.dataset["type"].apply(lambda x: 0 if x[3] == "J" else 1)

    def tokenizar_texto(self, texto):
        tokens = self.tokenizer(
            texto,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_overflowing_tokens=True,
            stride=50,
            padding="max_length",
            return_tensors="pt"  # Devuelve tensores de PyTorch
        )
        return tokens


    def aplicarMeanPooling(self, token_embeddings, attention_mask):
        #Cada post se ha dividido en fragmentos de 512 tokens, por lo que tenemos un tensor de [num_fragmentos, 512, 768]
        #512 Hace referencia al numero maximo de tokens que el modelo puede procesar, y 768 es la dimensión del embedding de cada token
        #La dimension del attention_mask es [num_fragmentos, 512]. Tiene un 1 para cada token real y un 0 para cada token de relleno (padding)
        #Para poder hacer la media de los vectores de cada fragmento, necesitamos expandir la mascara de atención para que tenga la misma dimensión que los embeddings de los tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #Aqui ampliamos la mascara de atención para que tenga la misma dimensión que los embeddings de los tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Multiplicamos los embeddings de los tokens por la mascara de atención expandida para que los tokens de relleno no contribuyan a la suma
        #La suma que realizamos es en la dimension de los tokens, por lo que obtenemos un tensor de [num_fragmentos, 768] que es la suma de los embeddings de los tokens reales de cada fragmento
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) #Averiguamos el total de tokens reales (no padding)
        return sum_embeddings / sum_mask #Aplicamos la media dividiendo la suma de los embeddings por el total de tokens reales para obtener un vector de embedding representativo de cada fragmento

    def extraer_embeddings(self, tokens):
        #Recuperamos los tensores de input_ids y attention_mask del dataset y los subimos a la VRAM
        input_ids = tokens['input_ids'].to(self.modelo.device)
        attention_mask = tokens['attention_mask'].to(self.modelo.device)

        with torch.no_grad(): #Solo estamos usando el modelo para inferencia, no necesitamos calcular gradientes
            salida = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
        
        token_embedding = salida.last_hidden_state #Recuperamos la salida de la última capa oculta (embedding de cada token)
        
        chunk_embeddings = self.aplicarMeanPooling(token_embedding, attention_mask) #Aplicamos mean pooling a nivel de fragmento para obtener un vector por cada fragmento

        embedding_final = torch.mean(chunk_embeddings, dim=0) #Hacemos la media de los vectores de cada fragmento para obtener un solo vector que represente todo el post

        return embedding_final.cpu().numpy().flatten() #Bajamos el vector de la VRAM a la CPU y lo convertimos de una matriz (1, 768) a un vector (768,)

    def procesar_dataset(self):
        #Esta funcion se encarga de, a partir del texto del dataset original, generar y guardar el embedding en una nueva columna llamada "embedding"

        #Paso 1: Limpiar el texto
        self.dataset[self.columna] = self.dataset[self.columna].apply(self.limpiar_texto)
        print("[INFO] ¡Limpieza completada!")

        # Paso 2: Tokenización y Embedding (Pesado - GPU)
        print(f"[INFO] Procesando embeddings con {self.nombre_modelo}...")
        
        embeddings_finales = []
        
        for texto in tqdm(self.dataset[self.columna], desc="Extrayendo Embeddings"): #Proceesamos cada post uno por uno. Tambien visualizamos el progreso con tqdm
            tokens = self.tokenizar_texto(texto)
            vector = self.extraer_embeddings(tokens)
            embeddings_finales.append(vector)

        # Paso 3: Preparar DataFrame final
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
        self.guardar_dataset(df_final)

    def guardar_dataset(self, dataset):
        modelo_limpio = self.nombre_modelo.replace("/", "_")
        ruta_archivo = os.path.join(self.nomCarpeta, f"{modelo_limpio}_dataset.parquet")
        dataset.to_parquet(ruta_archivo, engine="pyarrow")
        print(f"[EXITO] Dataset guardado en: {ruta_archivo}")



def ejecutar_preprocesador(preprocesador:Preprocesador):
    print("[EJECUCION] Ejecutando preprocesador ...]")

    preprocesador.procesar_dataset()

    print("[EJECUCION] Fin preprocesador ...]")


def main():
    
    print("[EJECUCION] Ejecutando ROBERTA BASE...]")
    robertaBase = Preprocesador("dataset9K","FacebookAI/roberta-base")
    robertaBase.reformular_clases()
    robertaBase.guardar_dataset(robertaBase.dataset)
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
    '''
if __name__ == "__main__":
    main()