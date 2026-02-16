import os.path

import pandas as pd
import re
from transformers import AutoTokenizer

class Procesador:
    def __init__(self, nomArchivo, nombre_modelo, max_length=512):
        print(f"[INFO] Cargando dataset desde: {nomArchivo}")
        self.dataset = pd.read_csv(nomArchivo)

        print(f"[INFO] Cargando tokenizador: {nombre_modelo}")
        self.tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
        self.max_length = max_length

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

    def preprocesar_dataset(self, columna):
        self.dataset["posts"] = self.dataset[columna].apply(self.limpiar_texto)
        print("[INFO] ¡Limpieza completada!")
        return self.dataset

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

    def procesar_dataset(self, columna):
        self.dataset = self.dataset[columna].apply(self.tokenizar_texto)

    def guardar_dataset(self, direccion):
        self.dataset.to_csv(direccion, index=False)

def ejecutar_procesador(procesador:Procesador, nombre_modelo:str):
    print("[EJECUCION] Ejecutando procesador ...]")
    procesador.preprocesar_dataset("posts")#Funcion de limpieza
    print("[EJECUCION] Funcion limpieza hecha ...]")
    procesador.procesar_dataset("posts")#Funcion de tokenizacion
    print("[EJECUCION] Funcion tokenizacion hecha ...]")
    procesador.guardar_dataset(os.path.join("datasets",nombre_modelo))
    print("[EJECUCION] Dataset guardado ...]")

def main():
    print("[EJECUCION] Ejecutando ROBERTA BASE...]")
    robertaBase = Procesador(os.path.join("datasets","MBTI_sinProcesar.csv"),"FacebookAI/roberta-base")
    ejecutar_procesador(robertaBase, "roberta-base")
    print("[EJECUCION] ROBERTA BASE guardado]\n\n")

    print("[EJECUCION] Ejecutando XML ROBERTA BASE...]")
    xml_robertaBase = Procesador(os.path.join("datasets","MBTI_sinProcesar.csv"),"FacebookAI/xlm-roberta-base")
    ejecutar_procesador(xml_robertaBase, "xml-roberta-base")
    print("[EJECUCION] XML ROBERTA BASE guardado]\n\n")

    print("[EJECUCION] Ejecutando XML ROBERTA LARGE...]")
    xml_robertaLarge = Procesador(os.path.join("datasets","MBTI_sinProcesar.csv"),"FacebookAI/xlm-roberta-large")
    ejecutar_procesador(xml_robertaLarge, "xml-roberta-large")
    print("[EJECUCION] XML ROBERTA LARGE guardado]\n\n")

if __name__ == "__main__":
    main()