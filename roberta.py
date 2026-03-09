import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class FineTunnerRoberta:
    def __init__(self, modelo_base="FacebookAI/roberta-base", nom_carpeta="dataset9K", nom_archivo="MBTI.csv"):
        '''
        Constructor de la clase encargada de realizar fine tunning al modelo de roberta deseado

        :param modelo_base: Modelo de Roberta al que se desea aplicar el fine tunning
        :param nom_carpeta: Nombre de la carpeta donde se encuentra el dataset
        :param nom_archivo: Nombre del archivo que corresponde con el dataset csv
        '''


        self.modelo_base = modelo_base
        self.dataset = pd.read_csv(os.path.join(nom_carpeta,nom_archivo))
        self.train_dataset = {}
        self.val_dataset = {}
        # El modelo se guardará en una carpeta específica para no machacar a los demás
        self.salida_modelo = None  # Se asignará para cada dimensión 
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_base)

    def preparar_datos(self, df_train, df_val, columna_texto="posts", dimension="E/I") -> None:
        '''
        Funcion que se encarga de realizar la tokenizacion a los datasets de entreno y validacion. 
        Guarda los datasets en los respectivos diccionarios de la clase train_dataset y val_dataset

        :param df_train: Dataset destinado al entreno del fine tunning
        :param df_val: Dataset destinado a la validación del fine tunning
        :param columna_texto: Nombre de la columna donde contiene los posts/textos a procesar
        :param dimension: Dimension correspondiente al entreno de este modelo (ej. E/I)
        '''
        df_train_sub = df_train[[columna_texto, dimension]].rename(columns={dimension: 'label'})
        df_val_sub = df_val[[columna_texto, dimension]].rename(columns={dimension: 'label'})
        
        ds_train = Dataset.from_pandas(df_train_sub)
        ds_val = Dataset.from_pandas(df_val_sub)

        def tokenizar(batch):
            return self.tokenizer(batch[columna_texto], padding="max_length", truncation=True, max_length=512)

        self.train_dataset[dimension] = ds_train.map(tokenizar, batched=True)
        self.val_dataset[dimension] = ds_val.map(tokenizar, batched=True)

    def entrenar_dimension(self, dimension="E/I"):
        '''
        Funcion que se encarga de entrenar una dimension concreta del dataset, creando un modelo especializado en esa dimension

        :param dimension: Dimension a entrenar (ej. E/I)
        '''
        self.salida_modelo = f"./robertaFT/{dimension.replace('/', '-')}_{self.modelo_base.split('/')[-1]}"
        
        print(f"[INFO] Iniciando Fine-Tuning BINARIO de {self.modelo_base} para la dimension {dimension}...")
        
        
        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.modelo_base, 
            num_labels=2 #num_labels=2 obliga a RoBERTa a ser un clasificador binario puro
        ).to('cuda')

        training_args = TrainingArguments(
            output_dir=self.salida_modelo,
            eval_strategy="epoch",
            #save_strategy="epoch", #Guardamos "instancias" del modelo cada cierto tiempo, se pueden eliminar una vez termina el entrenamiento
            learning_rate=2e-5, #Uso de learning rates bajos para no descontrolar demasiado los pesos

            per_device_train_batch_size=16, 
            per_device_eval_batch_size=16,  
            dataloader_num_workers=2,  #Indica el numero de nucleos de la cpu que alimenta a la grafica     
            num_train_epochs=3, #Incrementar el numero aumenta el riesgo de overfitting
            weight_decay=0.01,
            fp16=True,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=modelo,
            args=training_args,
            train_dataset=self.train_dataset[dimension],
            eval_dataset=self.val_dataset[dimension],
        )

        trainer.train()

        print(f"[EXITO] Guardando RoBERTa experta en {dimension} en: {self.salida_modelo}")
        trainer.save_model(self.salida_modelo)
        self.tokenizer.save_pretrained(self.salida_modelo)
    
    def entrenar_modelo(self):
        '''
        Funcion que sirve para entrenar a Roberta en todas las dimensiones que nos concierne
        '''
        
        df_train, df_val = train_test_split(
            self.dataset, 
            test_size=0.3,  #30% validación
            random_state=42  
        )        
        # Dimensiones a entrenar
        dimensiones = ["E/I", "S/N", "T/F", "J/P"]
        
        # Entrenar cada dimensión
        for dimension in dimensiones:
            print(f"\n[INFO] Entrenando dimensión {dimension}...")
            self.preparar_datos(df_train, df_val, columna_texto="posts", dimension=dimension)
            self.entrenar_dimension(dimension=dimension)
        
        print(f"\n[EXITO] Entrenamiento completado en todas las dimensiones")
        
def main():
    entrenador = FineTunnerRoberta(nom_archivo="MBTI_limpio.csv")
    entrenador.entrenar_modelo()

if __name__ == "__main__":
    main()