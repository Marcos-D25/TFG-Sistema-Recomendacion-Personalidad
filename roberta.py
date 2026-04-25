import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import warnings

#SILENCIAR WARNINGS
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Movemos los pesos a la misma tarjeta gráfica que el modelo
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Aplicamos la función de pérdida con nuestros pesos calculados
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Usamos macro para vigilar ambas clases
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

class FineTunnerRoberta:
    def __init__(self, modelo_base="FacebookAI/roberta-base", nom_carpeta="dataset9K", nom_archivo="MBTI.csv"):
        '''
        Constructor de la clase encargada de realizar fine tunning al modelo de roberta deseado.

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
        df_train_sub = df_train[[columna_texto, dimension]].dropna(subset=[columna_texto])
        df_train_sub[columna_texto] = df_train_sub[columna_texto].astype(str)
        df_train_sub = df_train_sub.rename(columns={dimension: 'label'})
        
        df_val_sub = df_val[[columna_texto, dimension]].dropna(subset=[columna_texto])
        df_val_sub[columna_texto] = df_val_sub[columna_texto].astype(str)
        df_val_sub = df_val_sub.rename(columns={dimension: 'label'})
        
        ds_train = Dataset.from_pandas(df_train_sub)
        ds_val = Dataset.from_pandas(df_val_sub)

        def tokenizar(batch):
            return self.tokenizer(batch[columna_texto], padding="max_length", truncation=True, max_length=512)

        self.train_dataset[dimension] = ds_train.map(tokenizar, batched=True)
        self.val_dataset[dimension] = ds_val.map(tokenizar, batched=True)

    def entrenar_dimension(self, dimension="E/I",nomCarpeta="robertaFT"):
        '''
        Funcion que se encarga de entrenar una dimension concreta del dataset, creando un modelo especializado en esa dimension

        :param dimension: Dimension a entrenar (ej. E/I)
        '''
        self.salida_modelo = f"./{nomCarpeta}/{dimension.replace('/', '-')}_{self.modelo_base.split('/')[-1]}"
        print(f"[INFO] Iniciando Fine-Tuning BINARIO de {self.modelo_base} para la dimension {dimension}...")
        
        #Calcular pesos dinámicos para esta dimensión
        etiquetas_train = self.train_dataset[dimension]['label']
        clases_unicas = np.unique(etiquetas_train)
        pesos = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=etiquetas_train) #Genera un peso personalizado para el dataset que tratamos dependiendo del numero de ejemplos de cada uno haya
        print(f"[INFO] Pesos calculados para la clase 0 y 1: {pesos}")
        
        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.modelo_base, 
            num_labels=2
        ).to('cuda')

        training_args = TrainingArguments(
            output_dir=self.salida_modelo,
            evaluation_strategy="epoch",
            save_strategy="epoch", 
            learning_rate=2e-5,
            
            per_device_train_batch_size=24, #Numero de textos que procesa en un batch (cambiar si satura la vram)
            per_device_eval_batch_size=24,  
            dataloader_num_workers=0,          
            dataloader_pin_memory=True,
            optim="adamw_torch_fused",
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=False,#Ya no uso la 4060ti, cambia la arquitectura
            bf16=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1, 
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
        )

        trainer = WeightedTrainer(
            class_weights=pesos,
            model=modelo,
            args=training_args,
            train_dataset=self.train_dataset[dimension],
            eval_dataset=self.val_dataset[dimension],
            compute_metrics=compute_metrics
        )

        trainer.train()

        print(f"[EXITO] Guardando RoBERTa experta en {dimension} en: {self.salida_modelo}")
        trainer.save_model(self.salida_modelo)
        self.tokenizer.save_pretrained(self.salida_modelo)
        
        #Limpieza
        del modelo
        del trainer
        torch.cuda.empty_cache()
        print("[INFO] VRAM liberada para la siguiente dimensión.\n")
    
    def entrenar_modelo(self, nomCarpeta:str):
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
            self.entrenar_dimension(dimension=dimension, nomCarpeta=nomCarpeta)
        
        print(f"\n[EXITO] Entrenamiento completado en todas las dimensiones")
        
def main():
    
    entrenador = FineTunnerRoberta(nom_carpeta="dataset9K",nom_archivo="MBTI_limpio.csv")
    entrenador.entrenar_modelo(nomCarpeta="robertaFT")
    """
    entrenador = FineTunnerRoberta(nom_archivo="MBTI_limpio.csv", modelo_base="FacebookAI/xlm-roberta-base")
    entrenador.entrenar_modelo(nomCarpeta="XLMBrobertaFT")

    entrenador = FineTunnerRoberta(nom_archivo="MBTI_limpio.csv", modelo_base="FacebookAI/xlm-roberta-large")
    entrenador.entrenar_modelo(nomCarpeta="XLMLrobertaFT")
    """
if __name__ == "__main__":
    main()