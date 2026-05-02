import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import gc

MI_TOKEN = os.getenv('TOKEN_ROBERTA')
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("[INFO] Cargando Tokenizador y Modelo base...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=MI_TOKEN)
tokenizer.pad_token = tokenizer.eos_token #El token que haría como relleno se sustituye por uno de End Of Sentence (varios textos a la vez en el entreno) 

bnb_config = BitsAndBytesConfig(#Cuantizar el modelo a un tamaño inferior (para que quepa en la grafica)
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=False,
)

modelo = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=MI_TOKEN
)


print("[INFO] Aplicando adaptadores LoRA...")
modelo = prepare_model_for_kbit_training(modelo)

lora_config = LoraConfig(#Usando Lora no haria falta entrenar todos los parametros del modelo (no podria si quisiera)
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #Donde focalizar en entreno
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

modelo = get_peft_model(modelo, lora_config)
modelo.print_trainable_parameters() #Muestra el conjunto de parametros que tendra que "modificar" durante el entreno (un 0.5% solamente)


print("[INFO] Preparando dataset ...")
dataset = load_dataset("json", data_files="dataset_llama.jsonl", split="train")

def formatear_dataset(ejemplo):
    texto = tokenizer.apply_chat_template(ejemplo["messages"], tokenize=False) #Format del texto de forma especifica para el entrenamiento
    return {"text": texto}

dataset = dataset.map(formatear_dataset)

# ==========================================
# 4. CONFIGURACIÓN DEL ENTRENADOR (Optimizado RTX 5070 Ti)
# ==========================================
print("[INFO] Configurando el motor de entrenamiento...")

args = TrainingArguments(
    output_dir="outputs_entrenamiento",
    per_device_train_batch_size=2, #Entreno de 2 en 2 textos
    gradient_accumulation_steps=4, #Actualizacion del modelo en el 4 ciclo (4x2)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    fp16=False,
    warmup_steps=15,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    torch_compile=False
)
modelo.config.use_cache = False
trainer = SFTTrainer(
    model=modelo,
    train_dataset=dataset,
    args=args,
)


print("[INFO] Iniciando Fine-Tuning...")
trainer.train()

print("[INFO] Guardando el modelo...")
trainer.save_model("modelo_entrevistador")
tokenizer.save_pretrained("modelo_entrevistador")
print("[INFO] Entrenamiento finalizado")


print("[INFO] INICIO LIMPIEZA...")

del modelo
del trainer
del dataset

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 

print("[INFO] FIN LIMPIEZA")