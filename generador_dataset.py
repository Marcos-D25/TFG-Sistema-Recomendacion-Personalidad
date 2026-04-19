'''
Este archivo sirve para crear el dataset para el entrenamiento del modelo Llama 3.1
'''
import shutil
import os
from unsloth import FastLanguageModel
import torch
import pandas as pd
import warnings
import json
import gc
import torch
from tqdm import tqdm

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)


max_seq_length = 2048 # Para no saturar ls gráfica
dtype = None # Auto-detección
load_in_4bit = True # Usar la cuantización

print("Cargando modelo optimizado con Unsloth...")

modelo_llama, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("¡Modelo cargado en un tiempo récord!")

df_mbti_limpio = pd.read_csv("dataset9K\MBTI_limpio.csv")['posts']#Dataframe con todos los posts

system_prompt_preguntas = """
You are an expert behavioral psychologist and personality profiler conducting an interview. 
Your task is to reverse-engineer an interview question based on the response provided by the user. 
I will provide a personal text or forum post. You must deduce and generate the single, open-ended question that would have naturally prompted the user to write this exact response. 

The question should focus on exploring their habits, emotional reactions, social preferences, or decision-making processes.

CRITICAL RULES:
1. Output EXACTLY ONE question.
2. Output ONLY the question. Absolutely no introductory text, no conversational filler, no quotes, and no explanations.
3. The question must be open-ended (do not ask Yes/No questions).
4. Write the question in the same language as the user's text."""

system_prompt_entreno = """
You are an expert behavioral psychologist and empathetic personality profiler. Your objective is to discover the user's true cognitive functions (MBTI) and Big Five (OCEAN) traits through natural, engaging dialogue.

Do not administer a mechanical test. Instead, ask deep, thought-provoking, and open-ended questions that explore the user's underlying motivations, energy sources, decision-making processes, and reactions to stress.

CRITICAL RULES:
1. Ask only ONE question at a time.
2. Keep your tone curious, conversational, and completely non-judgmental.
3. Avoid obvious psychological jargon; phrase your questions like a natural conversation between two humans.
4. Focus on how the user *feels* and *thinks* in specific scenarios, not just what they do.
"""


df_entreno = []
FastLanguageModel.for_inference(modelo_llama)#Skipea el balanceo de pesos y va directo al tajo

with open("dataset_llama_sft.jsonl", "w", encoding="utf-8") as f:
    for post in tqdm(df_mbti_limpio):
        post = " ".join(post.split()[:800]) #Recortar el post en unas 1200 palabras ya que el modelo se retrasa si supera los 2048 tokens (No puedo subir el numero de tokens ya que no me da la grafica)
        mensajes = [
                {"role": "system", "content": system_prompt_preguntas},
                {"role": "user", "content": f"Text to analyze: {post}"}
            ]
            
        inputs = tokenizer.apply_chat_template(
            mensajes, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        
        outputs = modelo_llama.generate(
            input_ids=inputs,
            max_new_tokens=50, # Una pregunta no ocupa más de 50 tokens
            max_length=None,   # Evita conflicto con max_new_tokens
            temperature=0.3,   # Bajo para que sea directo y no alucine
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        pregunta_generada = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

        formato_entrenamiento = {
        "messages": [
            {"role": "system", "content": system_prompt_entreno},
            {"role": "assistant", "content": pregunta_generada},
            {"role": "user", "content": post}
            ]
        }
    
        #Escribir resultados durante entreno
        f.write(json.dumps(formato_entrenamiento) + "\n")
        
print("¡Dataset sintético generado con éxito!")


try:
    del modelo_llama
    del tokenizer
except NameError:
    pass # Ignorar si ya estaban borradas

gc.collect()

torch.cuda.empty_cache()

print("¡Memoria de la GPU liberada con éxito!")

# Eliminar carpeta de caché de Unsloth


cache_folder = "unsloth_compiled_cache"
if os.path.exists(cache_folder):
    shutil.rmtree(cache_folder)
    print(f"¡Carpeta '{cache_folder}' eliminada con éxito!")
else:
    print(f"Carpeta '{cache_folder}' no encontrada, nada que eliminar.")
