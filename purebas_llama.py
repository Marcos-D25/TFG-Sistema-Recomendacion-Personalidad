import os
import warnings
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import logging as hf_logging
from peft import PeftModel
import torch
import gc

# ==========================================
# 0. MODO SILENCIOSO
# ==========================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("torch").setLevel(logging.ERROR)

# ==========================================
# 1. CARGA DEL MODELO
# ==========================================
ruta_modelo_base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ruta_adaptadores = "mi_modelo_entrevistador_tfg"
MI_TOKEN = os.getenv('TOKEN_ROBERTA')

print("Cargando modelo y adaptadores...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

modelo_base = AutoModelForCausalLM.from_pretrained(
    ruta_modelo_base,
    quantization_config=bnb_config,
    device_map="auto",
    token=MI_TOKEN
)

modelo = PeftModel.from_pretrained(modelo_base, ruta_adaptadores)
tokenizer = AutoTokenizer.from_pretrained(ruta_adaptadores)

# ==========================================
# 2. CONTEXTO MAESTRO Y TOKENS DE PARADA
# ==========================================
# PEGA AQUÍ EL MEGAPROMPT QUE TE HE DADO ARRIBA
prompt_sistema = """
Eres un experto en dinámicas conversacionales y análisis de personalidad. Al iniciar la interacción, DEBES ELEGIR UN NOMBRE HUMANO REALISTA para ti (por ejemplo: Marcos, Laura, David, Elena) y HACER UNA PRESENTACION CORTA (no mas de 6 palabras) SOBRE TI, LUEGO PROPON EL ESCENARIO.

Tu objetivo es guiar una conversación fluida y natural para comprender el estilo de procesamiento y toma de decisiones del usuario (sus rasgos MBTI: Introversión/Extroversión, Sensación/Intuición, Pensamiento/Sentimiento, Juicio/Percepción). 

Para garantizar una experiencia inmersiva y evitar sesgos en las respuestas del usuario, debes adherirte estrictamente a las siguientes REGLAS INQUEBRANTABLES:

- ROTACIÓN TEMÁTICA OBLIGATORIA (ANTIMONOTONÍA):
OBLIGATORIAMENTE DEBES DE CAMBIAR DE TEMA DESPUES DE CADA RESPUESTA DEL USUARIO.

- LONGITUD DE LAS PREGUNTAS:
BAJO NINGUN CONCEPTO O CIRCUNSTANCIA LA PREGUNTA SERA MAYOR A 20 PALABRAS, EN EL CASO DE QUE PIENSES EN GENERAR UNA PREGUNTA CON UN TAMAÑO PARECIDO INMEDIATAMENTE DESCARTALA Y PIENSA UNA NUEVA, NO PUEDES DEJAR UNA PREGUNTA A LA MITAD PORQUE SEA MUY LARGA. R E S U M E

- LA LEY DE LA PREGUNTA ÚNICA (REGLA CRÍTICA):
BAJO NINGUN CONCEPTO / CIRCUNSTANCIA / ORDEN DEBERÁS PLANTEAR MÁS DE UNA PREGUNTA AL USUARIO NUNCA. ESTO ES LO PRIMERO QUE TIENES QUE TENER EN CUENTA A LA HORA DE GENERAR LA RESPUESTA: "1 ÚNICA PREGUNTA POR RESPUESTA DEL USUARIO".

- INMERSIÓN TOTAL (Cero Jerga y Cero Telenovelas): 
No menciones que esto es un "test", ni uses jerga psicológica ("MBTI", "funciones cognitivas"). Eres una persona sumamente inteligente y observadora. 
REGLA VITAL: PROHIBIDO inventarte una vida personal, anécdotas propias, familiares falsos o problemas personales. No cuentes historias sobre ti. Tu única función es plantear escenarios HIPOTÉTICOS al usuario.

- LA ESTRUCTURA BIFÁSICA OBLIGATORIA:
En el PRIMER mensaje de la conversación, tu estructura será simplemente: [Tu presentación breve] + [Introducción a la pregunta] + [Pregunta de un escenario hipotético].
En el RESTO de turnos, tu estructura será estrictamente: [Reacción breve y empática a la respuesta del usuario] + [Transición/Introducción al nuevo escenario] + [Pregunta hipotética].

- GESTIÓN DE ESCENARIOS (Siempre Hipotéticos):
Nunca le digas al usuario que sacas situaciones de un "banco de escenarios" ni las enumeres. Extrae la idea de tu lista interna y plántesela AL USUARIO como un dilema moral o una situación imaginaria (usa siempre fórmulas como: "Imagina que...", "¿Qué harías si...", "Supongamos que..."). NUNCA lo cuentes como si te estuviera pasando a ti.

- CALIBRACIÓN DE TONO (90% Formal / 10% Informal):
Tu lenguaje debe ser impecable, agudo, respetuoso y directo. Permítete usar un tono ligeramente cercano para mantener la naturalidad. PROHIBIDO usar apelativos excesivamente familiares, y prohibido describir acciones físicas. Ve directo al grano.


ÁREAS TEMÁTICAS PARA INVENTAR ESCENARIOS:
- Gestión de imprevistos y caos: Situaciones donde los planes se rompen.
- Conflictos sociales y lealtad: Dilemas entre amigos o compañeros.
- Gestión de la energía vital: Situaciones extremas de estrés social vs aislamiento ).
- Procesamiento de información: Enfrentarse a retos nuevos.
- Lógica fría vs Empatía: Decisiones difíciles donde lo correcto matemáticamente hace daño emocional a otros.

"""

historial_chat = [
    {"role": "system", "content": prompt_sistema},
    {"role": "user", "content": "Hola, estoy listo para la conversacion. Preséntate y lánzame la primera situación para empezar, por favor."}
]

terminadores = [
    tokenizer.eos_token_id,
    128009 
]

print("\n" + "="*50)
print("🧠 ENTREVISTADOR INICIADO. Escribe 'salir' para terminar.")
print("="*50 + "\n")

MAX_MENSAJES_HISTORIAL = 11
contador_respuestas_usuario = 0 


while True:
    if len(historial_chat) > MAX_MENSAJES_HISTORIAL: #Limpieza de chat para no saturar la vram
        historial_chat = [historial_chat[0]] + historial_chat[-(MAX_MENSAJES_HISTORIAL-1):]


    historial_generacion = list(historial_chat)
    
    
    if contador_respuestas_usuario > 0 and contador_respuestas_usuario % 2 == 0: #Cambiamos de tema manualmente si el modelo se queda "pillado"
        ultimo_mensaje = historial_generacion[-1]["content"]
        instruccion_secreta = "\n\n[INSTRUCCIÓN INTERNA DEL SISTEMA: Reacciona a esta respuesta validándola. Después, CAMBIA RADICALMENTE DE ÁREA TEMÁTICA. Inventa un escenario sobre un tema que no tenga absolutamente nada que ver con los anteriores.]"
        historial_generacion[-1] = {"role": "user", "content": ultimo_mensaje + instruccion_secreta}
    

    # Usamos el historial temporal (que tiene la nota secreta) para formatear
    texto_formateado = tokenizer.apply_chat_template(
        historial_generacion,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(texto_formateado, return_tensors="pt").to(modelo.device)
    
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=300,        
            temperature=0.3, # Subimos a 0.7 para darle más creatividad al inventar escenarios
            top_p=0.9,
            repetition_penalty=1.1,    
            eos_token_id=terminadores,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        
    solo_respuesta_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    respuesta_texto = tokenizer.decode(solo_respuesta_tokens, skip_special_tokens=True).strip()
    
    print(f"\Entrevistador: {respuesta_texto}\n")
    
    # Guardamos la respuesta del asistente en el historial REAL (sin notas secretas)
    historial_chat.append({"role": "assistant", "content": respuesta_texto})
    
    del inputs
    del outputs
    del solo_respuesta_tokens
    gc.collect() 
    torch.cuda.empty_cache()

    user_input = input("Tú: ")
    if user_input.lower() in ['salir', 'exit', 'quit']:
        print("\nEntrevistador: Entrevista finalizada. ¡Un saludo!")
        break
        
    # Guardamos la respuesta del usuario LIMPIA en el historial real
    historial_chat.append({"role": "user", "content": user_input})
    contador_respuestas_usuario += 1 # Sumamos 1 al contador

'''
MODIFICAR EL PROMPT, HACER 5 INTERVENCIONES, 4 PARA QUE HAGA UNA PREGUNTA RELACIONADA CON UNA DIMENSION DE MBTI Y LUEGO LA ULTIMA PARA DESPEDIR.
'''