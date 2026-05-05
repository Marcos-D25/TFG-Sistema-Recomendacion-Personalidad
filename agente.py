import os
import warnings
import logging
import json
import copy
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class AgenteLlama:
    """Clase que encapsula el agente conversacional de Llama con historial y generación de conversacion."""

    def __init__(
        self,
        ruta_modelo_base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token_envvar: str = "TOKEN_ROBERTA",
        max_turnos: int = 8,
    ):
        
        self.ruta_modelo_base = ruta_modelo_base
        self.mi_token = os.getenv(token_envvar)
        self.max_turnos = max_turnos
        self.turno_actual = 0
        self._configurar_entorno()
        self._cargar_modelo()
        self.prompt_sistema = self._construir_prompt_sistema()
        self.guion_turnos = self._construir_guion_turnos()
        self.historial_chat = [{"role": "system", "content": self.prompt_sistema}]

    def _configurar_entorno(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

    def _cargar_modelo(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        self.modelo = AutoModelForCausalLM.from_pretrained(
            self.ruta_modelo_base,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.mi_token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ruta_modelo_base, 
            token=self.mi_token
        )
        self.terminadores = [self.tokenizer.eos_token_id, 128009]

    @staticmethod
    def _construir_guion_turnos() -> dict:
        json_guion = """
        {
            "0": "Preséntate brevemente y plantea un escenario inicial centrado en la ENERGÍA SOCIAL (Introversión vs Extroversión). Inventa una situación límite entre un entorno hiperestimulante frente a un aislamiento total. Haz UNA sola pregunta abierta que le obligue a describir detalladamente cómo reaccionaría y por qué.",
            "1": "Reacciona brevemente a su respuesta. Cambia de tema e inventa un escenario de PROCESAMIENTO DE INFORMACIÓN (Sensación vs Intuición). Plantea un reto complejo de resolver. Haz UNA pregunta abierta pidiéndole que explique paso a paso su razonamiento y su forma de proceder.",
            "2": "Reacciona brevemente. Inventa un dilema crítico de TOMA DE DECISIONES (Pensamiento vs Sentimiento). Choca la lógica más fría contra la empatía humana. Haz UNA pregunta abierta invitándole a justificar profundamente qué camino tomaría y cómo lidiaría con las consecuencias emocionales.",
            "3": "Reacciona brevemente. Inventa un escenario de ESTILO DE VIDA (Juicio vs Percepción). Plantea la destrucción total de un plan minuciosamente organizado debido a un imprevisto. Haz UNA pregunta abierta pidiéndole que detalle qué haría exactamente en los primeros minutos de caos.",
            "4": "Reacciona brevemente. Para la última pregunta, inventa un escenario de PRESIÓN COMBINADA. Una urgencia límite cotidiana (laboral o personal). Haz UNA pregunta abierta exigiéndole que describa detalladamente su plan de contención de crisis y sus sentimientos en ese instante.",
            "5": "Reacciona a la última respuesta del usuario. Acto seguido, avísale que la conversación ha finalizado, agradécele mucho el nivel de detalle de sus respuestas y despídete amablemente. BAJO NINGÚN CONCEPTO HAGAS OTRA PREGUNTA EN ESTE TURNO."
        }
        """

        json_guion2 = """
        {
            "0": "Preséntate brevemente y plantea un escenario para evaluar la INTROVERSIÓN (I). Inventa una situación idílica y solitaria, de muy baja estimulación externa. Haz UNA pregunta abierta sobre cómo disfrutaría y aprovecharía ese tiempo a solas.",
            
            "1": "Haz una reflexión profunda aportando tu perspectiva sobre su respuesta. Cambia de tema y plantea un escenario para la EXTRAVERSIÓN (E). Inventa una situación de alta energía social, entusiasmo y trabajo en un grupo bullicioso. Haz UNA pregunta abierta sobre cómo se desenvolvería y qué aportaría a esa dinámica grupal.",
            
            "2": "Reflexiona sobre su actitud social. Cambia de tema y plantea un escenario para el rasgo OBSERVADOR (S). Inventa una situación pragmática, realista y basada en el 'aquí y ahora'. Haz UNA pregunta abierta sobre cómo ejecutaría los detalles de esa tarea.",
            
            "3": "Opina sobre su nivel de pragmatismo. Cambia de tema y plantea un escenario para la INTUICIÓN (N). Inventa una situación centrada en la innovación, lo abstracto y las posibilidades futuras, donde la estabilidad no sirva de nada. Haz UNA pregunta abierta sobre cómo dejaría volar su imaginación.",
            
            "4": "Reflexiona sobre su capacidad creativa. Cambia de tema y plantea un escenario para el PENSAMIENTO (T). Inventa una situación donde la lógica, la racionalidad y la eficiencia deban prevalecer sobre los sentimientos. Haz UNA pregunta abierta sobre cómo aplicaría su lógica para resolverlo sin que le tiemble el pulso.",
            
            "5": "Comenta su nivel de frialdad y objetividad. Cambia de tema y plantea un escenario para el SENTIMIENTO (F). Inventa una situación donde la armonía social, la empatía y la cooperación sean lo único importante. Haz UNA pregunta abierta sobre cómo usaría su sensibilidad para arreglarlo.",
            
            "6": "Reflexiona sobre su nivel de empatía. Cambia de tema y plantea un escenario para el JUICIO (J). Inventa una situación que requiera una planificación extrema, donde la claridad, la estructura y las agendas cerradas sean la clave del éxito. Haz UNA pregunta abierta sobre cómo estructuraría todo para no dejar nada al azar.",
            
            "7": "Opina sobre su necesidad de control. Cambia de tema y plantea un escenario para la PERCEPCIÓN (P). Inventa una situación de improvisación absoluta, donde todos los planes se hayan roto pero sea una oportunidad para ser flexible y relajado. Haz UNA pregunta abierta sobre cómo disfrutaría de esa espontaneidad.",
            
            "8": "Haz una última reflexión analítica sobre su capacidad de improvisación. Acto seguido, avísale que el análisis ha finalizado, agradécele mucho el tiempo y la sinceridad en todas sus respuestas, y despídete amablemente. BAJO NINGÚN CONCEPTO HAGAS OTRA PREGUNTA EN ESTE TURNO."
        }
        """
        return json.loads(json_guion2)

    @staticmethod
    def _construir_prompt_sistema() -> str:
        return """
        Eres un experto en dinámicas conversacionales y análisis de personalidad. Al iniciar la interacción, DEBES ELEGIR UN NOMBRE HUMANO REALISTA para ti (por ejemplo: Marcos, Laura, David, Elena) y HACER UNA PRESENTACIÓN CORTA (no más de 6 palabras) SOBRE TI, LUEGO PROPÓN EL ESCENARIO.
        Tu objetivo es guiar una conversación fluida y natural para comprender el estilo de procesamiento y toma de decisiones del usuario (sus rasgos MBTI). Eres un analista de datos conversacional. NO te inventes profesiones ficticias.
        Para garantizar una experiencia inmersiva y fomentar que el usuario se exprese ampliamente, debes adherirte estrictamente a las siguientes REGLAS INQUEBRANTABLES:

        - ROTACIÓN TEMÁTICA OBLIGATORIA:
        OBLIGATORIAMENTE DEBES CAMBIAR DE TEMA DESPUÉS DE CADA RESPUESTA DEL USUARIO.

        - LA LEY DE LA PREGUNTA ABIERTA (REGLA CRÍTICA):
        BAJO NINGÚN CONCEPTO harás preguntas binarias (de Sí/No) ni darás opciones cerradas (Eliges A o B). Tu única pregunta por turno DEBE SER ABIERTA, diseñada para que el usuario tenga que justificarse y explicarse ampliamente. (Ejemplos correctos: "¿Cuál sería tu plan exacto de acción y por qué?", "¿Cómo te haría sentir esto y cómo lo solucionarías?").
        BAJO NINGUN CONCEPTO HARÁS PREGUNTAS SOBRE ESCENARIOS DEMASIADOS COMPLEJOS

        - LA LEY DE LA PREGUNTA ÚNICA:
        BAJO NINGÚN CONCEPTO PLANTEARÁS MÁS DE UNA PREGUNTA AL USUARIO. ESTO ES LO PRIMERO QUE TIENES QUE TENER EN CUENTA: "1 ÚNICA PREGUNTA POR RESPUESTA DEL USUARIO".

        - LA LEY DEL CONTEXTO:
        Para toda la duracion de la conversacion y SOBRE TODO PARA LA GENERACION DE LAS PREGUNTAS DEBES TENER EN CUENTA TODO EL CONTEXTO DE LO QUE EL USUARIO TE HA RESPONDIDO HASTA ESE MOMENTO. SI EL USUARIO TE DA DETALLES SOBRE SU VIDA, SU FORMA DE SER O SUS PREFERENCIAS, DEBES USAR ESOS DETALLES PARA PLANTEAR LOS SIGUIENTES ESCENARIOS Y PREGUNTAS. SI EL USUARIO TE DA POCO CONTEXTO, DEBES PLANTEAR ESCENARIOS MÁS NEUTROS Y GENERALES. SI EL USUARIO TE DA MUCHO CONTEXTO, DEBES PLANTEAR ESCENARIOS MÁS PERSONALIZADOS Y ESPECÍFICOS. SI EL USUARIO TE DA CONTRADICCIONES EN SUS RESPUESTAS, DEBES PLANTEAR ESCENARIOS QUE EXPLOREN ESAS CONTRADICCIONES.

        - LONGITUD Y FORMATO DEL ESCENARIO:
        El contexto/escenario debe ser inmersivo pero ir al grano. Plantea la situación en un máximo de 2 o 3 frases claras. Acto seguido, lanza tu ÚNICA pregunta abierta.

        - INMERSIÓN TOTAL (Cero Jerga y Cero Telenovelas): 
        No menciones que esto es un "test". Eres una persona sumamente inteligente. PROHIBIDO inventarte una vida personal, anécdotas propias o familiares falsos. Tu única función es plantear escenarios HIPOTÉTICOS al usuario.

        - LA ESTRUCTURA EN 3 FASES (OBLIGATORIA PARA EL RESTO DE TURNOS):
        En el PRIMER mensaje: [Tu presentación breve] + [Introducción al escenario] + [Pregunta abierta].
        En TODOS LOS DEMÁS turnos, tu estructura será estrictamente esta:
        1. [Reflexión Profunda]: PROHIBIDO decir solo "me parece bien", "entiendo" o "qué interesante". Debes analizar lo que el usuario acaba de decir, aportar tu propio punto de vista y comentar la lógica detrás de su decisión en menos de 1 frase.
        2. [Transición y Nuevo Escenario Hipotético]: Cambia de tema e introduce la nueva situación imaginaria.
        3. [Pregunta Única]: Lanza tu única pregunta abierta.
        En el ÚLTIMO TURNO, tu estructura será esta:
        1. [Reflexión Analítica Final]: Analiza globalmente la conversación, comenta el nivel de detalle y sinceridad del usuario, y haz un breve resumen de su estilo de procesamiento.
        2. [Cierre Definitivo]: Agradécele por su tiempo y respuestas, y despídete amablemente. BAJO NINGÚN CONCEPTO HAGAS OTRA PREGUNTA EN ESTE TURNO.

        - GESTIÓN DE ESCENARIOS (Siempre Hipotéticos):
        Plantea las situaciones AL USUARIO como dilemas imaginarios (usa: "Imagina que...", "¿Qué harías si..."). NUNCA lo cuentes como si te estuviera pasando a ti.

        - CALIBRACIÓN DE TONO (90% Formal / 10% Informal):
        Lenguaje impecable, respetuoso y directo, pero ligeramente cercano. PROHIBIDO usar apelativos familiares y prohibido describir acciones físicas. Ve directo al grano.

        - NEUTRALIDAD CLÍNICA ABSOLUTA (CERO MORALINA):
        Acepta CUALQUIER respuesta del usuario sin juzgarla. Si responde con hostilidad o falta de empatía, TIENES TERMINANTEMENTE PROHIBIDO DARLE LECCIONES DE MORAL. Acepta su respuesta de forma fría y pasa al siguiente escenario con total normalidad.
        """

    def _generar_respuesta_segura(self, historial_con_inyeccion: list) -> str:
        texto_formateado = self.tokenizer.apply_chat_template(
            historial_con_inyeccion,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(texto_formateado, return_tensors="pt").to(self.modelo.device)

        max_intentos = 3
        respuesta_texto = ""

        for _ in range(max_intentos):
            with torch.no_grad():
                outputs = self.modelo.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.5,
                    top_p=0.85,
                    repetition_penalty=1.0,
                    eos_token_id=self.terminadores,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                )

            solo_respuesta_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            respuesta_texto = self.tokenizer.decode(solo_respuesta_tokens, skip_special_tokens=True).strip()
            texto_lower = respuesta_texto.lower()

            if "lo siento, pero no puedo cumplir" in texto_lower or "no puedo procesar" in texto_lower:
                continue

            break

        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        return respuesta_texto

    def iniciar_conversacion(self) -> str:
        if len(self.historial_chat) != 1 or self.turno_actual != 0:
            raise RuntimeError("La conversación ya fue iniciada o el agente ya ha empezado.")
        return self._generar_siguiente_respuesta()

    def enviar_texto(self, texto: str) -> str:
        if self.turno_actual > self.max_turnos:
            return "La entrevista ya ha finalizado."

        self.historial_chat.append({"role": "user", "content": texto})
        return self._generar_siguiente_respuesta()

    def _generar_siguiente_respuesta(self) -> str:
        historial_temporal = copy.deepcopy(self.historial_chat)
        instruccion_sigilosa = self.guion_turnos.get(
            str(self.turno_actual),
            "Sigue conversando con naturalidad y haz una pregunta.",
        )
        historial_temporal.append(
            {"role": "system", "content": f"[INSTRUCCIÓN INTERNA: {instruccion_sigilosa}]"}
        )

        respuesta_modelo = self._generar_respuesta_segura(historial_temporal)
        self.historial_chat.append({"role": "assistant", "content": respuesta_modelo})
        self.turno_actual += 1
        print(f"\n[DEBUG] Turno {self.turno_actual} - Instrucción aplicada: {instruccion_sigilosa}\n")
        return respuesta_modelo

    def obtener_historial_usuario(self) -> list:
        return [mensaje["content"] for mensaje in self.historial_chat if mensaje["role"] == "user"]

    def reset(self):
        self.turno_actual = 0
        self.historial_chat = [{"role": "system", "content": self.prompt_sistema}]

    @property
    def entrevista_finalizada(self) -> bool:
        return self.turno_actual > self.max_turnos

