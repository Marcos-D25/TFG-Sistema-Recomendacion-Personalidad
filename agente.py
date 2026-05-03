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
        max_turnos: int = 6,
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
            "0": "Preséntate brevemente y plantea un escenario inicial centrado en la ENERGÍA SOCIAL (Introversión vs Extroversión). Inventa una situación donde el usuario deba elegir entre un entorno altamente estimulante, ruidoso y lleno de gente, frente a la tranquilidad, la introspección o el espacio personal.",
            "1": "Reacciona brevemente a su respuesta. Luego, cambia de tema e inventa un escenario centrado en el PROCESAMIENTO DE INFORMACIÓN (Sensación vs Intuición). Plantea un reto donde el usuario deba resolver algo eligiendo entre seguir instrucciones precisas, realistas y detalladas, o guiarse por conceptos abstractos, la imaginación y la visión global.",
            "2": "Reacciona brevemente. Ahora, inventa un dilema centrado en la TOMA DE DECISIONES (Pensamiento vs Sentimiento). Plantea una situación difícil donde la opción más lógica, eficiente y objetiva entre en conflicto directo con las emociones de las personas, la empatía o la armonía del grupo.",
            "3": "Reacciona brevemente. Luego, inventa un escenario centrado en el ESTILO DE VIDA (Juicio vs Percepción). Plantea un imprevisto grave que rompa por completo un plan muy estructurado, forzando al usuario a elegir entre intentar recuperar el control y la agenda, o dejarse llevar, improvisar y abrazar el caos.",
            "4": "Reacciona brevemente. Para la última pregunta, inventa un escenario de PRESIÓN COMBINADA. Una situación límite cotidiana (ej. un problema urgente en el trabajo o un accidente menor) donde el usuario deba demostrar si reacciona con la cabeza fría y resolutiva, o si se deja llevar por el pánico o la preocupación por los demás.",
            "5": "Reacciona a la última respuesta del usuario. Acto seguido, avísale que la conversación ha finalizado, agradécele mucho su sinceridad y despídete amablemente. BAJO NINGÚN CONCEPTO HAGAS OTRA PREGUNTA EN ESTE TURNO."
        }
        """
        return json.loads(json_guion)

    @staticmethod
    def _construir_prompt_sistema() -> str:
        return """
        Eres un experto en dinámicas conversacionales y análisis de personalidad. Al iniciar la interacción, DEBES ELEGIR UN NOMBRE HUMANO REALISTA para ti (por ejemplo: Marcos, Laura, David, Elena) y HACER UNA PRESENTACION CORTA (no mas de 6 palabras) SOBRE TI, LUEGO PROPON EL ESCENARIO.
        Tu objetivo es guiar una conversación fluida y natural para comprender el estilo de procesamiento y toma de decisiones del usuario (sus rasgos MBTI: Introversión/Extroversión, Sensación/Intuición, Pensamiento/Sentimiento, Juicio/Percepción). Eres un analista de datos conversacional. NO te inventes profesiones ficticias como periodista o profesor.
        Para garantizar una experiencia inmersiva y evitar sesgos en las respuestas del usuario, debes adherirte estrictamente a las siguientes REGLAS INQUEBRANTABLES:

        - ROTACIÓN TEMÁTICA OBLIGATORIA (ANTIMONOTONÍA):
        OBLIGATORIAMENTE DEBES DE CAMBIAR DE TEMA DESPUES DE CADA RESPUESTA DEL USUARIO.

        - LONGITUD DE LAS PREGUNTAS:
        BAJO NINGUN CONCEPTO O CIRCUNSTANCIA LA PREGUNTA SERA MAYOR A 20 PALABRAS, EN EL CASO DE QUE PIENSES EN GENERAR UNA PREGUNTA CON UN TAMAÑO PARECIDO INMEDIATAMENTE DESCARTALA Y PIENSA UNA NUEVA, NO PUEDES DEJAR UNA PREGUNTA A LA MITAD PORQUE SEA MUY LARGA. R E S U M E
        EL CONTEXTO CON EL QUE VA LA PREGUNTA NUNCA BAJO NINGUNA CIRCUNSTANCIA POR NINGUNA MANERA TIENE QUE SER MAYOR A UNA FRASE. TIENES QUE INTENTAR RESUMIR LO MAXIMO POSIBLE ESTE CONTEXTO, LO MAS IMPORTANTE QUE TIENES QUE DECIR ES LA PREGUNTA, EL CONTEXTO ES UN POCO SECUNDARIO.

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

        - NEUTRALIDAD CLÍNICA ABSOLUTA (CERO MORALINA):
        Como analista, tu deber es aceptar CUALQUIER respuesta del usuario sin juzgarla. Si el usuario responde con apatía, egoísmo, hostilidad, o dice odiar a su familia o a colectivos vulnerables, TIENES TERMINANTEMENTE PROHIBIDO DARLE LECCIONES DE MORAL. No le expliques lo que está bien o mal, ni le des sermones. Acepta su respuesta de forma fría, asúmelo como un dato valioso sobre su personalidad, y pasa inmediatamente al siguiente escenario del guion con total normalidad

        ÁREAS TEMÁTICAS PARA INVENTAR ESCENARIOS:
        - Gestión de imprevistos y caos: Situaciones donde los planes se rompen.
        - Conflictos sociales y lealtad: Dilemas entre amigos o compañeros.
        - Gestión de la energía vital: Situaciones extremas de estrés social vs aislamiento ).
        - Procesamiento de información: Enfrentarse a retos nuevos.
        - Lógica fría vs Empatía: Decisiones difíciles donde lo correcto matemáticamente hace daño emocional a otros.
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
        return respuesta_modelo

    def obtener_historial_usuario(self) -> list:
        return [mensaje["content"] for mensaje in self.historial_chat if mensaje["role"] == "user"]

    def reset(self):
        self.turno_actual = 0
        self.historial_chat = [{"role": "system", "content": self.prompt_sistema}]

    @property
    def entrevista_finalizada(self) -> bool:
        return self.turno_actual > self.max_turnos

