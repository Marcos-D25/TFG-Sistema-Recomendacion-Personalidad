from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agente import AgenteLlama
from pipeline import Pipeline
from correlacionador import Correlacionador
from recomendador import Recomendador
from deep_translator import GoogleTranslator

#python -m http.server 8080
#uvicorn api:app --reload

app = FastAPI()

# Permitir que el frontend (HTML) hable con esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mantener el agente vivo en memoria global
agente_global = AgenteLlama()

class MensajeUsuario(BaseModel):
    texto: str

@app.get("/iniciar")
async def iniciar_chat():
    agente_global.reset()
    primer_mensaje = agente_global.iniciar_conversacion()
    print(primer_mensaje)
    return {"respuesta": primer_mensaje, "finalizado": False}

@app.post("/chat")
async def chatear(mensaje: MensajeUsuario):
    if agente_global.entrevista_finalizada:
        return {"respuesta": "Entrevista finalizada", "finalizado": True}
    
    respuesta = agente_global.enviar_texto(mensaje.texto)
    finalizado = agente_global.entrevista_finalizada
    return {"respuesta": respuesta, "finalizado": finalizado}

@app.get("/analizar")
async def analizar_perfil():
    # 1. Obtener conversación
    historial_lista = agente_global.obtener_historial()

    if not historial_lista:
        historial = "I love technical challenges, building complex systems and working alone in my garage tinkering with robots."
    else:
        historial = " ".join(historial_lista)
    
    # Traducción (tu código)
    def traducir_conversacion(texto, source='es', target='en', max_len=3000):
        traducciones = []
        for i in range(0, len(texto), max_len):
            trozo = texto[i:i + max_len]
            traducciones.append(GoogleTranslator(source=source, target=target).translate(trozo))
        return " ".join(traducciones)
    historial_en = traducir_conversacion(historial)

    # 2. Pipeline MBTI
    lista_modelos = ["Modelos Definitivos\SMOTE\E-I_Perceptron_Multicapa.joblib",
                 "Modelos Definitivos\BorderlineSMOTE\S-N_XGBoost.joblib",
                 "Modelos Definitivos\SMOTETomek\T-F_Linear_SVC.joblib",
                 "Modelos Definitivos\SMOTE\J-P_Regresion_Logistica.joblib"]
    
    lista_encoders = ["robertaFT\\E-I_roberta-base","robertaFT\\S-N_roberta-base","robertaFT\\T-F_roberta-base","robertaFT\\J-P_roberta-base"]
    
    predictor = Pipeline(historial_en, lista_modelos, lista_encoders)
    probabilidades, perfil_mbti = predictor.generar_predicciones()

    # 3. Correlaciones
    correlacionador = Correlacionador(prediccion=probabilidades)
    ocean = correlacionador.correlacionar_OCEAN()
    cine = correlacionador.correlacionar_GENEROS()
    musica = correlacionador.correlacionar_MUSICA()
    videojuegos = correlacionador.correlacionar_JUEGOS()

    # 4. Recomendaciones
    recomendador = Recomendador()
    rec_juegos = recomendador.recomendar_online(tipo_contenido='videojuegos', puntuaciones_usuario=videojuegos, top_n=5)
    rec_series = recomendador.recomendar_online(tipo_contenido='series', puntuaciones_usuario=cine, top_n=5)
    rec_pelis = recomendador.recomendar_online(tipo_contenido='peliculas', puntuaciones_usuario=cine, top_n=5)
    rec_musica = recomendador.recomendar_online(tipo_contenido='musica', puntuaciones_usuario=musica, top_n=5)

    # Construir JSON de respuesta
    return {
        "mbti": perfil_mbti,
        "ocean": ocean,
        "generos": {"cine": cine, "musica": musica, "videojuegos": videojuegos},
        "recomendaciones": {
            "series": rec_series,
            "peliculas": rec_pelis,
            "musica": rec_musica,
            "videojuegos": rec_juegos
        }
    }