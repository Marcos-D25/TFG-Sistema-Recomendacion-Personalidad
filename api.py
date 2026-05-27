import uuid
import time # Añade este import arriba
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from agente import AgenteLlama
from pipeline import Pipeline
from correlacionador import Correlacionador
from recomendador import Recomendador
from deep_translator import GoogleTranslator

#uvicorn api:app
#ngrok http 8000
# --- CONFIGURACIÓN DE SEGURIDAD ---
PASSWORD_SISTEMA = "admin123" # Cambia esto por la contraseña que quieras
session_activa = {"token": None, "ocupado": False, "last_active": 0}
TIMEOUT = 120

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- INICIANDO CARGA DE MODELOS Y DATASETS ---")
    app.state.agente = AgenteLlama()
    app.state.recomendador = Recomendador()
    app.state.model_paths = [
        "Modelos Definitivos\\SMOTE\\E-I_Perceptron_Multicapa.joblib",
        "Modelos Definitivos\\BorderlineSMOTE\\S-N_XGBoost.joblib",
        "Modelos Definitivos\\SMOTETomek\\T-F_Linear_SVC.joblib",
        "Modelos Definitivos\\SMOTE\\J-P_Regresion_Logistica.joblib"
    ]
    app.state.encoder_paths = [
        "robertaFT\\E-I_roberta-base",
        "robertaFT\\S-N_roberta-base",
        "robertaFT\\T-F_roberta-base",
        "robertaFT\\J-P_roberta-base"
    ]
    print("--- SISTEMA LISTO PARA RECIBIR PETICIONES ---")
    yield
    print("--- CERRANDO SERVIDOR ---")

app = FastAPI(lifespan=lifespan)

# CORS abierto para permitir peticiones desde Netlify/Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MensajeUsuario(BaseModel):
    texto: str

class LoginData(BaseModel):
    password: str

# Modifica verificar_seguridad para refrescar el tiempo
def verificar_seguridad(authorization: str = Header(None)):
    # 1. Comprobamos si la cabecera existe
    if not authorization:
        raise HTTPException(status_code=401, detail="Falta la cabecera de autorización")
    
    # 2. Comprobamos el formato (debe empezar por Bearer )
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Formato de autorización inválido. Debe ser 'Bearer <token>'")
    
    # 3. EXTRAEMOS EL TOKEN de forma segura (cortando los primeros 7 caracteres: "Bearer ")
    token = authorization[7:]
    
    # 4. Comprobamos si el token extraído está vacío
    if not token:
        raise HTTPException(status_code=401, detail="Token vacío")

    # 5. Comprobamos validez de sesión
    if not session_activa["ocupado"] or session_activa["token"] != token:
        raise HTTPException(status_code=401, detail="Sesión inválida o expirada")
        
    # Actualizamos el timestamp de actividad
    session_activa["last_active"] = time.time()
    return token

def traducir_conversacion(texto, source='es', target='en', max_len=3000):
    traducciones = []
    for i in range(0, len(texto), max_len):
        trozo = texto[i:i + max_len]
        traducciones.append(GoogleTranslator(source=source, target=target).translate(trozo))
    return " ".join(traducciones)

# --- RUTAS DE SEGURIDAD ---

@app.post("/login")
async def login(data: LoginData):
    # Lógica de autoliberación: Si está ocupado pero pasó el tiempo, liberamos
    if session_activa["ocupado"]:
        if (time.time() - session_activa["last_active"]) > TIMEOUT:
            print("[SISTEMA] Sesión expirada por inactividad. Liberando...")
            session_activa["ocupado"] = False
        else:
            raise HTTPException(status_code=403, detail="⚠️ El núcleo está ocupado. Intenta de nuevo en unos minutos.")

    if data.password != PASSWORD_SISTEMA:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta.")
    
    nuevo_token = str(uuid.uuid4())
    session_activa["token"] = nuevo_token
    session_activa["ocupado"] = True
    session_activa["last_active"] = time.time() # Marcamos actividad
    return {"token": nuevo_token}



@app.post("/logout")
async def logout(authorization: str = Header(None)):
    try:
        token = verificar_seguridad(authorization)
        # Borra el token de sesión
        session_activa["token"] = None
        session_activa["ocupado"] = False
        session_activa["last_active"] = 0
        print("\n[SISTEMA] Sesión finalizada. Token eliminado. Sistema liberado y listo para nuevo usuario.")
        return {"status": "ok", "message": "Sesión cerrada. Token eliminado. Redirigiendo al login..."}
    except Exception as e:
        print(f"[ERROR] Logout fallido: {str(e)}")
        return {"status": "error", "message": "Error al cerrar sesión"}

# --- RUTAS CORE (Protegidas) ---

@app.get("/iniciar")
async def iniciar_chat(request: Request, authorization: str = Header(None)):
    verificar_seguridad(authorization)
    request.app.state.agente.reset()
    primer_mensaje = request.app.state.agente.iniciar_conversacion()
    return {"respuesta": primer_mensaje, "finalizado": False}

@app.post("/chat")
async def chatear(request: Request, mensaje: MensajeUsuario, authorization: str = Header(None)):
    verificar_seguridad(authorization)
    agente = request.app.state.agente
    if agente.entrevista_finalizada:
        return {"respuesta": "Entrevista finalizada", "finalizado": True}
    
    respuesta = agente.enviar_texto(mensaje.texto)
    return {"respuesta": respuesta, "finalizado": agente.entrevista_finalizada}

@app.get("/analizar")
async def analizar_perfil(request: Request, authorization: str = Header(None)):
    verificar_seguridad(authorization)
    agente = request.app.state.agente
    historial_lista = agente.obtener_historial()
    historial = " ".join(historial_lista) if historial_lista else "Texto de respaldo..."
    
    historial_en = traducir_conversacion(historial)
    predictor = Pipeline(historial_en, request.app.state.model_paths, request.app.state.encoder_paths)
    probabilidades, perfil_mbti = predictor.generar_predicciones()

    correlacionador = Correlacionador(prediccion=probabilidades)
    ocean = correlacionador.correlacionar_OCEAN()
    cine = correlacionador.correlacionar_GENEROS()
    musica = correlacionador.correlacionar_MUSICA()
    videojuegos = correlacionador.correlacionar_JUEGOS()

    rec = request.app.state.recomendador
    
    return {
        "mbti": perfil_mbti,
        "ocean": ocean,
        "generos": {"cine": cine, "musica": musica, "videojuegos": videojuegos},
        "recomendaciones": {
            "series": rec.recomendar_online(tipo_contenido='series', puntuaciones_usuario=cine),
            "peliculas": rec.recomendar_online(tipo_contenido='peliculas', puntuaciones_usuario=cine),
            "musica": rec.recomendar_online(tipo_contenido='musica', puntuaciones_usuario=musica),
            "videojuegos": rec.recomendar_online(tipo_contenido='videojuegos', puntuaciones_usuario=videojuegos)
        }
    }