#ARCHIVO PRINCIPAL
from agente import AgenteLlama
from pipeline import Pipeline
from correlacionador import Correlacionador
#FASE 1 -> AGENTE CONVERSACIONAL
agente = AgenteLlama()
print("\n" + "=" * 50)
print("🧠 ENTREVISTADOR INICIADO. Escribe 'salir' para terminar.")
print("=" * 50 + "\n")

print(agente.iniciar_conversacion())

while not agente.entrevista_finalizada:
    user_input = input("Tú: ")
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("\nEntrevistador: Entrevista abortada manualmente. ¡Un saludo!")
        break

    respuesta = agente.enviar_texto(user_input)
    print(f"\nEntrevistador: {respuesta}\n")

if agente.entrevista_finalizada:
    print("=" * 50)
    print("✅ ENTREVISTA FINALIZADA SEGÚN EL JSON.")
    conversacion  = agente.obtener_historial_usuario()
    conversacion = " ".join(conversacion) #Uno toda la conversacion en un único archivo
    print("Historial de usuario:", conversacion)


#FASE 2 -> PREDICCION MBTI
lista_modelos = ["Modelos Definitivos\\XGBoost\\E-I_XGBoost_calibrado.joblib",
                     "Modelos Definitivos\\Linear_SVC\\S-N_Linear_SVC_calibrado.joblib",
                     "Modelos Definitivos\\Linear_SVC\\T-F_Linear_SVC_calibrado.joblib",
                     "Modelos Definitivos\\Regresion_Logistica\\J-P_Regresion_Logistica_calibrado.joblib"]
    
lista_encoders = ["robertaFT\\E-I_roberta-base","robertaFT\\S-N_roberta-base","robertaFT\\T-F_roberta-base","robertaFT\\J-P_roberta-base"]
predictor = Pipeline(conversacion, lista_modelos, lista_encoders)

probabilidades, perfil = predictor.generar_predicciones()

#FASE 3 -> CORRELACION MBTI A OCEAN
correlacionador = Correlacionador(prediccion=probabilidades)
ocean = correlacionador.correlacionar_OCEAN()
print("-"*20+"RASGOS OCEAN"+"-"*20)
print("\n".join(f"{k}: {round(v, 2)}%" for k, v in ocean.items()))
print("-"*50)
#FASE 4 -> CORRELACION OCEAN A LOS DISTINTOS GENEROS
cine = correlacionador.correlacionar_GENEROS()
print("-"*20+"GENEROS PELICULAS/SERIES"+"-"*20)
print("\n".join(f"{k}: {round(v, 2)}%" for k, v in cine.items()))
print("-"*60)
print("-"*20+"GENEROS MUSICALES"+"-"*20)
musica = correlacionador.correlacionar_MUSICA()
print("\n".join(f"{k}: {round(v, 2)}%" for k, v in musica.items()))
print("-"*60)
print("-"*20+"GENEROS VIDEOJUEGOS"+"-"*20)
videojuegos = correlacionador.correlacionar_JUEGOS()
print("\n".join(f"{k}: {round(v, 2)}%" for k, v in videojuegos.items()))
print("-"*60)

#FASE 5 -> SUGERENCIAS DE CONTENIDOS 
