#ARCHIVO PRINCIPAL
from agente import AgenteLlama
from pipeline import Pipeline
from correlacionador import Correlacionador
from recomendador import Recomendador
from deep_translator import GoogleTranslator

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
    conversacion  = agente.obtener_historial()
    conversacion = " ".join(conversacion) #Uno toda la conversacion en un único archivo
    print("Historial de usuario:", conversacion+"\n")

conversacion_prueba = (
'''
 Creo que aprovecharía esa paz para desconectar de verdad y poner mis pensamientos en orden. Seguramente me pasaría las horas paseando por la orilla, observando la naturaleza y reflexionando con tranquilidad.
Me parece que tu respuesta sugiere que valoras la introspección y la conexión con la naturaleza, lo que podría indicar un fuerte sentido de introspección y pensamiento reflexivo. Esto podría ser un aspecto interesante en tu personalidad. Aunque tanta multitud me agobia un poco por el ruido, me centraría en la conexión con mis amigos en ese momento tan especial. Me aseguraría de que todos estemos cómodos y juntos, cantando nuestra canción para crear un recuerdo bonito y con mucho significado para el grupo.
Me parece que tu respuesta sugiere que valoras la conexión social y la importancia de compartir experiencias con amigos, lo que podría indicar un fuerte sentido de sociabilidad y empatía. Esto podría ser un aspecto interesante en tu personalidad. Me prepararía a fondo investigando qué necesitan realmente los clientes para enfocar la presentación en cómo nuestro proyecto resuelve sus problemas específicos. Durante la reunión, intentaría mantener un tono tranquilo y empático para que se sientan escuchados y confíen en nuestra visión.
Me parece que tu respuesta sugiere un alto nivel de pragmatismo y una fuerte capacidad para adaptarte a diferentes situaciones y necesidades. Esto podría ser un aspecto interesante en tu personalidad. Creo que la usaría para viajar al pasado y presenciar momentos históricos clave para intentar entender mejor los sentimientos y decisiones de la humanidad. También me gustaría crear espacios virtuales pacíficos y seguros donde la gente pudiera ir a reflexionar y sanar emocionalmente.
Me parece que tu respuesta sugiere un alto nivel de curiosidad, empatía y creatividad, lo que podría indicar un fuerte sentido de pensamiento crítico y habilidades para resolver problemas complejos. Aunque los números fríos no son lo mío, intentaría diseñar un plan que proteja primero las necesidades básicas de las familias más vulnerables. Creo que la clave sería frenar la especulación con medidas justas y transparentes para que la gente recupere la confianza en el sistema.
Me parece que tu respuesta sugiere un alto nivel de empatía y sensibilidad hacia las necesidades de los demás, lo que podría indicar un fuerte sentido de empatía y cooperación. Esto podría ser un aspecto interesante en tu personalidad. Me reuniría primero con las familias más afectadas para escuchar cómo se sienten y entender qué necesitan con más urgencia. A partir de ahí, organizaría pequeños grupos de voluntarios para repartir las tareas de limpieza y reconstrucción, asegurándome de que nadie se sienta solo o desamparado en el proceso.
Me parece que tu respuesta sugiere un alto nivel de empatía y capacidad para conectar con los demás, lo que podría indicar un fuerte sentido de cooperación y liderazgo. Establecería un cronograma claro y organizado desde el principio, asegurándome de asignar a cada persona el rol que mejor encaje con sus talentos naturales. Además, mantendría una comunicación cercana para gestionar el estrés del grupo y asegurar que todos trabajen en un ambiente de apoyo mutuo.
Imagina que estás en un vuelo de avión que se ha quedado sin combustible y está a punto de aterrizar en un campo de aterrizaje improvisado. El avión está en picada y no hay señal de ayuda cercana. ¿Cómo aprovecharías esta situación inesperada para disfrutar del momento y encontrar la calma en medio del caos? Aunque la situación es aterradora y es difícil hablar de "disfrutar", intentaría cerrar los ojos, respirar hondo y buscar la paz interior aceptando lo que venga. Seguramente le cogería la mano a la persona que tuviera al lado para transmitirle calma y asegurarnos de no sentirnos solos en ese momento.
Me parece que tu respuesta sugiere un alto nivel de resiliencia y capacidad para manejar situaciones de estrés, lo que podría indicar un fuerte sentido de adaptabilidad y calma en momentos críticos.
'''
)
def traducir_conversacion(texto, source='es', target='en', max_len=3000):
    traducciones = []
    for i in range(0, len(texto), max_len):
        trozo = texto[i:i + max_len]
        traducciones.append(GoogleTranslator(source=source, target=target).translate(trozo))
    return " ".join(traducciones)

if len(conversacion) > 5000:
    print("La conversación es demasiado larga para traducirla de una sola vez. Se dividirá en partes de menos de 5000 caracteres.")
conversacion = traducir_conversacion(conversacion)
#FASE 2 -> PREDICCION MBTI
lista_modelos = ["Modelos Definitivos\SMOTE\E-I_Perceptron_Multicapa.joblib",
                 "Modelos Definitivos\BorderlineSMOTE\S-N_XGBoost.joblib",
                 "Modelos Definitivos\SMOTETomek\T-F_Linear_SVC.joblib",
                 "Modelos Definitivos\SMOTE\J-P_Regresion_Logistica.joblib"]
    
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

recomendador = Recomendador()
recomendador.recomendar(tipo_contenido='videojuegos', puntuaciones_usuario=videojuegos, top_n=5)
recomendador.recomendar(tipo_contenido='series', puntuaciones_usuario=cine, top_n=5)
recomendador.recomendar(tipo_contenido='peliculas', puntuaciones_usuario=cine, top_n=5)
recomendador.recomendar(tipo_contenido='musica', puntuaciones_usuario=musica, top_n=5)
