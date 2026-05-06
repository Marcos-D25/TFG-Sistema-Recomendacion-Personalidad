#ARCHIVO PRINCIPAL
from agente import AgenteLlama
from pipeline import Pipeline
from correlacionador import Correlacionador
from recomendador import Recomendador
from deep_translator import GoogleTranslator
'''
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
    print("Historial de usuario:", conversacion)
'''
conversacion = '''
Buf, sinceramente creo que me volvería loco. A lo mejor el primer par de horas me doy un baño y descanso, pero luego me moriría del aburrimiento. Necesito estar haciendo cosas y hablar con gente. Seguro que me pondría a explorar toda la isla de arriba a abajo, intentar construir alguna locura de refugio o buscar la forma de salir, pero eso de quedarme tirado en la arena sin hacer absolutamente nada sería una tortura para mí.
Me parece que eres alguien que necesita estar en constante movimiento y actividad, alguien que busca desafíos y retos para mantenerse motivado. La idea de quedarse solo en la isla te parece una tortura, lo que sugiere que valoras la interacción social y el trabajo en equipo.
Ni me lo pienso. Dejo lo que tenga en la mano y voy corriendo con mis amigos a ayudar. Seguramente me metería de lleno a levantar los hierros o apartar escombros pesados. En cosas así de urgentes me sale natural ponerme a dirigir un poco a la gente que tengo alrededor, tipo "tú tira de aquí" o "vamos todos a la de tres", para hacer fuerza juntos y solucionar el problema rápido. ¡Cuanto antes lo arreglemos, antes vuelve la fiesta!
Me parece que eres alguien que se siente cómodo tomando el liderazgo en situaciones de emergencia y que valoras la acción rápida y decidida. Tu capacidad para dirigir a los demás y coordinar esfuerzos sugiere que eres alguien que se siente cómodo con el rol de líder.
Lo primero, pillo un cartel amarillo de esos de "Suelo mojado" y lo planto en medio para que nadie se rompa la crisma mientras busco las cosas. Luego le pido una fregona a algún camarero, lo seco rápido en un par de minutos y listo. No tiene mucho misterio, la verdad. En cuanto vea que ya no resbala, quito el cartel y a seguir con lo mío.
Me parece que eres alguien muy práctico y centrado en la resolución de problemas de manera eficiente. Tu enfoque en la seguridad y la eficacia es evidente.
Sinceramente, usaría eso para hacer cosas extremas sin matarme de verdad. Nada de ponerme a diseñar mundos complejos o cosas filosóficas raras. Crearía un circuito de karts gigante, o me tiraría en paracaídas desde el espacio con mis amigos. Lo veo simplemente como el parque de atracciones definitivo. Un sitio para hacer el cabra, competir en juegos locos y soltar adrenalina a tope con más gente, sin preocuparme de las consecuencias físicas.
Me parece que eres alguien que valoras la emoción y la adrenalina, y que no te preocupan demasiado las consecuencias a corto plazo.
Lo primero y más lógico es activar la alarma de incendios para que todo el mundo empiece a evacuar por las escaleras, nada de usar ascensores. Mientras la gente sale, llamo a los bomberos directamente. No me voy a poner a intentar arreglar los conductos de ventilación yo mismo porque no soy técnico y solo perdería un tiempo vital. La prioridad objetiva es vaciar el edificio lo más rápido posible de forma ordenada y dejar que los profesionales con equipo adecuado se encarguen de parar el humo.
Me parece que eres alguien muy práctico y centrado en la resolución de problemas de manera eficiente, con un enfoque objetivo y lógico.
A ver, yo no soy de dar discursos emotivos ni de ponerme sentimental. Mi forma de ayudar sería pillando una pala o una carretilla y haciendo el trabajo pesado que haga falta. Para que haya buen rollo y estemos unidos, me llevaría una nevera con cervezas y refrescos y pondría algo de música mientras curramos. Creo que la mejor forma de que el vecino se sienta arropado es que vea que le dejamos el jardín impecable rápidamente mientras pasamos un rato divertido y animado todos juntos, sin darle demasiadas vueltas al drama de su enfermedad.
Me parece que eres alguien que valoras la acción y la eficacia, y que prefieres evitar los aspectos emocionales y sentimentales.
Sinceramente, planificar todo al milímetro con cronogramas y agendas estrictas no es lo mío, me agobia tanta hoja de cálculo y tanto papeleo. Lo que haría sería contratar o delegar esa parte aburrida de la logística en un equipo que sea súper organizado. Yo me pediría ser el coordinador sobre el terreno el día de la carrera. Me pillaría una moto y estaría dando vueltas por todo el recorrido, comunicándome por radio y solucionando los problemas o imprevistos sobre la marcha, que es donde soy realmente rápido y útil.
A ver, siendo realista, si el barco se puede hundir lo primero que hago es asegurarme de que todos tengan el chaleco salvavidas puesto y que hayamos avisado por radio a emergencias. La seguridad es lo primero. Pero una vez hecho eso, si solo nos queda esperar, ¡pues a disfrutar de la adrenalina! Intentaría calmar los nervios de la gente haciendo alguna broma, sacaría lo que quede de comida o bebida y nos pondríamos a ver las olas gigantes como si fuera un espectáculo. Al final, los planes perfectos son aburridos y de esto seguro que sacamos una anécdota increíble para contar toda la vida.
Me parece que eres alguien que valoras la flexibilidad y la capacidad de adaptarte a situaciones imprevistas. A pesar de la incertidumbre y la adversidad, eres capaz de encontrar formas de disfrutar y aprovechar el momento. Tu capacidad para improvisar y adaptarte a situaciones es evidente.
En esta conversación, has demostrado ser una persona práctica, centrada en la resolución de problemas de manera eficiente, con un enfoque objetivo y lógico. Valoras la acción y la eficacia, y prefieres evitar los aspectos emocionales y sentimentales. También has demostrado ser capaz de improvisar y adaptarte a situaciones imprevistas, encontrando formas de disfrutar y aprovechar el momento.
'''
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

recomendador = Recomendador()
recomendador.recomendar(tipo_contenido='videojuegos', puntuaciones_usuario=videojuegos, top_n=5)
recomendador.recomendar(tipo_contenido='series', puntuaciones_usuario=cine, top_n=5)
recomendador.recomendar(tipo_contenido='peliculas', puntuaciones_usuario=cine, top_n=5)
recomendador.recomendar(tipo_contenido='musica', puntuaciones_usuario=musica, top_n=5)
