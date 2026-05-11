const URL_API = "http://localhost:8000";
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// DICCIONARIOS DE DESCRIPCIÓN
const DESCRIPCIONES_MBTI = {
    "INTJ": { "titulo": "Analistas (Racionales y Estratégicos): Arquitecto", "desc": "Eres una persona estratégica, independiente y con una mente que nunca descansa. Probablemente tienes un \"plan B\" (y un plan C y D) para casi todo en tu vida, porque odias dejar las cosas al azar. Disfrutas pasar tiempo a solas perdiéndote en tus propios pensamientos, construyendo sistemas complejos o analizando cómo mejorar las cosas a tu alrededor. Seguro que te molesta profundamente la ineficiencia, la incompetencia o cuando las personas hacen las cosas \"porque siempre se han hecho así\" sin una base lógica que lo sustente. Para ti, el conocimiento es poder; prefieres investigar a fondo un tema antes de abrir la boca para dar tu opinión. Eres de los que prefieren tener un círculo muy pequeño de amigos intelectualmente estimulantes, antes que perder tu valioso tiempo en charlas triviales sobre el clima o los chismes del momento." },
    "INTP": { "titulo": "Analistas (Racionales y Estratégicos): Lógico", "desc": "Eres una persona curiosa hasta la médula, un pensador abstracto que siempre está buscando el \"porqué\" oculto de las cosas. Seguro que tienes la costumbre de detectar de inmediato las inconsistencias lógicas en lo que dicen los demás; a veces, incluso corriges a la gente en tu cabeza o en voz alta sin querer sonar arrogante, simplemente porque tu cerebro necesita que la información sea exacta. Te encanta sumergirte en internet saltando de un artículo a otro, leyendo sobre teorías complejas, desde física cuántica hasta filosofía antigua, perdiendo por completo la noción del tiempo. Tu mente es una máquina generadora de ideas brillantes, y es muy probable que tengas una lista interminable de proyectos o hobbies que dejaste a medias una vez que ya descubriste cómo funcionaban. Las rutinas estrictas y las reglas sin sentido te asfixian; tú necesitas libertad absoluta para explorar mentalmente." },
    "ENTJ": { "titulo": "Analistas (Racionales y Estratégicos): Comandante", "desc": "Eres una persona que irradia autoridad natural y tienes una visión clarísima de hacia dónde quieres ir. Si te ponen en un grupo donde nadie sabe qué hacer, tú asumes el mando automáticamente, no por ego, sino porque la ineficiencia y la falta de dirección te resultan insoportables. Ves la vida como un gran tablero de ajedrez donde cada movimiento debe estar calculado para acercarte a tus ambiciosos objetivos a largo plazo. Eres increíblemente directo y sincero; de hecho, puede que a veces la gente te considere un poco intimidante porque no te andas con rodeos ni dejas que la sensiblería se interponga en la toma de decisiones lógicas. Te encantan los retos y disfrutas debatiendo intelectualmente, no para pelear, sino para afilar tus propios argumentos y poner a prueba la inteligencia de quienes te rodean." },
    "ENTP": { "titulo": "Analistas (Racionales y Estratégicos): Innovador", "desc": "Eres una persona ingeniosa, inconformista y a la que le fascina el juego mental. Reconócelo: alguna vez has defendido una postura en la que ni siquiera creías de verdad, actuando como \"abogado del diablo\", solo por el placer de debatir y ver hasta dónde podías empujar los argumentos de la otra persona. Tienes una mente eléctrica que salta de una idea a otra a la velocidad de la luz, conectando conceptos que para el resto del mundo no tienen relación alguna. Te aburren soberanamente las tareas rutinarias, la burocracia y los detalles minuciosos; lo tuyo es la fase de lluvia de ideas, arrancar proyectos con muchísima energía y delegar la ejecución aburrida a otros. Tu encanto, carisma y rapidez mental te sacan de muchos apuros, y siempre estás buscando la próxima gran idea que rompa las reglas establecidas." },

    "INFJ": { "titulo": "Diplomáticos (Empáticos e Idealistas): Abogado", "desc": "Eres una persona profundamente idealista y reflexiva, pero no de las que se quedan solo en sueños abstractos; tú sientes la necesidad visceral de tomar medidas reales para hacer del mundo un lugar mejor. Seguramente tienes una intuición casi mágica para leer a las personas, al punto de que tus amigos acuden a ti buscando ese consejo profundo, compasivo y revelador que solo tú sabes dar. Sin embargo, aunque te preocupas muchísimo por la humanidad, te agotas socialmente con facilidad y necesitas desaparecer del radar de vez en cuando, refugiándote en tu soledad para recargar tus baterías emocionales. Tienes unos principios morales inquebrantables y te duele físicamente presenciar la injusticia. Eres un perfeccionista oculto; trabajas incansablemente detrás de escena, a menudo exigiéndote demasiado porque sientes que tu propósito vital tiene un significado inmenso." },
    "INFP": { "titulo": "Diplomáticos (Empáticos e Idealistas): Mediador", "desc": "Eres una persona con un mundo interior tan rico, poético y vasto que a veces la realidad cotidiana exterior te resulta un poco gris o decepcionante. Te guías ciegamente por tus valores y tu brújula moral interna; si algo no resuena con lo que sientes que es auténtico y correcto, simplemente no puedes hacerlo, por más que te presionen. Eres el tipo de persona que puede quedarse mirando un atardecer, escuchando una melodía o leyendo una frase en un libro y sentir una emoción tan abrumadora que se te eriza la piel. Eres increíblemente empático y optimista respecto a la naturaleza humana, siempre buscando la bondad en las personas, incluso cuando los demás ya se han rendido con ellas. Odias el conflicto y la crítica dura, prefiriendo expresarte a través de la escritura, el arte o cualquier medio donde puedas volcar con libertad toda tu inmensa sensibilidad." },
    "ENFJ": { "titulo": "Diplomáticos (Empáticos e Idealistas): Protagonista", "desc": "Eres una persona que actúa como el verdadero pegamento emocional de tu círculo social, un líder natural pero que guía desde el cariño, la empatía y la inspiración. Probablemente te pasas la vida organizando planes, asegurándote de que todos estén incluidos y de que absolutamente nadie se sienta dejado de lado. Tienes un radar especial para detectar el potencial oculto de los demás; ves en qué pueden brillar tus amigos o compañeros y te dedicas en cuerpo y alma a motivarlos para que alcancen su mejor versión. El problema es que a veces te involucras tanto en los problemas y sentimientos de los demás que te olvidas por completo de cuidar de tus propias necesidades. Eres increíblemente persuasivo, y tu carisma cálido y pasión genuina hacen que la gente confíe en ti, te cuente sus secretos más profundos y te siga sin dudarlo." },
    "ENFP": { "titulo": "Diplomáticos (Empáticos e Idealistas): Activista", "desc": "Eres una persona libre, vibrante y llena de una curiosidad insaciable por la vida y por los seres humanos. Para ti, el mundo es un lugar mágico lleno de conexiones ocultas y significados profundos esperando ser descubiertos. Seguro que eres de los que pueden entablar una conversación súper profunda sobre el sentido del universo con un perfecto desconocido en la fila del supermercado. Odias sentirte encasillado, controlado o atrapado en una rutina monótona de 9 a 5; necesitas libertad absoluta para explorar nuevas aficiones, viajar o cambiar de planes en el último minuto si te lo pide el cuerpo. Tu entusiasmo es tan contagioso que a menudo inspiras a los que te rodean a salir de su zona de confort. Tienes mil ideas geniales al día, pero confiesa: te cuesta horrores concentrarte en terminar los detalles aburridos, ¡porque tu mente ya está volando hacia tu próxima gran aventura!" },
    
    "ISTJ": { "titulo": "Centinelas (Prácticos y Organizados): Logista", "desc": "Eres una persona que representa la definición exacta de la palabra \"fiabilidad\". Si tú dices que vas a hacer algo, lo haces, punto; sin excusas, sin atajos y cumpliendo los plazos. Tu mente es como un archivo excel perfectamente organizado; recuerdas hechos, normativas y detalles pasados con una precisión que asusta a los demás. Seguramente te frustra muchísimo cuando las personas son impuntuales, cuando cambian los planes a última hora o cuando no respetan las reglas, porque para ti, el orden y la estructura son el pilar para que la sociedad funcione. Tienes una ética de trabajo intachable y te enorgullece hacer las cosas de forma meticulosa. No eres de los que hacen ruido ni buscan aplausos; prefieres que tu trabajo impecable hable por sí solo, siendo ese pilar silencioso, fuerte e indispensable en el que todos se apoyan cuando las cosas se ponen difíciles." },
    "ISFJ": { "titulo": "Centinelas (Prácticos y Organizados): Defensor", "desc": "Eres una persona increíblemente cálida, protectora y con una atención asombrosa hacia los pequeños detalles de quienes te rodean. Seguro que eres tú quien recuerda los cumpleaños de absolutamente todos, quien aparece con la comida favorita de un amigo cuando sabes que ha tenido un mal día, o quien se asegura de que a nadie le falte de nada en una reunión. Tienes una memoria emocional enciclopédica y atesoras las tradiciones familiares y los recuerdos como si fueran oro. A menudo te cuesta horrores decir \"no\" porque tu instinto más profundo es servir y ayudar a los demás, lo que muchas veces hace que te sobrecargues de trabajo en silencio. Aunque eres bastante reservado y no te gusta ser el centro de atención, si alguien se mete con tus seres queridos, sacas las garras sin dudarlo. Eres el corazón bondadoso que hace que cualquier espacio se sienta como un hogar." },
    "ESTJ": { "titulo": "Centinelas (Prácticos y Organizados): Ejecutivo", "desc": "Eres una persona sumamente práctica, realista y que sabe exactamente cómo poner orden donde solo hay caos. Si hay que organizar un viaje, un evento o liderar un proyecto en el trabajo, tú eres el primero en tomar las riendas, crear una lista de tareas detallada y asignar roles precisos a cada uno. Crees firmemente en la honestidad, el esfuerzo duro y en hacer lo que es correcto según las normas. Te saca de quicio la pereza, la incompetencia, las excusas o la gente que se anda por las ramas en lugar de ir directamente al grano. Eres un ciudadano modelo al que le gustan las tradiciones, y te esfuerzas activamente por mantener unidos a tu familia y a tu comunidad. Tu estilo de comunicación es directo y sin adornos: llamas a las cosas por su nombre, dices exactamente lo que piensas y esperas que los demás tengan la madurez de hacer lo mismo." },
    "ESFJ": { "titulo": "Centinelas (Prácticos y Organizados): Cónsul", "desc": "Eres una persona extrovertida, extremadamente sociable y a la que le importa profundamente el bienestar de su comunidad. Seguramente eres el anfitrión perfecto por excelencia, ese que siempre organiza las cenas, los cumpleaños o las reuniones de amigos, asegurándose de que la comida alcance, la decoración sea bonita y, sobre todo, que todos estén sonriendo. Eres muy sensible a la opinión que los demás tienen de ti y buscas activamente crear armonía y recibir validación de tus seres queridos. Te impacientan los debates excesivamente teóricos o abstractos sobre temas irreales; tú prefieres centrarte en cosas prácticas y tangibles que afecten directamente la vida de las personas que conoces. Eres leal hasta la médula, respetas la jerarquía y tu mayor fuente de felicidad y realización proviene de saber que eres útil, apreciado y querido por tu entorno." },
    
    "ISTP": { "titulo": "Exploradores (Espontáneos y Prácticos): Virtuoso", "desc": "Eres una persona relajada, tremendamente independiente y que vive anclada en el mundo físico. Seguramente aprendes mejor metiendo las manos; te encanta desarmar cosas, ya sea el motor de un vehículo, el software de una computadora o un instrumento musical, solo para ver cómo funcionan sus engranajes y volver a armarlas mejoradas. Tienes una paciencia asombrosa para los detalles técnicos, pero te desconectas a los cinco minutos si alguien te empieza a hablar de teorías abstractas o emociones hipercomplejas. Eres de los que mantienen la cabeza fría como el hielo en situaciones de crisis; cuando los demás entran en pánico, tú simplemente evalúas el entorno y encuentras una solución práctica e inmediata usando las herramientas que tienes a mano. Valoras tu espacio personal por encima de todo y prefieres mil veces que tus acciones hablen en lugar de perder el tiempo en discursos." },
    "ISFP": { "titulo": "Exploradores (Espontáneos y Prácticos): Aventurero", "desc": "Eres una persona verdaderamente singular, guiada por una gran sensibilidad estética y un deseo profundo de experimentar la vida de primera mano, sin filtros. Probablemente tienes un estilo muy marcado y distintivo, ya sea en tu forma de vestir, de decorar tu espacio o en tu arte, porque para ti todo es una forma de autoexpresión visual. Vives intensamente inmerso en el presente, disfrutando plenamente de los pequeños detalles sensoriales que otros ignoran: el sonido de la naturaleza, la textura de una tela o la mezcla de colores en un plato de comida. Odias sentirte atrapado por horarios rígidos o promesas a largo plazo; necesitas que tus días fluyan con improvisación. Eres increíblemente tolerante, tu filosofía es \"vive y deja vivir\", y siempre evitas juzgar a los demás, esperando simplemente que te respeten de la misma manera y te dejen ser tú mismo." },
    "ESTP": { "titulo": "Exploradores (Espontáneos y Prácticos): Emprendedor", "desc": "Eres una persona llena de adrenalina, orientada puramente a la acción y a la que le encanta estar en el ojo del huracán. Seguramente eres de los que piensan que las normas estrictas son más bien \"sugerencias aburridas\" y aplicas constantemente la regla de \"es mejor pedir perdón que pedir permiso\". Tienes un radar espectacular para leer el lenguaje corporal, lo que te convierte en un negociador nato y en alguien capaz de convencer a cualquiera con tu encanto y labia. Te aburren soberanamente las clases teóricas o las largas reuniones de planificación; tú necesitas tirarte a la piscina y aprender a nadar mientras te hundes. Disfrutas tomando riesgos, viviendo al límite, y tienes un talento casi mágico para improvisar soluciones brillantes en el último segundo, cuando todos los demás ya han tirado la toalla." },
    "ESFP": { "titulo": "Exploradores (Espontáneos y Prácticos): Animador", "desc": "Eres una persona que, literalmente, ilumina la habitación en el instante en que cruza la puerta. Tienes un magnetismo natural, un sentido del humor físico y espontáneo, y una capacidad envidiable para hacer que hasta la tarea más aburrida parezca una fiesta espectacular. Seguro que alguna vez has empezado a cantar, actuar o a bailar de la nada solo para arrancar una carcajada a quien tenías al lado. Vives al 100% en el momento presente, siempre buscando la próxima experiencia emocionante, la mejor comida, o el evento más divertido. A veces, las tareas pesadas como planificar tu futuro financiero o hacer presupuestos se te hacen cuesta arriba, porque prefieres enfocarte en disfrutar del \"aquí y el ahora\". Además de ser el alma de la fiesta, eres un amigo extremadamente observador y generoso; si alguien está triste, tú serás el primero en notarlo y harás hasta lo imposible por cambiarle el estado de ánimo." }
};

const DESCRIPCIONES_OCEAN = {
    "Openness (Apertura)": "Mide la curiosidad intelectual, la creatividad y la disposición hacia la novedad, diferenciando a los buscadores de experiencias abstractas y originales de quienes prefieren la rutina, lo práctico y lo convencional.",
    "Conscientiousness (Responsabilidad)": "Evalúa el grado de organización, persistencia y sentido del deber, separando a las personas disciplinadas y metódicas que planifican sus objetivos de aquellas que son más espontáneas, informales o desorganizadas.",
    "Extraversion (Extraversión)": "Cuantifica el nivel de energía social, la asertividad y la búsqueda de estímulos externos, distinguiendo a quienes se recargan mediante la interacción y el entusiasmo de aquellos que prefieren la introspección y la tranquilidad.",
    "Agreeableness (Amabilidad)": "Define la tendencia hacia la cooperación, la empatía y la confianza en los demás, contrastando a los individuos que priorizan la armonía y el altruismo con aquellos que poseen un enfoque más competitivo, escéptico o crítico.",
    "Neuroticism (Neuroticismo)": "Analiza la estabilidad emocional y la sensibilidad ante el estrés, midiendo la propensión a experimentar ansiedad, irritabilidad o vulnerabilidad frente a la capacidad de mantener el temple, la calma y el equilibrio emocional."
};

function addMessage(text, isUser = false) {
    const div = document.createElement("div");
    div.classList.add("msg", isUser ? "msg-user" : "msg-bot");
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

window.onload = async () => {
    try {
        const res = await fetch(`${URL_API}/iniciar`);
        const data = await res.json();
        addMessage(data.respuesta, false);
    } catch (err) { addMessage("ERROR DE CONEXIÓN CON EL NÚCLEO.", false); }
};

sendBtn.addEventListener("click", async () => {
    const text = userInput.value.trim();
    if (!text) return;
    addMessage(text, true);
    userInput.value = ""; userInput.disabled = true; sendBtn.disabled = true;
    try {
        const res = await fetch(`${URL_API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texto: text })
        });
        const data = await res.json();
        addMessage(data.respuesta, false);
        if (data.finalizado) iniciarAnalisis();
        else { userInput.disabled = false; sendBtn.disabled = false; userInput.focus(); }
    } catch (err) { addMessage("ERROR EN TRANSMISIÓN.", false); }
});
userInput.addEventListener("keydown", (event) => {
    // Si la tecla presionada es "Enter"
    if (event.key === "Enter") {
        // Evitamos que el navegador haga cosas raras (como refrescar si fuera un form)
        event.preventDefault();
        
        // Solo enviamos si el botón no está desactivado (para evitar doble envío)
        if (!sendBtn.disabled) {
            sendBtn.click();
        }
    }
});

// BYPASS DUMMY
document.getElementById("btn-dev-skip").addEventListener("click", () => {
    document.getElementById("chat-section").classList.add("d-none");
    const mockData = {
        "perfil_mbti": "INFJ",
        "ocean": { "Openness (Apertura)": 79.05, "Conscientiousness (Responsabilidad)": 37.90, "Extraversion (Extraversión)": 15.08, "Agreeableness (Amabilidad)": 80.78, "Neuroticism (Neuroticismo)": 75.06 },
        "generos": {
            "cine": {"Drama": 64.4, "Thriller": 63.9, "Fantasy": 62.7},
            "musica": {"Folk": 90.7, "Classical": 84.3},
            "videojuegos": {"Adventure": 68.6, "RPG": 65.3}
        },
        "recomendaciones": {
            "peliculas": [{"name": "Trash", "genre": "Crime, Adventure", "overview": "Three kids in Brazil make a discovery..."}],
            "series": [{"name": "Bungo Stray Dogs", "genre": "Action, Crime", "overview": "A white tiger beast..."}],
            "musica": [{"name": "Iris", "artist": "The Goo Goo Dolls", "genre": "Alternative"}],
            "videojuegos": [{"name": "Valkyria Chronicles", "genre": "RPG", "overview": "Tactical RPG set in Europa..."}]
        }
    };
    mostrarResultados(mockData);
});

async function iniciarAnalisis() {
    document.getElementById("chat-section").classList.add("d-none");
    document.getElementById("loading-section").classList.remove("d-none");
    try {
        const res = await fetch(`${URL_API}/analizar`);
        const data = await res.json();
        mostrarResultados(data);
    } catch (err) { alert("ERROR CRÍTICO"); }
}

function mostrarResultados(data) {
    document.getElementById("loading-section").classList.add("d-none");
    document.getElementById("results-section").classList.remove("d-none");
    document.getElementById("chat-history-box").innerHTML = chatBox.innerHTML;

    // 1. MBTI CON DESCRIPCIÓN
    const mbti = data.perfil_mbti || data.mbti;
    const info = DESCRIPCIONES_MBTI[mbti];
    document.getElementById("mbti-title").textContent = info ? `${mbti}: ${info.titulo}` : mbti;
    document.getElementById("mbti-desc").textContent = info ? info.desc : "Perfil detectado.";

    // 2. OCEAN CON DESCRIPCIÓN
    const oceanContainer = document.getElementById("ocean-container");
    oceanContainer.innerHTML = ""; // Limpiar contenedor
    
    // Contenedor principal de la lista
    let oceanHTML = `<div class="col-12"><div class="ocean-list-wrapper">`;
    for (const [key, val] of Object.entries(data.ocean)) {
        const desc = DESCRIPCIONES_OCEAN[key] || "Descripción no disponible";
        const porcentaje = val.toFixed(2);
        
        oceanHTML += `
            <div class="ocean-row mb-4">
                <div class="row align-items-center">
                    <div class="col-md-3">
                        <span class="ocean-dim">${key.toUpperCase()}</span>
                    </div>
                    <div class="col-md-2">
                        <span class="ocean-val">${porcentaje}%</span>
                    </div>
                    <div class="col-md-7">
                        <p class="ocean-desc mb-0">${desc}</p>
                    </div>
                </div>
                
                <div class="cyber-bar-container mt-2">
                    <div class="cyber-bar-fill" style="width: ${porcentaje}%"></div>
                </div>
            </div>
        `;
    }
    
    oceanHTML += `</div></div>`;
    oceanContainer.innerHTML = oceanHTML;

    const getTop5 = (obj) => obj ? Object.entries(obj).sort((a,b)=>b[1]-a[1]).slice(0,5).map(x=>x[0]).join(", ") : "N/A";
    const render = (id, items, isM, isG) => {
        const container = document.getElementById(id);
        
        if (!items || items.length === 0) {
            container.innerHTML = "<p class='text-muted small'>No hay datos disponibles para este perfil.</p>";
            return;
        }

        container.innerHTML = items.map(i => `
            <div class="rec-row mb-4 pb-3">
                <div class="d-flex justify-content-between align-items-end border-bottom border-secondary pb-1">
                    <h5 class="mb-0 rec-item-title">${i.name || i.Title || "Desconocido"}</h5>
                    <span class="rec-item-genre">[ ${i.genre || "Varios"} ]</span>
                </div>
                
                ${!isG ? `
                <div class="mt-2">
                    ${isM ? 
                        `<p class="mb-0 rec-item-desc"><strong>ARTISTA:</strong> ${i.artist || "Desconocido"}</p>` : 
                        `<p class="mb-0 rec-item-desc"><strong>SINOPSIS:</strong> ${i.overview || "No hay sinopsis disponible."}</p>`
                    }
                </div>` : ''}
            </div>
        `).join("");
    };

    // Actualizamos las llamadas al final de mostrarResultados
    // Parámetros: (id_contenedor, lista_datos, es_musica, es_juego)
    render("series-list", data.recomendaciones.series, false, false);
    render("pelis-list", data.recomendaciones.peliculas, false, false);
    render("musica-list", data.recomendaciones.musica, true, false);
    render("juegos-list", data.recomendaciones.videojuegos, false, true);
}