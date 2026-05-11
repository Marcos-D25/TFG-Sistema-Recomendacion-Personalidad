const URL_API = "http://localhost:8000";
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// DICCIONARIOS DE DESCRIPCIÓN (Iguales que antes)
const DESCRIPCIONES_MBTI = {
    "INTJ": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Arquitecto", "desc": "Eres una persona estratégica, independiente y con una mente que nunca descansa..." },
    "INTP": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Lógico", "desc": "Eres una persona curiosa hasta la médula, un pensador abstracto que siempre está buscando el porqué..." },
    "ENTJ": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Comandante", "desc": "Eres una persona que irradia autoridad natural y tienes una visión clarísima de hacia dónde quieres ir..." },
    "ENTP": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Innovador", "desc": "Eres una persona ingeniosa, inconformista y a la que le fascina el juego mental..." },
    "INFJ": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Abogado", "desc": "Eres una persona profundamente idealista y reflexiva, pero no de las que se quedan solo en sueños abstractos; tú sientes la necesidad visceral de tomar medidas reales para hacer del mundo un lugar mejor. Seguramente tienes una intuición casi mágica para leer a las personas, al punto de que tus amigos acuden a ti buscando ese consejo profundo, compasivo y revelador que solo tú sabes dar. Sin embargo, aunque te preocupas muchísimo por la humanidad, te agotas socialmente con facilidad y necesitas desaparecer del radar de vez en cuando, refugiándote en tu soledad para recargar tus baterías emocionales. Tienes unos principios morales inquebrantables y te duele físicamente presenciar la injusticia. Eres un perfeccionista oculto; trabajas incansablemente detrás de escena, a menudo exigiéndote demasiado porque sientes que tu propósito vital tiene un significado inmenso." },
    "INFP": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Mediador", "desc": "Eres una persona con un mundo interior tan rico, poético y vasto que a veces la realidad cotidiana exterior te resulta un poco gris..." },
    "ENFJ": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Diplomáticos (Empáticos e Idealistas): Protagonista", "desc": "Eres una persona que actúa como el verdadero pegamento emocional de tu círculo social..." },
    "ENFP": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Diplomáticos (Empáticos e Idealistas): Activista", "desc": "Eres una persona libre, vibrante y llena de una curiosidad insaciable por la vida..." },
    "ISTJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Centinelas (Prácticos y Organizados): Logista", "desc": "Eres una persona que representa la definición exacta de la palabra fiabilidad..." },
    "ISFJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Centinelas (Prácticos y Organizados): Defensor", "desc": "Eres una persona increíblemente cálida, protectora y con una atención asombrosa hacia los pequeños detalles..." },
    "ESTJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Centinelas (Prácticos y Organizados): Ejecutivo", "desc": "Eres una persona sumamente práctica, realista y que sabe exactamente cómo poner orden donde solo hay caos..." },
    "ESFJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Centinelas (Prácticos y Organizados): Cónsul", "desc": "Eres una persona extrovertida, extremadamente sociable y a la que le importa profundamente el bienestar de su comunidad..." },
    "ISTP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Exploradores (Espontáneos y Prácticos): Virtuoso", "desc": "Eres una persona relajada, tremendamente independiente y que vive anclada en el mundo físico..." },
    "ISFP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Exploradores (Espontáneos y Prácticos): Aventurero", "desc": "Eres una persona verdaderamente singular, guiada por una gran sensibilidad estética..." },
    "ESTP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Exploradores (Espontáneos y Prácticos): Emprendedor", "desc": "Eres una persona llena de adrenalina, orientada puramente a la acción y a la que le encanta estar en el ojo del huracán..." },
    "ESFP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Exploradores (Espontáneos y Prácticos): Animador", "desc": "Eres una persona que, literalmente, ilumina la habitación en el instante en que cruza la puerta..." }
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

// Botón: Abortar desde el Chat
document.getElementById("btn-abort-chat").addEventListener("click", () => {
    const confirmar = confirm("⚠️ ADVERTENCIA CRÍTICA: Volver al menú principal eliminará todo el progreso y el historial de la conversación actual. ¿Desea abortar el protocolo?");
    
    if (confirmar) {
        document.getElementById("chat-section").classList.add("d-none");
        document.getElementById("home-section").classList.remove("d-none");
        
        // Limpiamos la interfaz del chat
        chatBox.innerHTML = "";
        
        // Aquí idealmente harías un fetch(`${URL_API}/iniciar`) por detrás 
        // para que tu AgenteLlama en Python también haga un .reset() de su historial.
        fetch(`${URL_API}/iniciar`).catch(e => console.error("Error reseteando el backend:", e));
    }
});

// Botón: Volver al Inicio desde los Resultados (No hace falta warning aquí, ya terminó)
document.getElementById("btn-restart-system").addEventListener("click", () => {
    document.getElementById("results-section").classList.add("d-none");
    document.getElementById("home-section").classList.remove("d-none");
    
    // Limpiamos el chat para la próxima vez
    chatBox.innerHTML = "";
});

// ------------------------------------------------------------------------
// LÓGICA DE LA PANTALLA PRINCIPAL
// ------------------------------------------------------------------------

// 1. Botón INICIAR CHAT
document.getElementById("btn-start-chat").addEventListener("click", async () => {
    // Ocultar Home y mostrar Carga
    document.getElementById("home-section").classList.add("d-none");
    document.getElementById("loading-text").textContent = "ESTABLECIENDO CONEXIÓN CON EL NÚCLEO...";
    document.getElementById("loading-section").classList.remove("d-none");

    try {
        const res = await fetch(`${URL_API}/iniciar`);
        const data = await res.json();
        
        // Ocultar Carga y mostrar Chat
        document.getElementById("loading-section").classList.add("d-none");
        document.getElementById("chat-section").classList.remove("d-none");
        
        addMessage(data.respuesta, false);
        userInput.focus();
    } catch (err) { 
        document.getElementById("loading-section").classList.add("d-none");
        document.getElementById("home-section").classList.remove("d-none");
        alert("ERROR: No se pudo conectar con el servidor backend (FastAPI)."); 
    }
});

// 2. Botón INFORMACIÓN
document.getElementById("btn-info").addEventListener("click", () => {
    document.getElementById("home-section").classList.add("d-none");
    document.getElementById("info-section").classList.remove("d-none");
});

document.getElementById("btn-back-home").addEventListener("click", () => {
    document.getElementById("info-section").classList.add("d-none");
    document.getElementById("home-section").classList.remove("d-none");
});

// 3. Botón DUMMY SKIP (Desde Home)
document.getElementById("btn-home-dev-skip").addEventListener("click", () => {
    document.getElementById("home-section").classList.add("d-none");
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

// ------------------------------------------------------------------------
// LÓGICA DEL CHAT
// ------------------------------------------------------------------------

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
    if (event.key === "Enter") {
        event.preventDefault();
        if (!sendBtn.disabled) sendBtn.click();
    }
});

async function iniciarAnalisis() {
    document.getElementById("chat-section").classList.add("d-none");
    document.getElementById("loading-text").textContent = "ANALIZANDO PATRONES PSICOLÓGICOS...";
    document.getElementById("loading-section").classList.remove("d-none");
    try {
        const res = await fetch(`${URL_API}/analizar`);
        const data = await res.json();
        mostrarResultados(data);
    } catch (err) { alert("ERROR CRÍTICO EN EL PROCESAMIENTO"); }
}

function mostrarResultados(data) {
    document.getElementById("loading-section").classList.add("d-none");
    document.getElementById("results-section").classList.remove("d-none");
    
    // Solo copia el historial si se jugó al chat real
    if (chatBox.innerHTML.trim() !== "") {
        document.getElementById("chat-history-box").innerHTML = chatBox.innerHTML;
    } else {
        document.getElementById("chat-history-box").innerHTML = "<p class='text-muted p-4'>No hay historial. Se usó el Bypass de Desarrollo.</p>";
    }

    // 1. MBTI CON DESCRIPCIÓN
    const mbti = data.perfil_mbti || data.mbti;
    const info = DESCRIPCIONES_MBTI[mbti];
    document.getElementById("mbti-title").textContent = info ? `${info.clase}` : "Perfil MBTI";
    document.getElementById("mbti-subtitle").textContent = info ? `${mbti}: ${info.titulo}` : mbti;
    document.getElementById("mbti-desc").textContent = info ? info.desc : "Perfil detectado.";

    // 2. OCEAN CON DESCRIPCIÓN
    const oceanContainer = document.getElementById("ocean-container");
    oceanContainer.innerHTML = ""; 
    
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

    render("series-list", data.recomendaciones.series, false, false);
    render("pelis-list", data.recomendaciones.peliculas, false, false);
    render("musica-list", data.recomendaciones.musica, true, false);
    render("juegos-list", data.recomendaciones.videojuegos, false, true);
}