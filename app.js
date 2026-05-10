const URL_API = "http://localhost:8000";
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// DICCIONARIOS DE DESCRIPCIÓN
const DESCRIPCIONES_MBTI = {
    "INTJ": { "titulo": "Arquitecto", "desc": "Pensadores imaginativos y estratégicos, con un plan para todo." },
    "INTP": { "titulo": "Lógico", "desc": "Inventores innovadores con una sed insaciable por el conocimiento." },
    "ENTJ": { "titulo": "Comandante", "desc": "Líderes audaces, imaginativos y de voluntad fuerte." },
    "ENTP": { "titulo": "Innovador", "desc": "Pensadores ingeniosos y curiosos que no pueden resistir un resto intelectual." },
    "INFJ": { "titulo": "Abogado", "desc": "Idealistas callados y místicos que, sin embargo, son muy inspiradores y tenaces." },
    "INFP": { "titulo": "Mediador", "desc": "Personas poéticas, amables y altruistas, siempre ansiosas por ayudar a una buena causa." },
    "ENFJ": { "titulo": "Protagonista", "desc": "Líderes carismáticos e inspiradores, capaces de cautivar a sus oyentes." },
    "ENFP": { "titulo": "Activista", "desc": "Espíritus libres entusiastas, creativos y sociales, que siempre pueden encontrar una razón para sonreír." },
    "ISTJ": { "titulo": "Logista", "desc": "Individuos prácticos y enfocados en los hechos, cuya confiabilidad no puede ponerse en duda." },
    "ISFJ": { "titulo": "Defensor", "desc": "Protectores muy dedicados y cálidos, siempre listos para defender a sus seres queridos." },
    "ESTJ": { "titulo": "Ejecutivo", "desc": "Administradores excelentes, inigualables al gestionar cosas o personas." },
    "ESFJ": { "titulo": "Cónsul", "desc": "Personas extraordinariamente consideradas, sociables y populares, siempre con el deseo de ayudar." },
    "ISTP": { "titulo": "Virtuoso", "desc": "Experimentadores audaces y prácticos, maestros en el uso de todo tipo de herramientas." },
    "ISFP": { "titulo": "Aventurero", "desc": "Artistas flexibles y encantadores, siempre listos para explorar y experimentar algo nuevo." },
    "ESTP": { "titulo": "Emprendedor", "desc": "Personas inteligentes, enérgicas y muy perceptivas, que realmente disfrutan vivir al límite." },
    "ESFP": { "titulo": "Animador", "desc": "Animadores espontáneos, enérgicos y entusiastas; la vida nunca es aburrida a su alrededor." }
};

const DESCRIPCIONES_OCEAN = {
    "Openness": "Curiosidad intelectual, imaginación y preferencia por la novedad.",
    "Conscientiousness": "Autodisciplina, organización y orientación al logro.",
    "Extraversion": "Energía social, asertividad y búsqueda de compañía.",
    "Agreeableness": "Compasión, cooperación y amabilidad hacia los demás.",
    "Neuroticism": "Tendencia a experimentar ansiedad o inestabilidad emocional."
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
        "ocean": { "Openness": 79.05, "Conscientiousness": 37.90, "Extraversion": 15.08, "Agreeableness": 80.78, "Neuroticism": 75.06 },
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