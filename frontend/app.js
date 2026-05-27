const URL_API = "https://sizzle-fleshy-tanned.ngrok-free.dev";
let SESSION_TOKEN = "";

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// DICCIONARIOS DE DESCRIPCIÓN 
const DESCRIPCIONES_MBTI = {
    "INTJ": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Arquitecto", "desc": "Eres una persona estratégica, independiente y con una mente que nunca descansa. Planificas todo al milímetro y no dejas nada al azar. Tienes una sed insaciable de conocimiento y buscas constantemente optimizar los sistemas a tu alrededor. Tu lógica es impecable, aunque a veces los demás puedan verte como alguien demasiado exigente o calculador." },
    "INTP": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Lógico", "desc": "Eres un pensador abstracto y profundamente curioso. Te apasiona entender cómo funciona el universo a nivel teórico, desarmando conceptos y volviéndolos a armar en tu mente. Disfrutas explorando ideas complejas y no te conformas con respuestas superficiales, aunque a veces la ejecución práctica te resulte tediosa." },
    "ENTJ": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Comandante", "desc": "Eres un líder nato, impulsado por una visión clara y una determinación de hierro. Sabes exactamente qué quieres y cómo organizar a las personas y los recursos para conseguirlo. Tu enfoque es puramente estratégico y eficiente, y no te tiembla el pulso a la hora de tomar las decisiones más difíciles." },
    "ENTP": { "clase": "Analistas (Racionales y Estratégicos)", "titulo": "Innovador", "desc": "Eres la máxima expresión del abogado del diablo. Te encanta el debate intelectual, cuestionar el status quo y jugar con ideas audaces. Tienes una mente ágil y creativa que salta de un concepto a otro con enorme facilidad, prefiriendo la lluvia de ideas a la ejecución rutinaria." },
    "INFJ": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Abogado", "desc": "Eres una persona profundamente idealista y reflexiva, con una intuición casi mágica para leer las emociones de los demás. No te conformas con soñar; actúas para hacer del mundo un lugar mejor. Aunque tienes convicciones morales inquebrantables, a menudo necesitas soledad para recargar tu gran desgaste emocional." },
    "INFP": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Mediador", "desc": "Tienes un mundo interior poético y guiado por valores muy profundos. Buscas la armonía y siempre intentas ver lo mejor en las personas. Eres compasivo, soñador y altamente creativo, priorizando la autenticidad personal y la expresión emocional por encima de la fría lógica." },
    "ENFJ": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Protagonista", "desc": "Eres un líder carismático e inspirador que conecta de forma genuina y rápida con las multitudes. Sientes una preocupación profunda por tu comunidad y tienes un talento innato para motivar a los demás a alcanzar su máximo potencial, siendo el verdadero pilar emocional de tu círculo." },
    "ENFP": { "clase": "Diplomáticos (Empáticos e Idealistas)", "titulo": "Activista", "desc": "Eres un espíritu libre, entusiasta y lleno de curiosidad. Encuentras magia y significado en todo lo que te rodea, conectando ideas y personas de formas sorprendentes. Tu energía es contagiosa y prefieres explorar nuevas posibilidades en lugar de seguir rutinas estrictas o convencionales." },
    "ISTJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Logista", "desc": "Eres el pilar de la fiabilidad y el deber. Valoras la tradición, el orden y la estructura por encima de todo. Eres meticuloso, observador y altamente responsable; cuando asumes un compromiso, lo llevas a cabo con precisión analítica y sin buscar atajos, apoyándote siempre en hechos." },
    "ISFJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Defensor", "desc": "Eres una persona extraordinariamente cálida y protectora, siempre atenta a los detalles y a las necesidades prácticas de tus seres queridos. Disfrutas manteniendo la armonía, y aunque eres reservado, tienes unas habilidades sociales excelentes cuando se trata de cuidar y apoyar a tu entorno." },
    "ESTJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Ejecutivo", "desc": "Eres el administrador por excelencia. Tienes un talento natural para gestionar personas y proyectos, instaurando reglas claras y procesos ultra eficientes. Eres directo, honesto y valoras profundamente el trabajo duro y la integridad, siendo el primero en tomar las riendas en el caos." },
    "ESFJ": { "clase": "Centinelas (Prácticos y Organizados)", "titulo": "Cónsul", "desc": "Eres el corazón de tu comunidad, increíblemente sociable, atento y servicial. Disfrutas asegurándote de que todos a tu alrededor se sientan valorados e incluidos. Te basas en la lealtad, las tradiciones y la cooperación para tomar decisiones, siendo un anfitrión inmejorable." },
    "ISTP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Virtuoso", "desc": "Eres un experimentador audaz con un enfoque puramente táctico y físico. Disfrutas entendiendo cómo funcionan las cosas desarmándolas y resolviendo problemas lógicos sobre la marcha. Eres tranquilo, altamente independiente y capaz de mantener la sangre fría en una crisis." },
    "ISFP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Aventurero", "desc": "Eres un verdadero artista en la forma en la que vives la vida. Eres espontáneo, flexible y tienes una sensibilidad estética enorme. Vives en el momento presente, explorando nuevas pasiones, disfrutando de la belleza del entorno y conectando con los demás sin juzgarlos." },
    "ESTP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Emprendedor", "desc": "Eres pura adrenalina y acción. No te gusta la teoría pesada; prefieres lanzarte de cabeza, observar tu entorno y adaptarte en el acto. Eres perceptivo, audaz y carismático, lo que te permite navegar por situaciones de alto riesgo con una habilidad y confianza deslumbrantes." },
    "ESFP": { "clase": "Exploradores (Espontáneos y Prácticos)", "titulo": "Animador", "desc": "Vives para el aquí y el ahora, y tu entusiasmo es magnético. Te encanta ser el centro de atención, improvisar y hacer que cada momento sea divertido y memorable. Tienes un gran sentido de la estética y un encanto natural que anima radicalmente cualquier entorno." }
};
const DESCRIPCIONES_OCEAN = {
    "Openness (Apertura)": "Mide la curiosidad intelectual, la creatividad y la disposición hacia la novedad.",
    "Conscientiousness (Responsabilidad)": "Evalúa el grado de organización, persistencia y sentido del deber.",
    "Extraversion (Extraversión)": "Cuantifica el nivel de energía social, la asertividad y la búsqueda de estímulos.",
    "Agreeableness (Amabilidad)": "Define la tendencia hacia la cooperación, la empatía y la confianza.",
    "Neuroticism (Neuroticismo)": "Analiza la estabilidad emocional y la sensibilidad ante el estrés."
};

function getHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SESSION_TOKEN}`,
        'ngrok-skip-browser-warning': 'true' 
    };
}

function addMessage(text, isUser = false) {
    const div = document.createElement("div");
    div.classList.add("msg", isUser ? "msg-user" : "msg-bot");
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// LÓGICA DE LOGIN
document.getElementById("btn-login").addEventListener("click", async () => {
    const passInput = document.getElementById("password-input").value.trim();
    const errorMsg = document.getElementById("login-error");

    try {
        const res = await fetch(`${URL_API}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
            body: JSON.stringify({ password: passInput })
        });
        const data = await res.json();
        
        if (!res.ok) {
            errorMsg.textContent = data.detail || "Error.";
            errorMsg.classList.remove("d-none");
        } else {
            SESSION_TOKEN = data.token;
            document.getElementById("login-section").classList.add("d-none");
            document.getElementById("home-section").classList.remove("d-none");
        }
    } catch (err) { alert("Error de conexión"); }
});

// LÓGICA DE DESCONEXIÓN
document.getElementById("btn-disconnect").addEventListener("click", async () => {
    try { 
        await fetch(`${URL_API}/logout`, { method: 'POST', headers: getHeaders() }); 
    } catch(e) {}
    SESSION_TOKEN = "";
    document.getElementById("home-section").classList.add("d-none");
    document.getElementById("login-section").classList.remove("d-none");
});

// LÓGICA DE INFORMACIÓN
document.getElementById("btn-info").addEventListener("click", () => {
    document.getElementById("home-section").classList.add("d-none");
    document.getElementById("info-section").classList.remove("d-none");
});

document.getElementById("btn-back-home").addEventListener("click", () => {
    document.getElementById("info-section").classList.add("d-none");
    document.getElementById("home-section").classList.remove("d-none");
});

// LÓGICA DE NAVEGACIÓN
document.getElementById("btn-start-chat").addEventListener("click", async () => {
    document.getElementById("home-section").classList.add("d-none");
    document.getElementById("loading-section").classList.remove("d-none");
    const res = await fetch(`${URL_API}/iniciar`, { headers: getHeaders() });
    const data = await res.json();
    document.getElementById("loading-section").classList.add("d-none");
    document.getElementById("chat-section").classList.remove("d-none");
    addMessage(data.respuesta, false);
});

// Función para enviar mensaje
async function enviarMensaje() {
    const text = userInput.value.trim();
    if (!text) return;
    addMessage(text, true);
    userInput.value = ""; 
    userInput.disabled = true; 
    sendBtn.disabled = true;
    userInput.style.height = 'auto';
    try {
        const res = await fetch(`${URL_API}/chat`, {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify({ texto: text })
        });
        const data = await res.json();
        addMessage(data.respuesta, false);
        if (data.finalizado) {
            document.getElementById("input-area").classList.add("d-none");
            document.getElementById("analyze-area").classList.remove("d-none");
        } else { 
            userInput.disabled = false; 
            sendBtn.disabled = false; 
            userInput.focus(); 
        }
    } catch (err) { addMessage("Error.", false); }
}

// Evento click del botón ENVIAR
sendBtn.addEventListener("click", enviarMensaje);

// Evento keydown - Enter para enviar, Shift+Enter para salto de línea
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        enviarMensaje();
    }
});

// Auto-resize del textarea
userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + "px";
});

// ANÁLISIS
document.getElementById("btn-start-analysis").addEventListener("click", iniciarAnalisis);

async function iniciarAnalisis() {
    document.getElementById("chat-section").classList.add("d-none");
    document.getElementById("loading-section").classList.remove("d-none");
    const res = await fetch(`${URL_API}/analizar`, { headers: getHeaders() });
    const data = await res.json();
    mostrarResultados(data);
}

function mostrarResultados(data) {
    document.getElementById("loading-section").classList.add("d-none");
    document.getElementById("results-section").classList.remove("d-none");
    
    // Rellenar MBTI
    const mbti = data.mbti;
    const info = DESCRIPCIONES_MBTI[mbti];
    document.getElementById("mbti-title").textContent = info ? info.clase : "Perfil";
    document.getElementById("mbti-subtitle").textContent = info ? `${mbti}: ${info.titulo}` : mbti;
    document.getElementById("mbti-desc").textContent = info ? info.desc : "No disponible.";

    // LLENAR HISTORIAL DEL CHAT
    const chatHistoryBox = document.getElementById("chat-history-box");
    if (chatHistoryBox) {
        chatHistoryBox.innerHTML = chatBox.innerHTML;
        chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
    }

    // LLENAR OCEAN
    const oceanContainer = document.getElementById("ocean-container");
    if (oceanContainer && data.ocean) {
        oceanContainer.innerHTML = Object.entries(data.ocean).map(([key, value]) => {
            const porcentaje = Math.round(typeof value === 'string' ? parseFloat(value) : value);
            return `
                <div class="col-md-6 mb-4">
                    <h6 class="text-white mb-2">${key}</h6>
                    <div class="progress" style="height: 25px; background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 4px;">
                        <div class="progress-bar" style="width: ${porcentaje}%; background: linear-gradient(90deg, var(--neon-cyan), var(--neon-green)); font-weight: bold; display: flex; align-items: center; justify-content: center; color: #000;">
                            ${porcentaje}%
                        </div>
                    </div>
                </div>
            `;
        }).join("");
    }

    // Renderizado de recomendaciones
    const render = (id, items, isM, isG) => {
        const container = document.getElementById(id);
        container.innerHTML = items.map(i => `
            <div class="rec-row mb-4 pb-3">
                <div class="d-flex justify-content-between align-items-end border-bottom border-secondary pb-1">
                    <h5 class="mb-0 rec-item-title">${i.name}</h5>
                    <span class="rec-item-genre">[ ${i.genre} ]</span>
                </div>
                ${!isG ? `<div class="mt-2"><p class="mb-0 rec-item-desc">${isM ? 'Artista: '+i.artist : 'Sinopsis: '+i.overview}</p></div>` : ''}
            </div>
        `).join("");
    };

    render("series-list", data.recomendaciones.series, false, false);
    render("pelis-list", data.recomendaciones.peliculas, false, false);
    render("musica-list", data.recomendaciones.musica, true, false);
    render("juegos-list", data.recomendaciones.videojuegos, false, true);
}

// Botón: Abortar desde el Chat
document.getElementById("btn-abort-chat").addEventListener("click", async () => {
    const confirmar = confirm("⚠️ ADVERTENCIA: Volver al menú principal eliminará todo el progreso y el historial actual. ¿Desea abortar el protocolo?");
    
    if (confirmar) {
        document.getElementById("chat-section").classList.add("d-none");
        document.getElementById("home-section").classList.remove("d-none");
        chatBox.innerHTML = "";
        
        // Restaurar estado de input y botón
        document.getElementById("input-area").classList.remove("d-none");
        document.getElementById("analyze-area").classList.add("d-none");
        userInput.disabled = false; 
        sendBtn.disabled = false;
        
        // NO eliminar el token - permite regresar a home y hacer otro chat sin login
    }
});

// Botón: Volver al Inicio desde Resultados
document.getElementById("btn-restart-system").addEventListener("click", async () => {
    document.getElementById("results-section").classList.add("d-none");
    document.getElementById("login-section").classList.remove("d-none");
    chatBox.innerHTML = "";
    
    // Resetear sesión
    try { await fetch(`${URL_API}/logout`, { method: 'POST', headers: getHeaders() }); } catch(e) {}
    SESSION_TOKEN = "";
});

// Detectar cuando el usuario cierra la pestaña o cambia de página
window.addEventListener("beforeunload", async (event) => {
    if (SESSION_TOKEN) {
        // Usamos sendBeacon: es una petición asíncrona que el navegador garantiza que se enviará
        // aunque la página se esté cerrando.
        const data = JSON.stringify({ token: SESSION_TOKEN });
        navigator.sendBeacon(`${URL_API}/logout`, data);
    }
});