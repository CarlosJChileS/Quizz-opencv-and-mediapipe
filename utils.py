import mediapipe as mp
import cv2
from matplotlib import pyplot as plt

COLORES_JUGADORES = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (255, 112, 67), (0, 172, 193), (255, 193, 7),
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.4,
)
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.55)
mp_draw = mp.solutions.drawing_utils

# Estado temporal por ronda
_RESPUESTAS_FRAMES = {}

def dedos_extendidos(landmarks, handedness):
    pulgar = landmarks[4].x > landmarks[3].x if handedness == "Right" else landmarks[4].x < landmarks[3].x
    dedos = [
        pulgar,
        landmarks[8].y < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y
    ]
    return sum(dedos)

def detectar_respuesta_por_rostro(img):
    global _RESPUESTAS_FRAMES
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(img_rgb)
    results_hands = hands.process(img_rgb)
    h, w, _ = img.shape

    jugadores = []
    colores_asignados = {}

    # 1. Detecta caras y asigna color
    if results_face.detections:
        for i, detection in enumerate(results_face.detections):
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            ancho = int(bbox.width * w)
            alto = int(bbox.height * h)
            if ancho < 70 or alto < 70:
                continue
            centro_cara = (x + ancho // 2, y + alto // 2)
            color_jugador = COLORES_JUGADORES[i % len(COLORES_JUGADORES)]
            jugadores.append({
                "id": i+1,
                "centro": centro_cara,
                "bbox": (x, y, ancho, alto),
                "color": color_jugador,
                "respuesta": None
            })
            colores_asignados[i+1] = color_jugador

    # 2. Lógica para evitar doble respuesta y falsos positivos
    if "_frames" not in _RESPUESTAS_FRAMES:
        _RESPUESTAS_FRAMES["_frames"] = {}

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            label = results_hands.multi_handedness[idx].classification[0].label
            xh = int(hand_landmarks.landmark[0].x * w)
            yh = int(hand_landmarks.landmark[0].y * h)
            n_dedos = dedos_extendidos(hand_landmarks.landmark, label)
            arriba = n_dedos >= 4

            # Buscar rostro más cercano
            min_dist = float("inf")
            jugador_idx = None
            for i, jug in enumerate(jugadores):
                xc, yc = jug["centro"]
                dist = abs(xc - xh) + abs(yc - yh)
                if dist < min_dist:
                    min_dist = dist
                    jugador_idx = i

            # Solo cuenta si está cerca de algún rostro (<260 px)
            if jugadores and jugador_idx is not None and arriba and min_dist < 260:
                jugador_id = jugadores[jugador_idx]["id"]
                # Inicializa memoria de frames si es necesario
                if jugador_id not in _RESPUESTAS_FRAMES["_frames"]:
                    _RESPUESTAS_FRAMES["_frames"][jugador_id] = {"count":0, "label":None, "t_start":None, "responded":False}
                # Si ya tiene respuesta, ignora (evita doble mano)
                if jugadores[jugador_idx]["respuesta"] is not None:
                    continue
                # Suma frames consecutivos de mano arriba
                import time
                mem = _RESPUESTAS_FRAMES["_frames"][jugador_id]
                now = time.time()
                if mem["t_start"] is None:
                    mem["t_start"] = now
                mem["count"] += 1
                mem["label"] = label
                # 0.5s y 10 frames mínimo
                if (now - mem["t_start"] > 0.5) and (mem["count"] >= 10):
                    jugadores[jugador_idx]["respuesta"] = "REAL" if label == "Right" else "IA"
                    mem["responded"] = True
                    color = jugadores[jugador_idx]["color"]
                    cv2.circle(img, (xh, yh), 36, color, -1)
                    cv2.putText(img, jugadores[jugador_idx]["respuesta"], (xh - 30, yh + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                # Feedback visual: círculo de progreso
                prog = min(1, (now - mem["t_start"])/0.5 if mem["t_start"] else 0)
                end_angle = int(360*prog)
                cv2.ellipse(img, (xh, yh), (42, 42), 0, 0, end_angle, (255,255,0), 4)
            else:
                # Reinicia conteo si se baja la mano o se mueve
                if jugadores and jugador_idx is not None:
                    jugador_id = jugadores[jugador_idx]["id"]
                    _RESPUESTAS_FRAMES["_frames"].pop(jugador_id, None)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 3. Dibuja rostros con color y nombre de jugador
    for jug in jugadores:
        x, y, ancho, alto = jug["bbox"]
        color = jug["color"]
        cv2.rectangle(img, (x, y), (x+ancho, y+alto), color, 4)
        cv2.putText(img, f"Jugador {jug['id']}", (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    respuestas = {}
    for jug in jugadores:
        if jug["respuesta"]:
            respuestas[jug["id"]] = jug["respuesta"]
    colores_por_id = {jug["id"]: jug["color"] for jug in jugadores}
    return respuestas, img, colores_por_id

# --- Gráficos llamativos

def mostrar_grafico_final(aciertos, errores):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    etiquetas = ['Correctas', 'Incorrectas']
    valores = [aciertos, errores]
    colores = ['#43e97b', '#f85032']

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#232526')
    fig.patch.set_facecolor('#232526')
    bars = ax.bar(etiquetas, valores, color=colores, width=0.55, edgecolor='white', linewidth=3, zorder=2)
    for i, bar in enumerate(bars):
        ax.add_patch(Rectangle((i-0.21, 0), 0.42, valores[i],
                               color=colores[i], alpha=0.24, zorder=1, linewidth=0))
    ax.set_title('🎯 Resultados Finales del Quizz', fontsize=27, color='#fff', pad=30, fontweight='bold')
    ax.set_ylabel('Cantidad de respuestas', fontsize=18, color='#fff', labelpad=20)
    ax.tick_params(axis='x', colors='#fff', labelsize=20)
    ax.tick_params(axis='y', colors='#fff', labelsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['bottom'].set_color('#fff')
    ax.spines['left'].set_color('#fff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval),
                ha='center', va='bottom', fontsize=26, color='#fff', fontweight='bold')
    plt.tight_layout()
    plt.show()

def mostrar_ranking_individual(puntaje_jugador, color_jugador):
    import matplotlib.pyplot as plt
    import numpy as np

    if not puntaje_jugador:
        print("No hay jugadores para mostrar ranking.")
        return

    jugadores_ids = list(puntaje_jugador.keys())
    jugadores = [f"Jugador {jid}" for jid in jugadores_ids]
    aciertos = [puntaje_jugador[jid] for jid in jugadores_ids]
    colores = [tuple([c/255 for c in color_jugador.get(jid, (0,0,0))]) for jid in jugadores_ids]

    max_score = max(aciertos)
    campeones_idx = [i for i, s in enumerate(aciertos) if s == max_score]

    jugadores_medalla = []
    for i, nombre in enumerate(jugadores):
        if i in campeones_idx:
            jugadores_medalla.append(f"{nombre} 🥇")
        else:
            jugadores_medalla.append(nombre)

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_facecolor('#0f2027')
    fig.patch.set_facecolor('#0f2027')

    bar_container = ax.bar(jugadores_medalla, [0]*len(aciertos), color=colores, width=0.65, edgecolor='#fff', linewidth=2, zorder=2)

    ax.set_title("🏆 Ranking Individual de Jugadores", fontsize=28, color='#fff', pad=30, fontweight="bold")
    ax.set_ylabel("Aciertos", fontsize=19, color='#fff', labelpad=15)
    ax.tick_params(axis='x', colors='#fff', labelsize=21)
    ax.tick_params(axis='y', colors='#fff', labelsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines['bottom'].set_color('#fff')
    ax.spines['left'].set_color('#fff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Animación de barras
    for i in range(1, max(aciertos)+1):
        for rect, val in zip(bar_container, aciertos):
            rect.set_height(min(val, i))
        plt.pause(0.13)
    for idx, (rect, score) in enumerate(zip(bar_container, aciertos)):
        y = rect.get_height() + 0.15
        ax.text(rect.get_x() + rect.get_width()/2, y, str(score),
                ha='center', va='bottom', fontsize=23, color='#fff', fontweight="bold")
        if idx in campeones_idx:
            ax.text(rect.get_x() + rect.get_width()/2, y + 0.6, "🥇",
                    ha='center', va='bottom', fontsize=39, fontweight="bold")
    plt.tight_layout()
    plt.show()
