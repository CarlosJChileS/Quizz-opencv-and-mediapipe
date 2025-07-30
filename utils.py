import mediapipe as mp
import cv2
from matplotlib import pyplot as plt

COLORES_JUGADORES = [
    (66, 133, 244),    # azul Google
    (219, 68, 55),     # rojo Google
    (244, 180, 0),     # amarillo Google
    (15, 157, 88),     # verde
    (171, 71, 188),    # violeta
    (255, 112, 67),    # naranja
    (0, 172, 193),     # cian
    (255, 193, 7),     # mostaza
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

def mano_arriba(landmarks, img_h, umbral=0.4):
    base = landmarks[0].y
    index = landmarks[8].y
    return base > umbral and index < umbral

def detectar_respuesta_por_rostro(img):
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

    # 2. Asocia manos a la cara más cercana y pinta con color de jugador
    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            label = results_hands.multi_handedness[idx].classification[0].label
            xh = int(hand_landmarks.landmark[0].x * w)
            yh = int(hand_landmarks.landmark[0].y * h)
            arriba = mano_arriba(hand_landmarks.landmark, h)

            # Buscar rostro más cercano
            min_dist = float("inf")
            jugador_idx = None
            for i, jug in enumerate(jugadores):
                xc, yc = jug["centro"]
                dist = abs(xc - xh) + abs(yc - yh)
                if dist < min_dist:
                    min_dist = dist
                    jugador_idx = i

            # Solo cuenta si está cerca de algún rostro (<300 px, ajusta si quieres)
            if jugadores and jugador_idx is not None and arriba and min_dist < 300:
                if jugadores[jugador_idx]["respuesta"] is None:
                    if label == "Right":
                        jugadores[jugador_idx]["respuesta"] = "REAL"
                    else:
                        jugadores[jugador_idx]["respuesta"] = "IA"
                    color = jugadores[jugador_idx]["color"]
                    cv2.circle(img, (xh, yh), 36, color, -1)
                    cv2.putText(img, jugadores[jugador_idx]["respuesta"], (xh - 30, yh + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 3. Dibuja rostros con color y nombre de jugador
    for jug in jugadores:
        x, y, ancho, alto = jug["bbox"]
        color = jug["color"]
        cv2.rectangle(img, (x, y), (x+ancho, y+alto), color, 4)
        cv2.putText(img, f"Jugador {jug['id']}", (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 4. Arma respuestas por jugador
    respuestas = {}
    for jug in jugadores:
        if jug["respuesta"]:
            respuestas[jug["id"]] = jug["respuesta"]
    colores_por_id = {jug["id"]: jug["color"] for jug in jugadores}
    return respuestas, img, colores_por_id

def mostrar_grafico_final(aciertos, errores):
    import matplotlib.pyplot as plt

    etiquetas = ['Correctas', 'Incorrectas']
    valores = [aciertos, errores]
    colores = ['#4CAF50', '#F44336']

    plt.figure(figsize=(12, 6))
    barras = plt.bar(etiquetas, valores, color=colores, width=0.5)

    plt.title('🎯 Resultados Finales del Quizz', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Cantidad de respuestas', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, yval + 0.5, int(yval),
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

def mostrar_ranking_individual(puntaje_jugador, color_jugador):
    import matplotlib.pyplot as plt

    if not puntaje_jugador:
        print("No hay jugadores para mostrar ranking.")
        return

    jugadores = [f"Jugador {jid}" for jid in puntaje_jugador.keys()]
    aciertos = [puntaje_jugador[jid] for jid in puntaje_jugador.keys()]
    colores = [tuple([c/255 for c in color_jugador.get(jid, (0,0,0))]) for jid in puntaje_jugador.keys()]

    plt.figure(figsize=(13, 6))
    bars = plt.bar(jugadores, aciertos, color=colores, width=0.6)
    plt.title("🏆 Ranking Individual de Jugadores", fontsize=22, fontweight="bold", pad=16)
    plt.ylabel("Aciertos", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, aciertos):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(score),
                 ha='center', va='bottom', fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()
