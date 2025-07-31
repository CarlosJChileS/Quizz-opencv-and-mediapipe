import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np

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
                if jugador_id not in _RESPUESTAS_FRAMES["_frames"]:
                    _RESPUESTAS_FRAMES["_frames"][jugador_id] = {"count":0, "label":None, "t_start":None, "responded":False}
                if jugadores[jugador_idx]["respuesta"] is not None:
                    continue
                import time
                mem = _RESPUESTAS_FRAMES["_frames"][jugador_id]
                now = time.time()
                if mem["t_start"] is None:
                    mem["t_start"] = now
                mem["count"] += 1
                mem["label"] = label
                if (now - mem["t_start"] > 0.5) and (mem["count"] >= 10):
                    jugadores[jugador_idx]["respuesta"] = "REAL" if label == "Right" else "IA"
                    mem["responded"] = True
                    color = jugadores[jugador_idx]["color"]
                    cv2.circle(img, (xh, yh), 36, color, -1)
                    cv2.putText(img, jugadores[jugador_idx]["respuesta"], (xh - 30, yh + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                prog = min(1, (now - mem["t_start"])/0.5 if mem["t_start"] else 0)
                end_angle = int(360*prog)
                cv2.ellipse(img, (xh, yh), (42, 42), 0, 0, end_angle, (255,255,0), 4)
            else:
                if jugadores and jugador_idx is not None:
                    jugador_id = jugadores[jugador_idx]["id"]
                    _RESPUESTAS_FRAMES["_frames"].pop(jugador_id, None)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for jug in jugadores:
        x, y, ancho, alto = jug["bbox"]
        color = jug["color"]
        cv2.rectangle(img, (x, y), (x+ancho, y+alto), color, 4)
        cv2.putText(img, f"Jugador {jug['id']}", (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)

    respuestas = {}
    for jug in jugadores:
        if jug["respuesta"]:
            respuestas[jug["id"]] = jug["respuesta"]
    colores_por_id = {jug["id"]: jug["color"] for jug in jugadores}
    return respuestas, img, colores_por_id

# --- Pantalla de celebración de campeón con foto ---
def mostrar_campeon_con_foto(campeon_id, color, rostro, segundos=3):
    import cv2
    import numpy as np

    W, H = 900, 600
    # Fondo neutro (gris azulado)
    img = np.full((H, W, 3), (60, 80, 120), dtype=np.uint8)

    if rostro is not None:
        rostro = cv2.resize(rostro, (400, 400))
        x0 = (W - 400) // 2
        y0 = (H - 400) // 2
        img[y0:y0+400, x0:x0+400] = rostro
    else:
        cv2.putText(img, "No se detecto rostro", (W//2-180, H//2), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255), 3, cv2.LINE_AA)

    cv2.namedWindow("CAMPEON", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAMPEON", W, H)
    cv2.imshow("CAMPEON", img)
    cv2.waitKey(int(segundos * 1000))
    cv2.destroyWindow("CAMPEON")
