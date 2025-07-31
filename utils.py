import mediapipe as mp
import cv2
import numpy as np

COLORES_JUGADORES = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (255, 112, 67), (0, 172, 193), (255, 193, 7),
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.75,
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

def mostrar_campeones_con_foto(ids, colores, rostros, segundos=None):
    import cv2
    import numpy as np

    N = len(ids)
    if N == 0:
        return

    max_win_width = 1100
    min_foto = 260
    max_foto = 400
    margen = 24

    # Calcula tamaño de la foto según cuántos ganadores hay
    foto_w = min(max_foto, max(min_foto, int((max_win_width - margen*(N+1)) / max(N, 1))))
    foto_h = foto_w
    W = max(margen + (foto_w + margen) * N, 540)
    H = 520

    img = np.full((H, int(W), 3), (60, 80, 120), dtype=np.uint8)

    for i, (rostro, jid, color) in enumerate(zip(rostros, ids, colores)):
        x0 = margen + i * (foto_w + margen)
        y0 = (H - foto_h)//2
        # Fondo blanco para cada recuadro
        cv2.rectangle(img, (x0, y0), (x0+foto_w, y0+foto_h), (255,255,255), -1)
        if rostro is not None and rostro.size > 0:
            rostro = cv2.resize(rostro, (foto_w, foto_h))
            img[y0:y0+foto_h, x0:x0+foto_w] = rostro
        else:
            cv2.putText(img, "No se detecto", (x0+16, y0+foto_h//2), cv2.FONT_HERSHEY_DUPLEX, 0.85, (50,50,50), 2, cv2.LINE_AA)
        # Borde de color del jugador
        cv2.rectangle(img, (x0, y0), (x0+foto_w, y0+foto_h), color, 5)
        # Etiqueta centrada abajo
        nombre = f"Jugador {jid}"
        size = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)[0]
        xname = x0 + (foto_w - size[0])//2
        cv2.putText(img, nombre, (xname, y0+foto_h+30), cv2.FONT_HERSHEY_DUPLEX, 0.95, color, 2, cv2.LINE_AA)

    # --- Mensaje visual de continuar ---
    msg = "Presiona ESC para continuar..."
    size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    xpos = int((W - size[0]) / 2)
    cv2.putText(img, msg, (xpos, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("CAMPEON", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAMPEON", int(W), H)
    cv2.imshow("CAMPEON", img)
    while True:
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break
        if cv2.getWindowProperty("CAMPEON", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow("CAMPEON")
