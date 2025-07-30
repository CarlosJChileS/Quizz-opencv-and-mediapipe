import cv2
import numpy as np
import time
from quizz_images import QUIZZ
from utils import detectar_respuesta_por_rostro, mostrar_grafico_final, mostrar_ranking_individual

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def letterbox(im, box_size=(400, 720), color=(30, 30, 30)):
    h, w = im.shape[:2]
    box_w, box_h = box_size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    im_resized = cv2.resize(im, (new_w, new_h))
    canvas = np.full((box_h, box_w, 3), color, dtype=np.uint8)
    x0 = (box_w - new_w) // 2
    y0 = (box_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = im_resized
    return canvas

def ejecutar_quizz():
    total_correctas = 0
    total_incorrectas = 0
    puntaje_jugador = {}
    color_jugador = {}

    for idx_preg, pregunta in enumerate(QUIZZ):
        tiempo_inicio = time.time()
        tiempo_max = 8

        respuestas_por_jugador = {}
        colores_por_id = {}

        # --- Carga la imagen y la ajusta universalmente ---
        img_q0 = cv2.imread(pregunta["image"])
        if img_q0 is None:
            print(f"No se pudo cargar la imagen {pregunta['image']}")
            continue
        img_q = letterbox(img_q0, box_size=(400, 720), color=(30,30,30))

        vis_width = 1100
        vis_height = 720

        while time.time() - tiempo_inicio < tiempo_max:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (vis_width, vis_height))

            vis_area = img.copy()
            respuestas, vis_area, colores_por_id = detectar_respuesta_por_rostro(vis_area)

            # Mensajes, barra de tiempo, conteo, instrucciones
            cv2.putText(vis_area, f"IMAGEN {idx_preg + 1} de {len(QUIZZ)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 4)
            cv2.putText(vis_area, "¿Hecha por IA?", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(vis_area, "Derecha=REAL  |  Izquierda=IA", (30, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

            elapsed = time.time() - tiempo_inicio
            progreso = min(elapsed / tiempo_max, 1.0)
            bar_width = int((1 - progreso) * (vis_width - 100))
            cv2.rectangle(vis_area, (50, 20), (vis_width - 50, 40), (80, 80, 80), -1)
            color_barra = (244, 180, 0)
            cv2.rectangle(vis_area, (50, 20), (50 + bar_width, 40), color_barra, -1)
            cv2.rectangle(vis_area, (50, 20), (vis_width - 50, 40), (255, 255, 255), 2)

            real_count = list(respuestas.values()).count("REAL")
            ia_count = list(respuestas.values()).count("IA")
            cv2.putText(vis_area, f"REAL: {real_count}", (60, vis_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
            cv2.putText(vis_area, f"IA: {ia_count}", (vis_width - 300, vis_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

            # --- Combina imagen de pregunta + área de visualización ---
            separador = np.ones((vis_height, 20, 3), dtype=np.uint8) * 220
            if img_q.shape[0] != vis_area.shape[0]:
                img_q = cv2.resize(img_q, (img_q.shape[1], vis_area.shape[0]))
            combined = np.hstack((img_q, separador, vis_area))

            cv2.namedWindow("QUIZZ VISUAL", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("QUIZZ VISUAL", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("QUIZZ VISUAL", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                return

            respuestas_por_jugador = respuestas
            color_jugador.update(colores_por_id)

        correcta = pregunta["answer"]
        for jid, resp in respuestas_por_jugador.items():
            if resp == correcta:
                puntaje_jugador[jid] = puntaje_jugador.get(jid, 0) + 1
            else:
                puntaje_jugador.setdefault(jid, 0)

        aciertos = list(respuestas_por_jugador.values()).count(correcta)
        errores = len(respuestas_por_jugador) - aciertos
        total_correctas += aciertos
        total_incorrectas += errores

        print(f"✅ Aciertos: {aciertos}, ❌ Errores: {errores} (Correcta: {correcta})")

    mostrar_grafico_final(total_correctas, total_incorrectas)
    mostrar_ranking_individual(puntaje_jugador, color_jugador)

# === Bucle principal con opción a repetir ===
while True:
    ejecutar_quizz()
    print("\n¿Repetir juego? (s/n): ", end="")
    if input().lower() != "s":
        break

cap.release()
cv2.destroyAllWindows()
