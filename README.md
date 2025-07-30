# Quizz Visual: ¿IA o Real?

¡Juego interactivo multijugador para descubrir si una imagen fue creada por inteligencia artificial o por humanos!

---

## 🎯 Descripción

Este proyecto es un sistema de quizz visual local que utiliza la cámara web y técnicas de visión por computadora (MediaPipe + OpenCV) para:
- Mostrar imágenes (reales y generadas por IA) en pantalla dividida.
- Detectar y asignar respuestas a cada persona usando reconocimiento facial y de manos en tiempo real.
- Permitir respuestas simultáneas (levantar mano derecha = “Real”, izquierda = “IA”).
- Mostrar resultados, ranking individual y gráficos llamativos.
- Pensado para ferias, eventos educativos, museos y exhibiciones tecnológicas.

---

## 🚀 Funcionalidades principales

- Detección robusta de rostros y manos para evitar errores y duplicidad de respuestas.
- Pantalla dividida: imagen de la ronda y área amplia de participantes.
- Asignación de color único y etiquetas a cada jugador.
- Barra de tiempo animada, feedback visual, gráficos finales y ranking individual.
- Compatible con cualquier set de imágenes.

---

## 🛠️ Requisitos

- Python 3.8+
- Cámara web funcional
- Espacio suficiente para que múltiples personas puedan ser detectadas por la cámara

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd dibujador
```

### 2. Crear un entorno virtual

**En Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**En Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar la instalación

```bash
python main.py
```

## 🎮 Cómo usar

1. **Activar el entorno virtual** (si no está activado):
   - Windows: `.\venv\Scripts\Activate.ps1`
   - macOS/Linux: `source venv/bin/activate`

2. **Ejecutar el juego**:
   ```bash
   python main.py
   ```

3. **Controles del juego**:
   - **Mano derecha levantada**: Votar "Real" (imagen creada por humanos)
   - **Mano izquierda levantada**: Votar "IA" (imagen generada por inteligencia artificial)
   - **Espacio**: Pasar a la siguiente pregunta (solo para administrador)
   - **ESC**: Salir del juego

4. **Agregar nuevas imágenes**:
   - Coloca las imágenes en la carpeta `images/`
   - Edita el archivo `quizz_images.py` para agregar las nuevas preguntas

## 📁 Estructura del proyecto

```
dibujador/
├── main.py              # Archivo principal del juego
├── quizz_images.py      # Configuración de preguntas y respuestas
├── utils.py             # Funciones auxiliares
├── questions.py         # Funciones de manejo de preguntas
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Documentación
├── .gitignore          # Archivos a ignorar por Git
├── venv/               # Entorno virtual (creado después de la instalación)
├── images/             # Carpeta con imágenes del quizz
│   ├── h1.jpg
│   ├── ia1.jpg
│   ├── ia2.jpeg
│   └── ia3.jpg
└── __pycache__/        # Archivos compilados de Python
```

## 🔧 Dependencias

El proyecto utiliza las siguientes librerías principales:

- **OpenCV**: Para procesamiento de imágenes y captura de video
- **MediaPipe**: Para detección de rostros y manos en tiempo real
- **NumPy**: Para operaciones matemáticas con arrays
- **Matplotlib**: Para generación de gráficos y visualizaciones

Todas las dependencias se encuentran listadas en `requirements.txt`.

## 🚨 Solución de problemas

### Error de cámara
- Verifica que tu cámara web esté conectada y funcionando
- Asegúrate de que ninguna otra aplicación esté usando la cámara
- En Windows, verifica los permisos de cámara en Configuración > Privacidad

### Error de importación de MediaPipe
- Asegúrate de estar en el entorno virtual activado
- Reinstala MediaPipe: `pip uninstall mediapipe && pip install mediapipe`

### Rendimiento lento
- Cierra otras aplicaciones que puedan estar usando la cámara o CPU
- Ajusta la resolución de la cámara en el código si es necesario

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
