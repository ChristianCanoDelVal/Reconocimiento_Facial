# Sistema de Control de Acceso por Reconocimiento Facial en Tiempo Real

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0.1-orange.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.100-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.1-yellow.svg)

Sistema de visión artificial y biometría diseñado para validar en tiempo real el acceso a instalaciones seguras mediante el flujo de una webcam o vídeo. El sistema discrimina entre personal autorizado (acceso permitido) e intrusos (acceso denegado) mediante cálculo de distancias biométricas sobre embeddings faciales.

---

## Arquitectura del Pipeline

El sistema implementa un pipeline desacoplado en dos etapas para optimizar la latencia y la precisión:

1. **Detección Facial:**
   * **MediaPipe (`BlazeFace`):** Procesa cada fotograma para extraer la región del rostro (`bounding box`). 
   * Se ejecuta de manera nativa y ultraligera, permitiendo altas tasas de FPS sin saturar los recursos computacionales.

2. **Reconocimiento Biométrico:**
   * **DeepFace (`VGG-Face`):** Extrae un vector numérico (*embedding*) representativo de la identidad.
   * Se utiliza `detector_backend="skip"` para transferir directamente la matriz recortada por MediaPipe, evitando análisis redundantes o conflictos de tensores.

3. **Autenticación Vectorial Matemática:**
   * Comparación contra la base de datos local serializada (`embeddings.pkl`) mediante distancia euclidiana ($L_2$).
   * **Umbral de Decisión Calibrado:** Un valor empírico óptimo separa las coincidencias de un mismo usuario frente a intrusos no registrados.

---

## Características Principales

* **Inicialización Anticipada (*Warm-up*):** Inicialización previa de la red neuronal en memoria para eliminar la latencia de inferencia inicial (*cold start*) al detectar una persona.
* **Procesamiento de Video Híbrido:** Capacitado para alimentarse del flujo en vivo de una cámara web o archivos de vídeo.
* **Soporte Híbrido CPU / GPU (Fallback Automático):** Detección dinámica del hardware en la inicialización; utiliza aceleración CUDA si está presente o conmuta a ejecución optimizada en CPU sin interrupciones.
* **Gestión Modular por Consola:** Subcomandos basados en `argparse` para administrar altas y bajas de identidades sin reentrenar modelos.

---

## Estructura del Repositorio

```text
├── videos/
│   ├── prueba.mp4                # Muestra de vídeo de prueba (autorizado)
├── imagenes_trabajadores/
│   ├── Christian_congafas.jpg    # Foto de empleado
│   ├── Christian_singafas.jpg    # Foto de empleado
│   └── foto_google1.jpg          # Foto de empleado
├── blaze_face_short_range.tflite # Modelo detector de MediaPipe
├── embeddings.pkl                # Base de datos vectorial de identidades
├── main.py                       # Pipeline principal de inferencia en tiempo real
├── registrar.py                  # Gestor de usuarios (CRUD de embeddings)
├── requirements.txt              # Dependencias del proyecto
├── requirements_singpu.txt       # Dependencias purificadas y ligeras (Recomendado para Portátiles/CPU)
├── environment.yml               # Clon exacto del entorno Conda para máxima reproducibilidad
└── README.md                     # Documentación técnica
