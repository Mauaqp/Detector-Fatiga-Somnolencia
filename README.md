# Driver Drowsiness Detection

> Sistema de detección de fatiga y somnolencia en tiempo real para conductores / Real-time driver drowsiness detection system

[Español](#español) | [English](#english)

---

## Español

### Descripción del Proyecto

**Detección de Fatiga y Somnolencia** es un sistema de detección en tiempo real que monitorea el estado de fatiga del conductor utilizando visión por computadora. El sistema analiza los ojos, la boca y la posición de la cabeza para detectar signos de somnolencia y alertar al conductor. Posese una interfaz gráfica que permite la carga y análisis de videos pre grabados, así como la exportación con los overlays de detección.

### Características

- 🔍 **Detección de rostros** utilizando dlib (HOG + 68 puntos de referencia faciales)
- 👁️ **Detección de ojos cerrados** mediante el cálculo del Eye Aspect Ratio (EAR)
- 👄 **Detección de bostezos** mediante el cálculo del Mouth Aspect Ratio (MAR)
- 📐 **Estimación de pose de cabeza** para detectar inclinación hacia adelante
- 🎥 **Soporte para cámara en vivo y video pregrabado**
- 🌐 **Interfaz bilingual** (Español/Inglés)
- 💾 **Exportación de video** analizado en formato MP4

### Requisitos del Sistema

```
numpy==1.26.4      # Versión específica requerida (ver notas)
opencv-python==4.13.0.92
dlib==19.24.1
imutils==0.5.4
scipy==1.15.3
Pillow (PIL)
```

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Mauaqp/Detector-Fatiga-Somnolencia.git
cd Driver-Drowsiness-Detection
```

2. Instalar dependencias:
```bash
pip install -r Requirements.txt
```

3. Ejecutar la aplicación:
```bash
python DrowsinessDetectorGUI.py
```

### Uso de la Interfaz

1. **Seleccionar Fuente de Video**:
   - Elegir una cámara del dropdown
   - O explorar y seleccionar un archivo de video (.mp4, .avi, .mov)

2. **Controles de Video** (solo para video):
   - Reproducir/Pausar
   - Adelantar +10s / Retroceder -10s
   - Barra de progreso

3. **Exportar Video**:
   - Haga clic en "Exportar MP4" para procesar y guardar el video completo con los análisis

4. **Cambio de Idioma**:
   - Use el menú para cambiar entre Español e Inglés

### Problemas Conocidos

- **NumPy 2.0**: dlib no es compatible con NumPy 2.0. Use `numpy==1.26.4`
- **Cámaras sin conectar**: Los errores de cámara son esperados en PCs sin webcam

### Estructura del Proyecto

```
Driver-Drowsiness-Detection/
├── DrowsinessDetectorGUI.py    # Interfaz gráfica principal
├── DriverDrowsinessDetection.py # Script original de consola
├── EAR.py                     # Cálculo del Eye Aspect Ratio
├── MAR.py                     # Cálculo del Mouth Aspect Ratio
├── HeadPose.py                # Estimación de pose de cabeza
├── Requirements.txt            # Dependencias del proyecto
├── img/
│   └── isologo color.png      # Logo de la aplicación
├── dlib_shape_predictor/
│   └── shape_predictor_68_face_landmarks.dat  # Modelo de 68 puntos
└── README.md                  # Este archivo
```

### Algoritmo de Detección

El sistema utiliza un enfoque trifuncional:

1. **Eye Aspect Ratio (EAR)**: Mide la relación de aspecto de los ojos
   - Si EAR < 0.25 durante 3 frames consecutivos → Ojos cerrados

2. **Mouth Aspect Ratio (MAR)**: Mide la apertura de la boca
   - Si MAR > 0.79 → Bostezo detectado

3. **Pose de Cabeza**: Estima la inclinación de la cabeza
   - Utiliza Perspective-n-Point (PnP) para calcular orientación 3D

### Licencia

MIT License

---

## English

### Project Description

**Driver Drowsiness Detection** is a real-time monitoring system that detects driver fatigue using computer vision. The system analyzes eyes, mouth, and head position to detect signs of drowsiness and alert the driver.

### Features

- 🔍 **Face detection** using dlib (HOG + 68 facial landmarks)
- 👁️ **Closed eye detection** using Eye Aspect Ratio (EAR)
- 👄 **Yawning detection** using Mouth Aspect Ratio (MAR)
- 📐 **Head pose estimation** to detect forward tilt
- 🎥 **Live camera and video file support**
- 🌐 **Bilingual interface** (Spanish/English)
- 💾 **Video export** to MP4 format

### System Requirements

```
numpy==1.26.4      # Specific version required (see notes)
opencv-python==4.13.0.92
dlib==19.24.1
imutils==0.5.4
scipy==1.15.3
Pillow (PIL)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mauaqp/Detector-Fatiga-Somnolencia.git
cd Driver-Drowsiness-Detection
```

2. Install dependencies:
```bash
pip install -r Requirements.txt
```

3. Run the application:
```bash
python DrowsinessDetectorGUI.py
```

### GUI Usage

1. **Select Video Source**:
   - Choose a camera from the dropdown
   - Or browse and select a video file (.mp4, .avi, .mov)

2. **Video Controls** (video only):
   - Play/Pause
   - Forward +10s / Backward -10s
   - Progress bar

3. **Export Video**:
   - Click "Export MP4" to process and save the complete analyzed video

4. **Language Change**:
   - Use the menu to switch between Spanish and English

### Known Issues

- **NumPy 2.0**: dlib is not compatible with NumPy 2.0. Use `numpy==1.26.4`
- **No cameras connected**: Camera errors are expected on PCs without webcam

### Project Structure

```
Driver-Drowsiness-Detection/
├── DrowsinessDetectorGUI.py    # Main GUI application
├── DriverDrowsinessDetection.py # Original console script
├── EAR.py                     # Eye Aspect Ratio calculation
├── MAR.py                     # Mouth Aspect Ratio calculation
├── HeadPose.py                # Head pose estimation
├── Requirements.txt           # Project dependencies
├── img/
│   └── isologo color.png     # Application logo
├── dlib_shape_predictor/
│   └── shape_predictor_68_face_landmarks.dat  # 68-point model
└── README.md                  # This file
```

### Detection Algorithm

The system uses a threefold approach:

1. **Eye Aspect Ratio (EAR)**: Measures eye aspect ratio
   - If EAR < 0.25 for 3 consecutive frames → Eyes closed

2. **Mouth Aspect Ratio (MAR)**: Measures mouth opening
   - If MAR > 0.79 → Yawning detected

3. **Head Pose**: Estimates head tilt
   - Uses Perspective-n-Point (PnP) to calculate 3D orientation

### License

MIT License

---

## Credits

- Desarrollador: [Mauricio Peraltilla Cuadros](https://github.com/Mauaqp)
- Creditos Especiales : [Neelanjan Manna](https://github.com/neelanjan00)
- Facial landmark model: [dlib](http://dlib.net/)
