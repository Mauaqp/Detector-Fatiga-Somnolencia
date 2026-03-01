# Driver Drowsiness Detection / Detección de Fatiga y Somnolencia

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

---

## Instalación (IMPORTANTE - Leer completamente)

### Requisitos Previos

Este proyecto requiere paquetes con dependencias nativas (cmake, dlib, opencv) que pueden presentar dificultades en algunos sistemas. Siga las instrucciones según su caso:

### Opción 1: Instalación Limpia (Windows)

```bash
# 1. Crear entorno virtual (RECOMENDADO)
python -m venv venv
venv\Scripts\activate

# 2. Instalar dependencias en orden EXACTO
pip install --upgrade pip setuptools wheel
pip install cmake==3.18.0
pip install numpy==1.26.4
pip install opencv-python==4.8.0.74
pip install dlib==19.24.1
pip install imutils==0.5.4
pip install scipy==1.11.4
pip install Pillow
```

### Opción 2: Si tiene problemas con dlib

**Problema**: dlib requiere un compilador C++ para instalarse desde código fuente.

**Solución 1**: Instalar versión pre-compilada
```bash
pip install dlib==19.24.1 --verbose
```

**Solución 2**: Usar wheel pre-compilado
```bash
# Descargar desde: https://pypi.org/simple/dlib/
# Elegir la versión compatible con su Python
pip install /ruta/al/archivo.whl
```

### Opción 3: Solución de problemas comunes

#### Error: "Microsoft Visual C++ Build Tools"
```bash
# Instalar Visual Build Tools desde:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Seleccionar "Desktop development with C++"
```

#### Error: "numpy 2.0 no compatible"
```bash
# IMPORTANTE: dlib NO es compatible con NumPy 2.0
# SIEMPRE usar:
pip install numpy==1.26.4

# Verificar versión instalada
python -c "import numpy; print(numpy.__version__)"
# Debe mostrar: 1.26.4
```

#### Error: "No module named 'cv2'"
```bash
# Reinstalar opencv
pip uninstall opencv-python
pip install opencv-python==4.8.0.74
```

### Verificación de Instalación

```bash
# Ejecutar este comando para verificar
python -c "
import numpy as np
import cv2
import dlib
import imutils
import scipy
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'dlib: {dlib.__version__}')
print(f'imutils: {imutils.__version__}')
print(f'scipy: {scipy.__version__}')
"
```

### Archivos Requeridos

El proyecto incluye el modelo de 68 puntos faciales:
```
dlib_shape_predictor/
└── shape_predictor_68_face_landmarks.dat  (~87MB)
```

### Ejecución

```bash
# Acticar entorno virtual (si lo creó)
venv\Scripts\activate

# Ejecutar GUI
python DrowsinessDetectorGUI.py
```

---

## English

### Project Description

**Driver Drowsiness Detection** is a real-time monitoring system that detects driver fatigue using computer vision. The system analyzes eyes, mouth, and head position to detect signs of drowsiness and alert the driver.

---

## Installation (IMPORTANT - Read completely)

### Prerequisites

This project requires packages with native dependencies (cmake, dlib, opencv) that may present difficulties on some systems. Follow the instructions according to your case:

### Option 1: Clean Install (Windows)

```bash
# 1. Create virtual environment (RECOMMENDED)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies in EXACT order
pip install --upgrade pip setuptools wheel
pip install cmake==3.18.0
pip install numpy==1.26.4
pip install opencv-python==4.8.0.74
pip install dlib==19.24.1
pip install imutils==0.5.4
pip install scipy==1.11.4
pip install Pillow
```

### Option 2: If you have problems with dlib

**Problem**: dlib requires a C++ compiler to install from source.

**Solution 1**: Install pre-compiled version
```bash
pip install dlib==19.24.1 --verbose
```

**Solution 2**: Use pre-compiled wheel
```bash
# Download from: https://pypi.org/simple/dlib/
# Choose the version compatible with your Python
pip install /path/to/file.whl
```

### Option 3: Troubleshooting

#### Error: "Microsoft Visual C++ Build Tools"
```bash
# Install Visual Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++"
```

#### Error: "numpy 2.0 not compatible"
```bash
# IMPORTANT: dlib is NOT compatible with NumPy 2.0
# ALWAYS use:
pip install numpy==1.26.4

# Verify installed version
python -c "import numpy; print(numpy.__version__)"
# Should show: 1.26.4
```

#### Error: "No module named 'cv2'"
```bash
git clone https://github.com/Mauaqp/Detector-Fatiga-Somnolencia.git
cd Driver-Drowsiness-Detection
```

### Installation Verification

```bash
# Run this command to verify
python -c "
import numpy as np
import cv2
import dlib
import imutils
import scipy
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'dlib: {dlib.__version__}')
print(f'imutils: {imutils.__version__}')
print(f'scipy: {scipy.__version__}')
"
```

### Required Files

The project includes the 68-point facial landmark model:
```
dlib_shape_predictor/
└── shape_predictor_68_face_landmarks.dat  (~87MB)
```

### Running

```bash
# Activate virtual environment (if you created one)
venv\Scripts\activate

# Run GUI
python DrowsinessDetectorGUI.py
```

---

## Uso de la Interfaz / GUI Usage

### Español

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

---

## Problemas Conocidos / Known Issues

| Problema | Solución |
|----------|----------|
| NumPy 2.0 incompatible | Usar `numpy==1.26.4` |
| dlib build fails | Instalar Visual C++ Build Tools |
| OpenCV not found | `pip install opencv-python==4.8.0.74` |
| No cameras detected | Normal en PC sin webcam |

---

## Algoritmo de Detección / Detection Algorithm

### Español

El sistema utiliza un enfoque trifuncional:

1. **Eye Aspect Ratio (EAR)**: Mide la relación de aspecto de los ojos
   - Si EAR < 0.25 durante 3 frames consecutivos → Ojos cerrados

2. **Mouth Aspect Ratio (MAR)**: Mide la apertura de la boca
   - Si MAR > 0.79 → Bostezo detectado

3. **Pose de Cabeza**: Estima la inclinación de la cabeza
   - Utiliza Perspective-n-Point (PnP) para calcular orientación 3D

### English

The system uses a threefold approach:

1. **Eye Aspect Ratio (EAR)**: Measures eye aspect ratio
   - If EAR < 0.25 for 3 consecutive frames → Eyes closed

2. **Mouth Aspect Ratio (MAR)**: Measures mouth opening
   - If MAR > 0.79 → Yawning detected

3. **Head Pose**: Estimates head tilt
   - Uses Perspective-n-Point (PnP) to calculate 3D orientation

---

## Estructura del Proyecto / Project Structure

```
Driver-Drowsiness-Detection/
├── .gitignore
├── README.md
├── Requirements.txt
├── DrowsinessDetectorGUI.py    # Interfaz gráfica / GUI
├── DriverDrowsinessDetection.py # Script original / Original script
├── EAR.py                       # Cálculo EAR / EAR calculation
├── MAR.py                       # Cálculo MAR / MAR calculation
├── HeadPose.py                  # Pose de cabeza / Head pose
├── img/
│   └── isologo color.png       # Logo
└── dlib_shape_predictor/
    └── shape_predictor_68_face_landmarks.dat  # Modelo ML / ML model
```

---

## Licencia / License

MIT License

---

## Créditos / Credits

- Desarrollador/Developer: [Mauricio Peraltilla Cuadros](https://github.com/Mauaqp)
- Créditos especiales/Special credits: [Neelanjan Manna](https://github.com/neelanjan00)
- Modelo de puntos faciales/Facial landmark model: [dlib](http://dlib.net/)
