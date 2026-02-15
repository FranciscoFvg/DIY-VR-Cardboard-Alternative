# CameraVR - Driver de Rastreamento VR por Câmera

Sistema completo de rastreamento 6DoF para **SteamVR** usando **câmera** e **MediaPipe Hands**.

## 📋 Arquitetura

```
Câmera → MediaPipe Hands → Pose 6DoF → Filtro → Driver OpenVR → SteamVR
```

## 🔧 Componentes

### 1. **CameraCapture** (`src/CameraCapture.cpp`)

- Captura frames da webcam em thread separada
- Configurável: resolução, FPS, índice da câmera

### 2. **HandPoseReceiver** (`src/HandPoseReceiver.cpp`)

- Recebe poses por UDP (enviadas pelo script MediaPipe)
- Suporta maos esquerda/direita
- Tolerante a perda de frames

### 3. **hand_tracker.py**

- Captura camera via OpenCV-Python
- Roda MediaPipe Hands
- Envia pose 6DoF por UDP

### 4. **PoseEstimator** (`src/PoseEstimator.cpp`)

- Converte poses OpenCV → Eigen → OpenVR
- Transforma coordenadas câmera → mundo VR
- Suporta offset de calibração

### 5. **PoseFilter** (`src/PoseFilter.cpp`)

- **Filtro complementar**: suaviza jitter
- **Média móvel**: reduz ruído
- **Kalman**: predição + correção

### 6. **VirtualController** (`driver/VirtualController.cpp`)

- Emula controlador VIVE no SteamVR
- Publica poses em tempo real
- Suporta inputs (botões, trigger, trackpad)

### 7. **CameraVRDriver** (`driver/CameraVRDriver.cpp`)

- Driver OpenVR completo
- Gerencia múltiplos controladores
- Entry point: `HmdDriverFactory`

## 📦 Dependências

### Necessárias

- **Eigen 3.x**
- **OpenVR SDK** (baixar de [github.com/ValveSoftware/openvr](https://github.com/ValveSoftware/openvr))
- **CMake 3.15+**
- **Visual Studio 2019+** (ou MinGW)
- **Python 3.10+**
- **MediaPipe** + **OpenCV-Python**

### Windows

```powershell
# Instalar vcpkg (gerenciador de pacotes C++)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Instalar dependencias C++
.\vcpkg install eigen3:x64-windows
```

## 🛠️ Compilar

```powershell
# 1. Baixar OpenVR SDK
cd libs
git clone https://github.com/ValveSoftware/openvr.git

# 2. Configurar CMake
cd ..
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]\scripts\buildsystems\vcpkg.cmake

# 3. Compilar
cmake --build . --config Release
```

## 🚀 Usar

### 1. Preparar ambiente Python

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install mediapipe opencv-python numpy
```

### 2. Iniciar rastreador de maos (MediaPipe)

```powershell
python hand_tracker.py
```

Se quiser rodar sem preview:

```powershell
python hand_tracker.py --no-preview
```

### 3. Instalar driver no SteamVR

```powershell
# Copiar DLL para pasta do SteamVR
mkdir "C:\Program Files (x86)\Steam\steamapps\common\SteamVR\drivers\cameravr\bin\win64"
copy build\Release\driver_cameravr.dll "C:\...\cameravr\bin\win64\"

# Criar driver.vrdrivermanifest
```

**driver.vrdrivermanifest:**

```json
{
  "alwaysActivate": true,
  "name": "cameravr",
  "directory": ""
}
```

### 4. Executar

```powershell
# Executar aplicacao
.\build\Release\CameraVRApp.exe

# Ou integrar na DLL para rodar automaticamente com SteamVR
```

## 📐 Ajustes

### Offset espacial

Se os controles aparecerem em posicao errada, ajuste o offset:

```cpp
poseEstimator.setCalibrationOffset(
    Eigen::Vector3d(0.0, 1.5, 0.0), // x, y, z em metros
    Eigen::Quaterniond(1, 0, 0, 0)  // rotacao
);
```

### Filtro

Ajuste `alpha` do filtro complementar (0-1):

- **Próximo de 1**: mais confiança na medição (rápido, mais jitter)
- **Próximo de 0**: mais suavização (lag, menos jitter)

```cpp
filter.setAlpha(0.7); // padrão
```

## 🎮 Input de botões (TODO)

Para emular botões, complete:

```cpp
controller->UpdateButtonState(vr::k_EButton_SteamVR_Trigger, true);
controller->UpdateTrigger(0.8f);
```

## 📝 Estrutura de arquivos

```
vr-camera-driver/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── CameraCapture.h
│   ├── HandPoseReceiver.h
│   ├── PoseEstimator.h
│   └── PoseFilter.h
├── src/
│   ├── CameraCapture.cpp
│   ├── HandPoseReceiver.cpp
│   ├── PoseEstimator.cpp
│   ├── PoseFilter.cpp
│   └── main.cpp
├── driver/
│   ├── CameraVRDriver.h
│   ├── CameraVRDriver.cpp
│   ├── VirtualController.h
│   └── VirtualController.cpp
├── hand_tracker.py
└── libs/
  └── openvr/  (baixar separadamente)
```

## 🐛 Troubleshooting

### Câmera não abre

- Verifique se está sendo usada por outro app
- Tente outro índice: `CameraCapture camera(1);`

### Maos nao detectadas

- Melhore iluminacao
- Aproxime as maos da camera
- Evite fundos com muito ruido visual

### Driver não aparece no SteamVR

- Verifique logs: `C:\Program Files (x86)\Steam\logs\vrserver.txt`
- Reinstale o driver: `vrpathreg adddriver [path]`

### Pose instável

- Aumente suavização: `filter.setAlpha(0.5);`
- Melhore calibração da câmera
- Use marcadores maiores/mais distantes

## 📚 Referências

- [OpenVR Wiki](https://github.com/ValveSoftware/openvr/wiki)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [SteamVR Driver Tutorial](https://github.com/ValveSoftware/openvr/wiki/Driver-Documentation)

## 📚 Referências

- [OpenVR Wiki](https://github.com/ValveSoftware/openvr/wiki)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [SteamVR Driver Tutorial](https://github.com/ValveSoftware/openvr/wiki/Driver-Documentation)

## 📄 Licença

MIT

---

**Criado para DIY VR Cardboard Alternative Project**
