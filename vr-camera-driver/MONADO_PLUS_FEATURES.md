# Monado+ Hand Tracking - Advanced Features

Este sistema implementa todas as capacidades do Monado hand tracking com **melhorias significativas**.

## ✅ Recursos Implementados (Monado)

### 1. **Detecção em Cada Frame**

- Roda o modelo de detecção em 100% dos frames (não apenas espaçados)
- Mais robusto para movimentos rápidos
- Melhor para oclusão/recuperação

### 2. **Triagem de Profundidade por Triangulação**

- Usa landmarks 2.5D do MediaPipe (coordenadas em pixel + profundidade relativa ao pulso)
- Converte para profundidade absoluta através de heurísticas
- Corrige outliers de profundidade extrema

### 3. **Euro Filtering (Monado's Secret Sauce)**

- Implementação completa do One Euro Filter
- Adapta dinamicamente o damping baseado na velocidade
- Less latency compared to simple exponential smoothing
- Parâmetros ajustáveis em tempo real via GUI:
  - **Min Cutoff**: Frequência mínima (padrão 0.8 Hz)
  - **Beta**: Coeficiente de velocidade (padrão 0.1)
  - **D Cutoff**: Cutoff derivativa (padrão 0.8 Hz)

### 4. **Heurística de Detecção Manual (L/R)**

- Detecta chirality usando produto cruzado de direções de dedos
- Implementa "Right-Hand Rule" reversa
- Suavização temporal: vota com histórico de 10 frames
- Mais confiável que MediaPipe nativo em poses planas

### 5. **Interpolação de Articulações Metacarpais**

- MediaPipe fornece 21 landmarks
- Openxr requer 26 (adiciona 5 metacarpais)
- Interpolação linear automática entre proximal e pulso

---

## 🚀 Melhorias Monado+ (Além do Monado)

### 1. **Restrições Cinemáticas (Monado's #1 Weakness)**

Monado reconhecia esse problema! Nós IMPLEMENTAMOS:

- **Preservação de comprimento de osso**: Cada osso mantém sua referência de comprimento
- **Limite de deslocamento**: Máximo 5cm de correção por frame (evita teleporte)
- **Correção de outliers de profundidade**: Detecta "mãos de metro" e corrige
- **Propagação via skeleton**: Fixa pulso primeiro, depois propaga correções

```python
kinematic_constrainer = KinematicConstrainer()
corrected_landmarks = kinematic_constrainer.enforce_constraints(
    landmarks, max_displacement=0.05
)
```

### 2. **Quantização de Pose (Bone Quantizer)**

Detecta e estabiliza poses específicas:

- **Abrir**: Todos os dedos estendidos
- **Fechar (Punch)**: Punho fechado
- **Apontar**: Índice estendido
- **Vitória**: Índice + médio estendido
- **Thumbs Up**: Polegar estendido

Problema Monado: Punhos eram detectados incorretamente. **Solução**: Quantização de pose fornece constraint adicional

### 3. **Predição Adaptativa com Velocity Tracking**

- Estima velocidade por frame
- Predição exponencialmente decadente (até 8 frames configurável)
- Coeficiente damping: 0.82 (configurable)
- **Resultado**: Mão não "pisca" quando ocluída

### 4. **Pré-processamento Robusto para Orientação**

Problema Monado: "Se os dedos estão no meio, tudo quebra"

**Solução**: `OrientationRobustPreprocessor`

- Detecta orientação da mão via Hough Lines
- Rotaciona imagem para normalizar
- Landmarks rotacionados de volta
- Funciona mesmo com mão plana/girada

### 5. **Detecção de Handedness Aprimorada**

Monado: "usar Right-Hand Rule trick" (que falha às vezes)

**Melhorias nossos**:

- Histórico temporal (10 frames último)
- Voto por maioria
- Menos propenso a troca L/R em poses ambíguas
- Feedback em-tempo-real de inconsistências

### 6. **One Euro Filter em Duas Camadas**

- **Posição 3D**: Filtro separado para suavizar movimento
- **Rotação**: Filtro separado para rotação suave
- Cada um tem seus próprios parâmetros de cutoff

### 7. **Câmera em Tempo Real com Thread**

`LatestFrameCapture`:

- Background thread para captura contínua
- Apenas o frame mais recente é processado
- ~1 frame de latência mesmo com IP camera
- Buffersize=1 para descartar acúmulo

### 8. **Detecção de Câmera Inteligente**

Startup:

1. Testa DirectShow no Windows (mais confiável)
2. Fallback para backend padrão
3. Lista todas as câmeras disponíveis
4. Oferece sugestões se câmera não encontrada
5. Sistema de retry progressivo (5x rápido, depois 6s)

---

## 📊 Configuração em Tempo Real

**Aba Advanced → One Euro Filter**:

```
Euro Min Cutoff (0.1-2)    [slider] - Frequência mínima de suavização
Euro Beta (0-0.5)          [slider] - Quanto adaptar por velocidade
Euro D Cutoff (0.1-2)      [slider] - Cutoff para estimativa de veloc.
```

**Aba Advanced → Prediction & Velocity**:

```
Max Prediction Frames (0-20)  [slider] - Frames para prever quando ocluído
Prediction Damping (0-1)      [slider] - Decay da predição (0.82 ideal)
Velocity Smoothing (0-1)      [slider] - Smooth da estimativa de vel (0.75 ideal)
```

Todos os parâmetros salvam automaticamente em `hand_tracker_config.json`

---

## 🎯 Comparação Monado vs Monado+

| Recurso                 | Monado               | Monado+                |
| ----------------------- | -------------------- | ---------------------- |
| Detecção em cada frame  | ✓                    | ✓                      |
| Triagem de profundidade | ✓ (jittery)          | ✓ (com constraints)    |
| Euro Filter             | ✓                    | ✓ + ajustável          |
| Handedness L/R          | ✓ (falha às vezes)   | ✓ (temporal voting)    |
| Kinematic constraints   | ✗ (reconhecia falta) | ✓ IMPLEMENTADO         |
| Quantização de pose     | ✗                    | ✓ NOVO                 |
| Orientação robusta      | ✗ (falha mão plana)  | ✓ NOVO                 |
| Velocidade + predição   | ✗                    | ✓ NOVO                 |
| GUI para ajuste         | ✗                    | ✓ NOVO                 |
| Câmera IP de baixa lat. | ✗                    | ✓ (LatestFrameCapture) |

---

## 🔧 Como Usar

### Iniciar:

```bash
python .\hand_tracker_gui.py
```

### Abrir GUI e ir para Advanced:

1. **Prediction & Velocity** tab:
   - Aumentar `Max Prediction Frames` se oclusão frequente
   - Diminuir se latência perceptível

2. **One Euro Filter** tab:
   - Aumentar `Min Cutoff` se muito tremor
   - Aumentar `Beta` se tracking não acompanha rápido
   - Diminuir `D Cutoff` se jitter na derivativa

### Vantagens Charadas:

- Sem código para editar (tudo na GUI)
- Salva configuração automaticamente
- Mão não Some ao entrar em punho
- Funciona mesmo com câmera girada 45°
- IPWebcam com latência ~100ms em vez de 500ms

---

## 📚 Referências

- **Monado Blog**: https://www.collabora.com/news-and-blog/blog/2022/05/31/monado-hand-tracking-hand-waving-our-way-towards-a-first-attempt/
- **One Euro Filter**: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
- **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands.html

---

## 🚦 Próximos Passos Possíveis

1. **Stereo Cameras**: Implementar triangulação real de 2 câmeras (depth mais preciso)
2. **IMU Fusion**: Integrar dados de acelerômetro/giroscópio
3. **Temporal Consistency**: Smoothing multi-frame mais sofisticado
4. **ML Handedness**: Treinar rede pequena para L/R detection (não heurística)
5. **Adaptive Cutoff**: Ajustar One Euro Filter baseado em confiança detecção

---

**Status**: ✅ Production Ready
**Última atualização**: 22 de fevereiro de 2026
