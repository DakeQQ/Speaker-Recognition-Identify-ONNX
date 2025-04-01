---

## Speaker-Recognition-Identify-ONNX  
Leverage ONNX Runtime for efficient speaker role identification and speaker change detection.

### Supported Models  
- [Speaker Identification - ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
- [Speaker Identification -  ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)
- [Speaker Change Detection - CAM++_Transformer](https://modelscope.cn/models/iic/speech_campplus-transformer_scl_zh-cn_16k-common)

### Features  
- End-to-end speaker recognition with built-in `STFT` processing.  
  **Input**: Audio file  
  **Output**: Speaker identification result / Speaker change position.
- Suggested integrations for enhanced performance:  
  - [Voice Activity Detection (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### Downloads
 - [Link](https://drive.google.com/drive/folders/1tm_i0HqjDJCKCXwCNV7rS5TW0WG4NcfW?usp=drive_link)

### Learn More  
- Visit the [project overview](https://github.com/DakeQQ?tab=repositories) for additional details.

---

## 性能 Performance  

| **OS**          | **Device** | **Backend**           | **Model**                   | **Real-Time Factor**<br>(Chunk Size: 128000 or 8s) |
|:----------------:|:----------:|:---------------------:|:---------------------------:|:--------------------------------------------------:|
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | ERes2NetV2<br>f32           | 0.056                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | ERes2NetV2<br>q8f32         | 0.066                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | CAM++_Transformer<br>f32    | 0.01                                               |

---

### 说话人识别 ONNX  
使用 ONNX Runtime 高效实现角色说话人识别 & 说话人转变点检测。

### 支持模型  
- [说话人识别 - ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
- [说话人识别 - ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)
- [说话人转变点检测 - CAM++_Transformer](https://modelscope.cn/models/iic/speech_campplus-transformer_scl_zh-cn_16k-common)

### 功能特点  
- 端到端说话人识别，内置 `STFT` 处理。  
  **输入**：音频文件  
  **输出**：说话人识别结果 / 说话人转变点
- 推荐搭配以下工具，提升性能：  
  - [语音活动检测 (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [音频去噪](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### Downloads
 - [Link](https://drive.google.com/drive/folders/1tm_i0HqjDJCKCXwCNV7rS5TW0WG4NcfW?usp=drive_link)

### 了解更多  
- 访问[项目概览]([https://dakeqq.github.io/overview/](https://github.com/DakeQQ?tab=repositories))获取更多信息。

---
