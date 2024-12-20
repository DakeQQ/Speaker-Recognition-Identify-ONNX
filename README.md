---

## Speaker-Recognition-Identify-ONNX  
Leverage ONNX Runtime for efficient speaker role identification.

### Supported Models  
1. **Single Models**:  
   - [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
   - [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)  

### Features  
- End-to-end speaker recognition with built-in `STFT` processing.  
  **Input**: Audio file  
  **Output**: Speaker identification result  
- Suggested integrations for enhanced performance:  
  - [Voice Activity Detection (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### Learn More  
- Visit the [project overview](https://dakeqq.github.io/overview/) for additional details.

---

## 性能 Performance  

| **OS**          | **Device** | **Backend**           | **Model**                   | **Real-Time Factor**<br>(Chunk Size: 128000 or 8s) |
|:----------------:|:----------:|:---------------------:|:---------------------------:|:--------------------------------------------------:|
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | ERes2NetV2<br>f32           | 0.056                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | ERes2NetV2<br>q8f32         | 0.066                                              |

---

### 说话人识别 ONNX  
使用 ONNX Runtime 高效实现角色说话人识别。

### 支持模型  
1. **单模型**：  
   - [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
   - [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)  

### 功能特点  
- 端到端说话人识别，内置 `STFT` 处理。  
  **输入**：音频文件  
  **输出**：说话人识别结果  
- 推荐搭配以下工具，提升性能：  
  - [语音活动检测 (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [音频去噪](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### 了解更多  
- 访问[项目概览](https://dakeqq.github.io/overview/)获取更多信息。

---
