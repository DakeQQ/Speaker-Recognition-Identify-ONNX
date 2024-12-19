# Speaker-Recognition-Identify-ONNX
Uses ONNX Runtime for character role speaker identification.
1. Now support:
   - [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
   - [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)
2. This end-to-end version includes internal `STFT` processing. Input audio; output is speaker identification result.
3. It is recommended to work with the [VAD](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX) and the [denoised](https://github.com/DakeQQ/Audio-Denoiser-ONNX) model.
4. See more -> https://dakeqq.github.io/overview/

# Speaker-Recognition-Identify-ONNX
1. 现在支持:
   - [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
   - [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)
2. 这个端到端版本包括内部的 `STFT` 处理。输入为音频，输出为说话者识别结果。
3. 建议与 [VAD](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX) 和 [去噪模型](https://github.com/DakeQQ/Audio-Denoiser-ONNX) 一起使用。.
4. See more -> https://dakeqq.github.io/overview/

# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 128000 or 8s ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | ERes2NetV2<br>f32 | 0.056 |
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | ERes2NetV2<br>q8f32 | 0.066 |
