import gc
import time
import shutil
import site
import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/speech_campplus-transformer_scl_zh-cn_16k-common"       # The CAM++_Transformer download path.
onnx_model_A = "/home/DakeQQ/Downloads/CAM_ONNX/CAM.onnx"                                    # The exported onnx model path.
test_audio = model_path + "/examples/scl_example2.wav"                                       # The test audio list.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 160000 if not DYNAMIC_AXES else 16000  # Set for static axis export: the length of the audio input signal (in samples). Iy use DYNAMIC_AXES, Default to 320000, you can adjust it.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT = 400                                                  # Number of FFT components for the STFT process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.

python_package_path = site.getsitepackages()[-1]
shutil.copyfile('./modeling_modified/DTDNN.py',  python_package_path + "/modelscope/models/audio/sv/DTDNN.py")
shutil.copyfile('./modeling_modified/speaker_change_locator.py', python_package_path + "/modelscope/models/audio/sv/speaker_change_locator.py")
from modelscope.models.base import Model


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class CAMPPLUS(torch.nn.Module):
    def __init__(self, campplus, stft_model, nfft, n_mels, sample_rate, pre_emphasis):
        super(CAMPPLUS, self).__init__()
        self.campplus = campplus
        self.stft_model = stft_model
        self.pre_emphasis = pre_emphasis
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        embed = torch.zeros((1, 1, campplus.model_config['anchor_size']), dtype=torch.float32)
        embed[:, :, 1::2] = 1
        self.anchors = torch.cat((embed, 1 - embed), dim=0)

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-5).log()
        mel_features -= mel_features.mean(dim=-1, keepdim=True)
        output = self.campplus(mel_features, self.anchors)
        argmax_values = output.argmax(dim=-1).int()
        output = torch.nonzero(argmax_values[1:] - argmax_values[:-1]).squeeze(-1)
        return output.shape[0].int(), output.float()


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = Model.from_pretrained(
        model_name_or_path=model_path,
        disable_update=True,
        device="cpu",
    ).eval()
    campplus = CAMPPLUS(model, custom_stft,  NFFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        campplus,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=['output_len', 'output'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'output': {0: 'output_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model
    del campplus
    del audio
    gc.collect()
print('\nExport done!\n\nStart to run ERes2NetV2 by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
if isinstance(shape_value_in, str):
    dynamic_axes = True
else:
    dynamic_axes = False
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name

# Load the input audio
print(f"\nTest Input Audio: {test_audio}")
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
if dynamic_axes:
    INPUT_AUDIO_LENGTH = min(320000, audio_len)  # Default to 20 seconds audio, You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
if SLIDING_WINDOW <= 0:
    stride_step = INPUT_AUDIO_LENGTH
else:
    stride_step = SLIDING_WINDOW
if audio_len > INPUT_AUDIO_LENGTH:
    num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
    pad_amount = total_length_needed - audio_len
    final_slice = audio[:, :, -pad_amount:]
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    white_noise = (np.sqrt(np.mean(audio * audio)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


# Start to run CAM++_Transformer
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
sample_rate_factor = np.array([SAMPLE_RATE * 0.01], dtype=np.float32)
bias = np.array([0.0], dtype=np.float32)  # Experience value
results = []
start_time = time.time()
while slice_end <= aligned_len:
    output_len, output = ort_session_A.run(
        [out_name_A0, out_name_A1],
        {
            in_name_A0: audio[:, :, slice_start: slice_end]
        })
    if output_len != 0:
        speech_change_start = ((output[0] + bias) * sample_rate_factor).astype(np.int32)
        results.append(speech_change_start + slice_start)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH

print(f"\nInference Time Cost: {time.time() - start_time:.3f} Seconds")
if results:
    print("\nSpeaker change found:")
    for i in results:
        print(f"\n  Change_Point = {i}")
else:
    print("No speaker change detected.")
