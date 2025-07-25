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
INPUT_AUDIO_LENGTH = 160000                                 # The maximum input audio length.
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
VOICE_EMBED_DIM = 192                                       # Model setting.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


python_package_path = site.getsitepackages()[-1]
shutil.copyfile('./modeling_modified/DTDNN.py',  python_package_path + "/modelscope/models/audio/sv/DTDNN.py")
shutil.copyfile('./modeling_modified/speaker_change_locator.py', python_package_path + "/modelscope/models/audio/sv/speaker_change_locator.py")
from modelscope.models.base import Model


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class PosEncoding(torch.nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_word_vec, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_word_vec))
        pos_enc = torch.zeros(max_seq_len, d_word_vec)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term[:d_word_vec // 2])
        pad_row = torch.zeros(1, d_word_vec)
        pos_enc = torch.cat([pad_row, pos_enc], dim=0)
        self.pos_enc = torch.nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = torch.nn.Parameter(pos_enc, requires_grad=False)
        self.arrange = torch.arange(1, self.pos_enc.num_embeddings).repeat(2, 1).to(torch.int16)

    def forward(self, input_len):
        input_pos = self.arrange[:, :input_len].int()
        return self.pos_enc(input_pos)


class CAMPPLUS(torch.nn.Module):
    def __init__(self, campplus, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis):
        super(CAMPPLUS, self).__init__()
        self.campplus = campplus
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.inv_int16 = float(1.0 / 32768.0)
        embed = torch.zeros((1, 1, campplus.model_config['anchor_size']), dtype=torch.int8)
        embed[:, :, 1::2] = 1
        self.anchors = torch.cat((embed, 1 - embed), dim=0)
        self.campplus.backend.pos_enc_plus = PosEncoding(2048, 256)

    def forward(self, audio, voice_embed_x, voice_embed_y, control_factor):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-5).log()
        mel_features = mel_features - mel_features.mean(dim=-1, keepdim=True)
        anchors = torch.cat((voice_embed_x, voice_embed_y), dim=0) * control_factor + (self.anchors * (1 - control_factor)).float()
        output = self.campplus(mel_features, anchors)
        argmax_values = output.argmax(dim=-1).int()
        output = torch.nonzero(argmax_values[1:] - argmax_values[:-1]).squeeze(-1)
        return output.shape[0].int(), output.int()


print('\nExport start ...\n')
with torch.inference_mode():
    from modelscope.utils.config import Config, ConfigDict
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = Model.from_pretrained(
        model_name_or_path=model_path,
        disable_update=True,
        device="cpu"
    ).eval()
    campplus = CAMPPLUS(model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    voice_embed_x = torch.ones((1, 1, VOICE_EMBED_DIM), dtype=torch.float32)
    voice_embed_y = torch.ones((1, 1, VOICE_EMBED_DIM), dtype=torch.float32)
    control_factor = torch.tensor([0], dtype=torch.int8)
    torch.onnx.export(
        campplus,
        (audio, voice_embed_x, voice_embed_y, control_factor),
        onnx_model_A,
        input_names=['audio', 'voice_embed_x', 'voice_embed_y', 'control_factor'],
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
    del voice_embed_x
    del voice_embed_y
    del control_factor
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
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
in_name_A3 = in_name_A[3].name
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
    final_slice = audio[:, :, -pad_amount:].astype(np.float32)
    white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    audio_float = audio.astype(np.float32)
    white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
    audio = np.concatenate((audio, white_noise), axis=-1)
aligned_len = audio.shape[-1]


# Start to run CAM++_Transformer
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
sample_rate_factor = np.array([SAMPLE_RATE * 0.01], dtype=np.int32)
bias = np.array([0], dtype=np.int32)                                  # Experience value
voice_embed_x = np.zeros((1, 1, VOICE_EMBED_DIM), dtype=np.float32)   # You can modify this with the outputs from the ERes2Net model.
voice_embed_y = np.zeros((1, 1, VOICE_EMBED_DIM), dtype=np.float32)   # You can modify this with the outputs from the ERes2Net model.
control_factor = np.array([0], dtype=np.int8)                         # If you are using the ERes2Net voice vector, set the value to 1; otherwise, set it to 0.
results = []
start_time = time.time()
while slice_end <= aligned_len:
    output_len, output = ort_session_A.run(
        [out_name_A0, out_name_A1],
        {
            in_name_A0: audio[:, :, slice_start: slice_end],
            in_name_A1: voice_embed_x,
            in_name_A2: voice_embed_y,
            in_name_A3: control_factor
        })
    if output_len != 0:
        speech_change_start = (output[0] + bias) * sample_rate_factor
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
