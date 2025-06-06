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

model_path = "/home/DakeQQ/Downloads/speech_eres2netv2_sv_zh-cn_16k-common"       # The ERes2NetV2 download path.
onnx_model_A = "/home/DakeQQ/Downloads/ERes2NetV2_ONNX/ERes2NetV2.onnx"           # The exported onnx model path.
test_audio = [model_path + "/examples/speaker2_a_cn_16k.wav", model_path + "/examples/speaker1_a_cn_16k.wav", model_path + "/examples/speaker1_b_cn_16k.wav"]    
# The test audio list.

ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 160000                                 # The maximum input audio length.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SPEAKERS = 50                                           # Maximum number of saved speaker features.
SIMILARITY_THRESHOLD = 0.5                                  # Threshold to determine the speaker's identity.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile('./modeling_modified/ERes2NetV2.py', site.getsitepackages()[-1] + "/modelscope/models/audio/sv/ERes2NetV2.py")
from modelscope.models.base import Model


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class ERES2NETV2(torch.nn.Module):
    def __init__(self, eres2netv2, stft_model, nfft_stft, stft_signal_len, n_mels, sample_rate, pre_emphasis):
        super(ERES2NETV2, self).__init__()
        self.eres2netv2 = eres2netv2
        self.stft_model = stft_model
        self.pre_emphasis = pre_emphasis
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.inv_int16 = float(1.0 / 32768.0)

    def forward(self, audio, saved_embed, num_speakers):
        audio = audio.float() * self.inv_int16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-5).log()
        mel_features -= mel_features.mean(dim=-1, keepdim=True)
        embed = self.eres2netv2(mel_features)
        score = self.cos_similarity(embed, saved_embed) if DYNAMIC_AXES else self.cos_similarity(embed, saved_embed[:num_speakers])
        score, target_idx = torch.max(score, dim=-1)
        return target_idx.int(), score, embed


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = Model.from_pretrained(
        model_name_or_path=model_path,
        disable_update=True,
        device="cpu",
    ).embedding_model.eval()
    eres2netv2 = ERES2NETV2(model, custom_stft, NFFT_STFT, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    saved_embed = torch.randn((MAX_SPEAKERS, model.embed_dim), dtype=torch.float32)
    num_speakers = torch.tensor([1], dtype=torch.int64)
    torch.onnx.export(
        eres2netv2,
        (audio, saved_embed, num_speakers),
        onnx_model_A,
        input_names=['audio', 'saved_embed', 'num_speakers'],
        output_names=['target_idx', 'score', 'embed'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'saved_embed': {0: 'max_speakers'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model
    del eres2netv2
    del audio
    del saved_embed
    del num_speakers
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
model_type = ort_session_A._inputs_meta[1].type
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
if isinstance(shape_value_in, str):
    in_name_A2 = None
    dynamic_axes = True
else:
    in_name_A2 = in_name_A[2].name
    dynamic_axes = False
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name


num_speakers = np.array([1], dtype=np.int64)  # At least 1.
if dynamic_axes:
    saved_embed = np.zeros((2, ort_session_A._inputs_meta[1].shape[1]), dtype=np.float32)  # At least 2.
    empty_space = np.zeros((1, ort_session_A._inputs_meta[1].shape[1]), dtype=np.float32)
else:
    saved_embed = np.zeros((ort_session_A._inputs_meta[1].shape[0], ort_session_A._inputs_meta[1].shape[1]), dtype=np.float32)
    empty_space = None
if "float16" in model_type:
    saved_embed = saved_embed.astype(np.float16)
    if dynamic_axes:
        empty_space = empty_space.astype(np.float16)


# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
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

    # Start to run ERes2NetV2
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    while slice_end <= aligned_len:
        start_time = time.time()
        input_feed = {
                    in_name_A0: audio[:, :, slice_start: slice_end],
                    in_name_A1: saved_embed
                }
        if not dynamic_axes:
            input_feed[in_name_A2] = num_speakers
        target_idx, score, embed = ort_session_A.run([out_name_A0, out_name_A1, out_name_A2], input_feed)
        end_time = time.time()
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        if score >= SIMILARITY_THRESHOLD:
            saved_embed[target_idx] = (saved_embed[target_idx] + embed) * 0.5
            print(f"\nLocate the identified speaker with ID = {target_idx}, Similarity = {score:.3f}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
        else:
            saved_embed[num_speakers] = embed
            print(f"\nIt's an unknown speaker. Assign it a new ID = {num_speakers[0]}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
            num_speakers += 1
            if dynamic_axes:
                saved_embed = np.concatenate((saved_embed, empty_space), axis=0)
        print("----------------------------------------------------------------------------------------------------------")
