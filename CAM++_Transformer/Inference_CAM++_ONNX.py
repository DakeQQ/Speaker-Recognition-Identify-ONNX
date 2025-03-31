import time
import numpy as np
import onnxruntime
from pydub import AudioSegment


model_path = "/home/DakeQQ/Downloads/speech_campplus-transformer_scl_zh-cn_16k-common"       # The CAM++_Transformer download path.
onnx_model_A = "/home/DakeQQ/Downloads/CAM_Optimized/CAM.onnx"                               # The exported onnx model path.
test_audio = model_path + "/examples/scl_example2.wav"                                       # The test audio list.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.


def normalize_to_int16(audio_int64):
    max_val = np.max(np.abs(audio_int64.astype(np.float32)))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio_int64 * float(scaling_factor)).astype(np.int16)


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
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int64)
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
        results.append(speech_change_start)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH

print(f"\nInference Time Cost: {time.time() - start_time:.3f} Seconds")
if results:
    print("\nSpeaker change found:")
    for i in results:
        print(f"\n  Change_Point = {i}")
else:
    print("No speaker change detected.")
