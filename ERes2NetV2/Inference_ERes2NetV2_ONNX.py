import time
import numpy as np
import onnxruntime
from pydub import AudioSegment


model_path = "/home/DakeQQ/Downloads/speech_eres2netv2_sv_zh-cn_16k-common"           # The SenseVoice download path.
onnx_model_A = "/home/DakeQQ/Downloads/ERes2NetV2_Optimized/ERes2NetV2.ort"           # The exported onnx model path.
test_audio = [model_path + "/examples/speaker2_a_cn_16k.wav", model_path + "/examples/speaker1_a_cn_16k.wav", model_path + "/examples/speaker1_b_cn_16k.wav"]   # The test audio list.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
SIMILARITY_THRESHOLD = 0.5              # Threshold to determine the speaker's identity.


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


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
