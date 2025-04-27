import re
import time
import numpy as np
import onnxruntime


onnx_model_A = r"/home/DakeQQ/Downloads/Bert_Speaker_Diarization_Optimized/Bert_Speaker_Diarization.onnx"                    # The exported onnx model save path.
vocab_path = '/home/DakeQQ/Downloads/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese/vocab.txt'     # Set the path where the Bert model vocab.txt stored.
sentence = "你是如何看待这个问题的呢？这个问题挺好解决的，我们只需要增加停车位就行了。嗯嗯，好，那我们业主就放心了。"                          # The sentence for test.

ORT_Accelerate_Providers = ['CPUExecutionProvider']       # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                          # else keep empty.
MAX_THREADS = 8                                           # Max CPU parallel threads.
DEVICE_ID = 0                                             # The GPU id, default to 0.
TOKEN_UNKNOWN = 100                                       # The model parameter, do not edit it.
TOKEN_BEGIN = 101                                         # The model parameter, do not edit it.
TOKEN_END = 102                                           # The model parameter, do not edit it.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False                 # Enable it carefully
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,      # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'use_tf32': '0'
        }
    ]
else:
    # Please config by yourself for others providers.
    provider_options = None


# Read the model vocab.
with open(vocab_path, 'r', encoding='utf-8') as file:
    vocab = file.readlines()
vocab = np.array([line.strip() for line in vocab], dtype=np.str_)


# For Bert Model
def tokenizer(input_string, max_input_words, is_dynamic):
    input_ids = np.zeros((1, max_input_words), dtype=np.int32)
    punc_ids = input_ids  
    input_string = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|[^\w\s]', input_string.lower())
    input_ids[0] = TOKEN_BEGIN
    ids_len = 1
    full = max_input_words - 1
    for i in input_string:
        indices = np.where(vocab == i)[0]
        if len(indices) > 0:
            input_ids[:, ids_len] = indices[0]
            ids_len += 1
            if ids_len == full:
                break
        else:
            for j in list(i):
                indices = np.where(vocab == j)[0]
                if len(indices) > 0:
                    input_ids[:, ids_len] = indices[0]
                else:
                    input_ids[:, ids_len] = TOKEN_UNKNOWN
                ids_len += 1
                if ids_len == full:
                    break
    input_ids[:, ids_len] = TOKEN_END
    ids_len += 1

    # Process the punc_ids
    for i, ch in enumerate(input_string):
        if ch in ['。', '，', '？', '！']:
            punc_ids[:, i] = 1
        else:
            punc_ids[:, i] = 0

    if is_dynamic:
        input_ids = input_ids[:, :ids_len]
        punc_ids = punc_ids[:, :ids_len]

    return input_ids, punc_ids, np.array([ids_len], dtype=np.int64), input_string


print("\nRun the Bert Speaker Diarization by ONNXRuntime.")
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A0 = ort_session_A.get_inputs()[0].name
in_name_A1 = ort_session_A.get_inputs()[1].name
in_name_A2 = ort_session_A.get_inputs()[2].name
out_name_A0 = ort_session_A.get_outputs()[0].name
out_name_A1 = ort_session_A.get_outputs()[1].name
if isinstance(shape_value_in, str):
    max_input_words = 1024                  # Default value, you can adjust it.
    is_dynamic = True
else:
    max_input_words = shape_value_in
    is_dynamic = False

# Run the Bert_Speaker_Diarization
input_ids, punc_ids, ids_len, input_string = tokenizer(sentence, max_input_words, is_dynamic)
start_time = time.time()
turn_indices, turn_indices_len = ort_session_A.run([out_name_A0, out_name_A1], {in_name_A0: input_ids, in_name_A1: punc_ids, in_name_A2: ids_len})
end_time = time.time()
if turn_indices_len != 0:
    shift = 1
    for i in turn_indices:
        input_string.insert(i + shift, "\n-> ")
        shift += 1
    input_string = "".join(input_string)
    print(f"\nSpeaker Turn Position: {turn_indices}\n\nSplit Sentence:\n-> {input_string}\n\nTime Cost: {end_time - start_time:.3f} seconds")
else:
    print("\nNo Speaker Turn detected.")
  
