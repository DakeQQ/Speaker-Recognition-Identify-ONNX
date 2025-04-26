import re
import time
import torch
import numpy as np
import onnxruntime
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


model_path = r"/home/DakeQQ/Downloads/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese"      # Path to the entire downloaded Bert model project.
onnx_model_A = r"/home/DakeQQ/Downloads/Bert_Speaker_Diarization_ONNX/Bert_Speaker_Diarization.onnx"                 # The exported onnx model save path.
vocab_path = f'{model_path}/vocab.txt'                                                                               # Set the path where the Bert model vocab.txt stored.
sentence = "你是如何看待这个问题的呢？这个问题挺好解决的，我们只需要增加停车位就行了。嗯嗯，好，那我们业主就放心了。"                 # The sentence for test.


DYNAMIC_AXES = False          # Whether both are set to True or False, they must still be less than MAX_INPUT_WORDS.
MAX_INPUT_WORDS = 128         # The maximum input words for a sentence.
TOKEN_UNKNOWN = 100           # The model parameter, do not edit it.
TOKEN_BEGIN = 101             # The model parameter, do not edit it.
TOKEN_END = 102               # The model parameter, do not edit it.


# Read the vocab.
with open(vocab_path, 'r', encoding='utf-8') as file:
    vocab = file.readlines()
vocab = np.array([line.strip() for line in vocab], dtype=np.str_)


class BERT(torch.nn.Module):
    def __init__(self, bert_model, max_seq_len, token_end):
        super(BERT, self).__init__()
        self.bert_model = bert_model.bert
        self.head = bert_model.head
        self.token_end = token_end
        attention_head_size_factor = float(self.bert_model.encoder.layer._modules["0"].attention.self.attention_head_size ** -0.25)
        for layer in self.bert_model.encoder.layer:
            layer.attention.self.query.weight.data *= attention_head_size_factor
            layer.attention.self.query.bias.data *= attention_head_size_factor
            layer.attention.self.key.weight.data *= attention_head_size_factor
            layer.attention.self.key.bias.data *= attention_head_size_factor
        self.bert_model.embeddings.token_type_embeddings.weight.data = self.bert_model.embeddings.token_type_embeddings.weight.data[[0], :max_seq_len].unsqueeze(-1)
        self.bert_model.embeddings.position_embeddings.weight.data = self.bert_model.embeddings.position_embeddings.weight.data[:max_seq_len, :].unsqueeze(0)

    def forward(self, input_ids: torch.IntTensor, punc_ids: torch.IntTensor):
        if DYNAMIC_AXES:
            ids_len = input_ids.shape[-1].unsqueeze(0)
        else:
            ids_len = torch.where(input_ids == self.token_end)[-1] + 1
            input_ids = input_ids[:, :ids_len]
            punc_ids = punc_ids[:, :ids_len]
        hidden_states = self.bert_model.embeddings.LayerNorm(self.bert_model.embeddings.word_embeddings(input_ids) + self.bert_model.embeddings.token_type_embeddings.weight.data[:, :ids_len] + self.bert_model.embeddings.position_embeddings.weight.data[:, :ids_len])
        for layer in self.bert_model.encoder.layer:
            query_layer = layer.attention.self.query(hidden_states).view(-1, layer.attention.self.num_attention_heads, layer.attention.self.attention_head_size).transpose(0, 1)
            key_layer = layer.attention.self.key(hidden_states).view(-1, layer.attention.self.num_attention_heads, layer.attention.self.attention_head_size).permute(1, 2, 0)
            value_layer = layer.attention.self.value(hidden_states).view(-1, layer.attention.self.num_attention_heads, layer.attention.self.attention_head_size).transpose(0, 1)
            attn_out = torch.matmul(torch.nn.functional.softmax(torch.matmul(query_layer, key_layer), dim=-1), value_layer).transpose(0, 1).contiguous().view(1, -1, layer.attention.self.all_head_size)
            attn_out = layer.attention.output.LayerNorm(layer.attention.output.dense(attn_out) + hidden_states)
            hidden_states = layer.output.LayerNorm(layer.output.dense(layer.intermediate.intermediate_act_fn(layer.intermediate.dense(attn_out))) + attn_out)
        hidden_states = self.head.classifier(hidden_states)
        max_logit_ids = torch.argmax(hidden_states, dim=-1).int() * punc_ids
        turn_indices = torch.nonzero(max_logit_ids, as_tuple=True)[-1].int()
        return turn_indices, turn_indices.shape[-1].int()


print("\nExport Start...")
with torch.inference_mode():
    model = pipeline(
        task=Tasks.speaker_diarization_semantic_speaker_turn_detection,
        model=model_path,
        device='cpu'
    ).model.eval().float()
    input_ids = torch.zeros((1, MAX_INPUT_WORDS), dtype=torch.int32)
    punc_ids = torch.zeros((1, MAX_INPUT_WORDS), dtype=torch.int32)
    if not DYNAMIC_AXES:
        input_ids[:, 0] = TOKEN_END
    model = BERT(model, MAX_INPUT_WORDS, TOKEN_END)
    torch.onnx.export(model,
                      (input_ids, punc_ids),
                      onnx_model_A,
                      input_names=['text_ids', 'punc_ids'],
                      output_names=['turn_indices', 'turn_indices_len'],
                      dynamic_axes={
                          'text_ids': {1: 'ids_len'},
                          'punc_ids': {1: 'ids_len'},
                          'turn_indices': {-1: 'turn_indices_len'}
                      } if DYNAMIC_AXES else {'turn_indices': {-1: 'turn_indices_len'}},
                      do_constant_folding=True,
                      opset_version=17)
del model
del input_ids
del punc_ids
print("\nExport Done!")


# For Bert Model
def tokenizer(input_string, max_input_words, is_dynamic):
    input_ids = np.zeros((1, max_input_words), dtype=np.int32)
    input_string = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|[^\w\s]', input_string.lower())
    input_ids[0] = TOKEN_BEGIN
    full = max_input_words - 1
    ids_len = 1
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

    punc_ids = np.zeros((1, max_input_words), dtype=np.int32)
    for i, ch in enumerate(input_string):
        if ch in ['。', '，', '？', '！']:
            punc_ids[:, i] = 1
        else:
            punc_ids[:, i] = 0

    if is_dynamic:
        input_ids = input_ids[:, :ids_len + 1]
        punc_ids = punc_ids[:, :ids_len + 1]

    return input_ids, punc_ids, input_string


print("\nRun the Bert Speaker Diarization by ONNXRuntime.")
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

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A0 = ort_session_A.get_inputs()[0].name
in_name_A1 = ort_session_A.get_inputs()[1].name
out_name_A0 = ort_session_A.get_outputs()[0].name
out_name_A1 = ort_session_A.get_outputs()[1].name
if isinstance(shape_value_in, str):
    max_input_words = 1024                  # Default value, you can adjust it.
    is_dynamic = True
else:
    max_input_words = shape_value_in
    is_dynamic = False

# Run the Bert_Speaker_Diarization
input_ids, punc_ids, input_string = tokenizer(sentence, max_input_words, is_dynamic)
start_time = time.time()
turn_indices, turn_indices_len = ort_session_A.run([out_name_A0, out_name_A1], {in_name_A0: input_ids, in_name_A1: punc_ids})
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
