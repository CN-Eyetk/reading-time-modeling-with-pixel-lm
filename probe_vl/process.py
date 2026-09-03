try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except:
    print("Qwen2_5_VLForConditionalGeneration not found")
    from transformers import AutoProcessor
from typing import List
try:
    from qwen_vl_utils import process_vision_info
except:
    print("qwen_vl_utils not found")
import torch
import re
def get_condition_length(processor: AutoProcessor, messages: List[dict], condition = None, add_generation_prompt = True):

    image_inputs, video_inputs = process_vision_info(messages)
    if condition is None:
        condition = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    inputs = processor(
        text=[condition],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return len(inputs.input_ids[0])


def get_depth_ids(processor: AutoProcessor, input_ids, sep = " "):
    #Given is sequence of tokens, we group the tokens, naively, according to how many spaces are there before
    #input_ids [B, L]
    seqs = [processor.batch_decode(input_ids[:,:i+1]) for i in range(input_ids.size(-1))]
    seqs = [list(row) for row in zip(*seqs)]
    depth_ids = [[seq.count(sep) for seq in sample] for sample in seqs]
    depth_ids = torch.LongTensor(depth_ids)
    return depth_ids


def get_depth_ids_by_ia_labels(processor: AutoProcessor, input_ids: list[int], ia_labels: list[str], sep = ' ', decoding_clean_func = None):
    
    def is_cut_off(ia_labels, seq):
        for i in range(len(ia_labels)):
            cur_ias  = sep.join(ia_label for ia_label in ia_labels[:i+1])
            if "[UNK]" in seq:
                if decoding_clean_func(processor.decode(processor.encode(cur_ias, add_special_tokens=False))) == seq:
                    return 1
            else:            
                if cur_ias == seq:
                    return 1
        return 0
        
    #Given is sequence of tokens, we group the tokens, naively, according to how many spaces are there before
    #input_ids [B, L]
    seqs = [processor.decode(input_ids[:i+1]) for i in range(len(input_ids))]
    if decoding_clean_func is not None:
        seqs = [decoding_clean_func(seq) for seq in seqs]

    
    cuf_offs = [0] + [is_cut_off(ia_labels, seq) for seq in seqs]
    cuf_offs = torch.LongTensor(cuf_offs)
    depth_ids = cuf_offs.cumsum(-1)[:-1]
    return depth_ids

#def force_segmentation(tokenizer, words):
#    for i in range(len(words)):
def force_segmentation_old(tokenizer, words, add_special_tokens=False):
    """
    优先用滚雪球式前缀差分；
    如果发生 tokenizer 回溯重切分，则退化为整句 + offset 对齐。
    """
    prev_ids = []
    all_input_ids = []
    per_word_input_ids = []
    prefix_text = ""

    ok = True
    for w in words:
        prefix_text += w
        cur_ids = tokenizer.encode(prefix_text, add_special_tokens=add_special_tokens)

        if cur_ids[:len(prev_ids)] == prev_ids:
            new_ids = cur_ids[len(prev_ids):]
            per_word_input_ids.append(new_ids)
            all_input_ids.extend(new_ids)
            prev_ids = cur_ids
        else:
            ok = False
            break

    if ok:
        return {
            "input_ids": all_input_ids,
            "per_word_input_ids": per_word_input_ids,
            "method": "prefix_incremental",
        }

    # fallback: 整句 offsets 对齐
    text = "".join(words)
    enc = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    spans = []
    pos = 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w)

    per_word_input_ids = [[] for _ in words]
    for tid, (s, e) in zip(input_ids, offsets):
        for i, (ws, we) in enumerate(spans):
            if not (e <= ws or s >= we):
                per_word_input_ids[i].append(tid)

    return {
        "input_ids": input_ids,
        "per_word_input_ids": per_word_input_ids,
        "offset_mapping": offsets,
        "method": "offset_fallback",
    }
    
def force_segmentation(tokenizer, words, add_special_tokens=False):
    """
    每个词单独 tokenize，结果直接拼接。
    完全避免整句上下文导致的跨词合并问题。
    """
    per_word_input_ids = []
    all_input_ids = []

    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        per_word_input_ids.append(ids)
        all_input_ids.extend(ids)

    if add_special_tokens:
        # 手动加首尾特殊 token
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        if bos is not None:
            all_input_ids = [bos] + all_input_ids
        if eos is not None:
            all_input_ids = all_input_ids + [eos]

    return {
        "input_ids": all_input_ids,
        "per_word_input_ids": per_word_input_ids,
        "method": "per_word_independent",
    }

def segment_chinese(text):
    """
    Splits Chinese text into sentences, correctly handling:
    - 。！？as sentence ends
    - ……。(ellipsis + period as ONE sentence end)
    - Closing punctuation after quotes: 」。『』
    - Consecutive punctuation: ！！？？
    """
    # Step 1: Normalize — treat ellipsis (...... or ……) as a unit
    # Don't split on lone … unless followed by sentence-end
    pattern = re.compile(
        r'('
        r'[^。！？]*'          # any non-terminal chars
        r'(?:……|…)?'           # optional ellipsis
        r'[。！？]+'            # one or more terminal punctuation
        r'(?:[」』"\'）])*'    # optional closing brackets/quotes
        r')'
    )
    
    sentences = pattern.findall(text)
    
    # Capture any trailing text without terminal punctuation
    matched_len = sum(len(s) for s in sentences)
    remainder = text[matched_len:].strip()
    if remainder:
        sentences.append(remainder)
    
    return [s.strip() for s in sentences if s.strip()]

def generate_sentence_id(ias, lang = "zh"):
    if lang == "zh":
        sentence_ids = []

        for i,ia in enumerate(ias):
            cur_seq = ''.join(x for x in ias[:i+1])
            n_sent = len(segment_chinese(cur_seq))
            sentence_ids.append(n_sent)
    else:
        assert 1 == 2
    return sentence_ids



def get_attention_head_map(attentions, layer_indices):
    attention_by_layers = []
    
    for l in layer_indices:
        attention_cur_layer = attentions[l]#[b,n_head, s,t]
        attention_cur_layer_all_head = attention_cur_layer.mean(dim = 1)
        attention_cur_layer_all_target = attention_cur_layer_all_head.mean(dim = 1)#[b,t]
        attention_by_layers.append(attention_cur_layer_all_target)
    return torch.stack(attention_by_layers, dim = 1)
