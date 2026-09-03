DATASET_PARAS = '../scripts/probe_paras.yml'
PIXEL_M4 = "..."
DE_PIXEL = "..."
PERSIAN_PIXEL = "..."
KO_PIXEL = "..."
ES_PIXEL = "..."
TR_PIXEL = "..."
LANG_CONFIG = {

    "es":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'DeepESP/gpt2-spanish',
        "bert":"dccuchile/bert-base-spanish-wwm-cased",
        "force_special_tokens_for_gpt":{"bos_token":"<|endoftext|>", "eos_token":"<|endoftext|>"},
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":ES_PIXEL
    },
    "es_ch":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'DeepESP/gpt2-spanish',
        "bert":"dccuchile/bert-base-spanish-wwm-cased",
        "force_special_tokens_for_gpt":{"bos_token":"<|endoftext|>", "eos_token":"<|endoftext|>"},
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":ES_PIXEL
    },
    "ge_zu":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'dbmdz/german-gpt2',
        "bert":"google-bert/bert-base-german-cased",
        "force_special_tokens_for_gpt":{"bos_token":"<s>", "eos_token":"</s>"},
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":DE_PIXEL
    },

    "ge_po":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'dbmdz/german-gpt2',
        "bert":"google-bert/bert-base-german-cased",
        "force_special_tokens_for_gpt":{"bos_token":"<s>", "eos_token":"</s>"},
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":DE_PIXEL
    },

    "ko":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'skt/kogpt2-base-v2',
        "bert":"kykim/bert-kor-base",
        "force_special_tokens_for_gpt":{"bos_token":"<s>", "eos_token":"</s>"},
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":KO_PIXEL
    },
    "simp":{
        "probe_group_by":['stimuli_id'],
        "gpt":'uer/gpt2-chinese-cluecorpussmall',
        "bert":"google-bert/bert-base-chinese",
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "force_special_tokens_for_gpt":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":False,
        "use_force_segmentation":True,
        "ia_post_process_func":None,
        "pixel":PIXEL_M4,
    },
    "trad":{
        "probe_group_by":['stimuli_id'],
        "gpt":'ckiplab/gpt2-base-chinese',
        "bert":"ckiplab/bert-base-chinese",
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "force_special_tokens_for_gpt":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":False,
        "use_force_segmentation":True,
        "ia_post_process_func":None,
        "pixel":PIXEL_M4,
    },
    "tr":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'ytu-ce-cosmos/turkish-gpt2',
        "bert":"dbmdz/bert-base-turkish-cased",
        "force_special_tokens_for_gpt":None,
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":TR_PIXEL,
    },
    "ge_uk":{
        "probe_group_by":['stimuli_id','sent_idx'],
        "gpt":'gpt2',
        "bert":"google-bert/bert-base-cased",
        "force_special_tokens_for_gpt":None,
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},

        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":PIXEL_M4,
    },
    "persian2":{
        "probe_group_by":['stimuli_id'],
        "gpt":'bolbolzaban/gpt2-persian',
        "bert":"HooshvareLab/bert-base-parsbert-uncased",
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "force_special_tokens_for_gpt":None,
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":PERSIAN_PIXEL,
    },
    "en":{
        "probe_group_by":['stimuli_id'],
        "gpt":'gpt2',
        "bert":"google-bert/bert-base-cased",
        "force_special_tokens_for_bert":{"bos_token":"[CLS]", "eos_token":"[SEP]"},
        "force_special_tokens_for_gpt":None,
        "is_spaced":True,
        "use_force_segmentation":False,
        "ia_post_process_func":None,
        "pixel":PIXEL_M4,
    }
}
