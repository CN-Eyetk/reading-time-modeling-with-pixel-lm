
from dataclasses import dataclass, field
import yaml, pprint
import torch
import sys
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
sys.path.append("../scripts")

from utils import np_to_pil, reshape_position_ids_for_rtl, present_rtl
try:
    from qwen_vl_utils import process_vision_info
except:
    print("qwen_vl_utils not available, please ensure this env is for pixellm")
from torch.distributions import Categorical
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
import re
import os
from process import get_depth_ids, get_depth_ids_by_ia_labels, force_segmentation, generate_sentence_id
from get_ia_measures import get_durations_and_counts, get_normal_measures
import matplotlib.pyplot as plt
from process import get_attention_head_map
#from fast_sentence_segment import segment_text
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

handler = logging.FileHandler("bug.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)

def split_sentence(text, max_words=128):
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    mid_word_length = len(words) // 2
    first_part_words = words[:mid_word_length]
    remaining_words = words[mid_word_length:]

    remaining_text = " ".join(remaining_words)

    period_index = remaining_text.find(".")

    if period_index == -1:
        return [
            " ".join(first_part_words),
            remaining_text
        ]
    
    split_point = period_index + 1
    
    second_part_1 = remaining_text[:split_point].strip()
    second_part_2 = remaining_text[split_point:].strip()
    
    return [
        " ".join(first_part_words) + " " + second_part_1,
        second_part_2
    ]
def process_incoming_attention(all_incoming_attn):
    """
    Args:
        all_incoming_attn:
            shape = (batch, masked_pos, layers, heads, query_pos)

    Returns:
        attn_scores:
            shape = (batch, masked_pos)
    """

    # Take final 2 layers
    # shape: (batch, masked_pos, 2, heads, query_pos)
    last_two = all_incoming_attn[:, :, -2:, :, :]

    # Mean over:
    #   layers
    #   heads
    #   query positions
    #
    # Result:
    #   (batch, masked_pos)
    attn_scores = last_two.mean(dim=(2, 3, 4))

    return attn_scores
@dataclass
class AutoProberConfig:
    path_to_model: str
    path_to_processor: str
    path_to_plot_paras: str = None
    dataset_paras: str = None
    dataset_name: str = None
    is_vl: bool = True
    is_pixel_lm: bool = False
    pixel_lm_right_blur: bool = False
    is_spaced: bool = True
    force_special_tokens: dict = None
    use_force_segmentation: bool = False
    use_official_aoi_positions: bool = False
    crop: bool = False
    feat_idx_for_sent_idx: int = None #in the field 'ias', provide this field, it extract sent idx from this field
    probe_group_by: list[str] = field(default_factory=lambda: ['stimuli_id']) #['stimuli_id']
    local_window: tuple = None
    local_window_size: tuple = None
    cut_off: float = 0.0
    is_pixel_lm_m4: bool = False
    path_to_precomputed: str = None
    left_truncate_to_picture: bool = False
    pixel_whole_ia_masking: bool = False
    ia_post_process_func: str = None
    pixel_use_arabic_renderer: bool = False
    is_hebrew: bool = False
    is_pixel_denoiser: bool = False
    is_bidir: bool = False
    pixel_preview_window: int = 0
    pixel_remove_white: bool = False
    pixel_blur_kernel_size: int = 5
    pixel_lm_right_cut: bool = False
    
#True, when 
    #separate_stimuli_by: list = []


def gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):

    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    
    kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # [k, k]
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    # [channels, 1, k, k]
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def ssim(pred, target, kernel_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):

    B, C, H, W = pred.shape
    
    kernel = gaussian_kernel(kernel_size, sigma, C).to(pred.device)
    padding = kernel_size // 2
    
    mu_pred   = F.conv2d(pred,   kernel, padding=padding, groups=C)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=C)
    
    mu_pred_sq   = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq   = F.conv2d(pred**2,        kernel, padding=padding, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target**2,      kernel, padding=padding, groups=C) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=padding, groups=C) - mu_pred_target
    
    numerator   = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / denominator  # [B, C, H, W]
    
    return ssim_map.mean(dim=(1, 2, 3))  # [B]


class SSIMLoss(torch.nn.Module):

    def __init__(self, kernel_size=11, sigma=1.5, reduction='none'):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.reduction = reduction  # 'mean' | 'none'
    
    def forward(self, pred, target):
        """
        pred, target: [B, C, H, W]
        """
        ssim_score = ssim(
            pred, target,
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )  # [B]
        
        loss = 1 - ssim_score  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss  # [B]
    
def patch_entropy_gradient(patches):


    gray = patches.mean(dim=1)
    

    dx = gray[:, :, 1:] - gray[:, :, :-1]   # [B, H, W-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]   # [B, H-1, W]
    

    h = min(dx.shape[1], dy.shape[1])  # H-1
    w = min(dx.shape[2], dy.shape[2])  # W-1
    
    grad_magnitude = dx[:, :h, :w] ** 2 + dy[:, :h, :w] ** 2  # [B, H-1, W-1]
    

    return grad_magnitude.mean(dim=(1, 2))  # [B]
def collect_images_from_messages(messages):
    """
    Extract PIL images from a multimodal chat message list.
    Keeps the same order as they appear in the conversation.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    # support either {"image": pil_image} or {"url": "..."}
                    if "image" in item:
                        images.append(item["image"])
                    elif "url" in item:
                        images.append(item["url"])
    return images

def clip(x: torch.Tensor):
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x

def collate_to_batch(inputs_list, device):
    result = {}
    for k in inputs_list[0].keys():
        if isinstance(inputs_list[0][k], torch.Tensor):
            tensors = [inp[k] for inp in inputs_list]
            if inputs_list[0][k].dim() == 0:    # scalar → stack into (B,)
                result[k] = torch.stack(tensors).to(device)
            else:                                # (1,...) → cat into (B,...)
                result[k] = torch.cat(tensors, dim=0).to(device)
        else:
            result[k] = inputs_list[0][k]
    return result


def discretized_logistic_likelihood(x, mean, log_scale):
    """
    Computes the log-likelihood of a discretized logistic distribution.
    x: input pixels, usually scaled to [-1, 1]
    mean: predicted mean
    log_scale: predicted log scale (to ensure positivity)
    """
    scale = torch.exp(log_scale)
    
    # Standardize the inputs
    centered_x = x - mean
    inv_stdv = 1.0 / scale
    
    # Calculate CDF for upper and lower bounds of the bin
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    
    minus_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(minus_in)
    
    # Log-likelihood for the middle bins
    log_probs = torch.log(cdf_plus - cdf_min + 1e-7)
    
    # Edge cases: 
    # For x = -1 (or 0 in 0-255 range), use the CDF from -infinity to x+0.5
    log_probs_low = F.logsigmoid(plus_in)
    
    # For x = 1 (or 255 in 0-255 range), use the CDF from x-0.5 to +infinity
    log_probs_high = torch.log(1.0 - cdf_min + 1e-7)
    
    # Combine based on pixel value
    res = torch.where(x < -0.999, log_probs_low, 
                     torch.where(x > 0.999, log_probs_high, log_probs))
    
    return res
class AutoProber:
    def __init__(self, config: AutoProberConfig):
        self.config = config
        self.is_spaced = self.config.is_spaced
        self.is_pixel_lm = config.is_pixel_lm
        self.is_bidir = config.is_bidir
        if config.path_to_model is None:
            pass
        elif config.is_vl:
            pass
        elif config.is_pixel_lm:
            from pixelm4_engine import PixelLM
            if config.is_pixel_lm_m4:
                self.engine = PixelLM(config.path_to_model, fallback_fonts_dir = "fallback_fonts_dir", is_m4 = True, 
                                    use_arabic_renderer=config.pixel_use_arabic_renderer,
                                    is_hebrew = config.is_hebrew,
                                    is_denoiser = config.is_pixel_denoiser
                                    )
            else:
                self.engine = PixelLM(config.path_to_model, is_m4 = False, use_arabic_renderer=config.pixel_use_arabic_renderer)
            self.eos_token = None
            self.bos_token = None
            
            
        else:
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
            from process import get_depth_ids, get_depth_ids_by_ia_labels, force_segmentation
            if self.is_bidir:
                model = AutoModelForMaskedLM.from_pretrained(config.path_to_model,  use_auth_token=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(config.path_to_model,  use_auth_token=True)
            processor = AutoTokenizer.from_pretrained(config.path_to_processor, use_auth_token=True)
            self.model = model
            self.processor = processor
            if config.force_special_tokens is not None:
                self.eos_token = config.force_special_tokens['eos_token']
                self.bos_token = config.force_special_tokens['bos_token']
            else:
                if processor.eos_token_id is None:
                    self.eos_token = processor.decode(processor.sep_token_id)
                else:
                    self.eos_token = processor.decode(processor.eos_token_id)
                if processor.bos_token_id is None:
                    self.bos_token = processor.decode(processor.eos_token_id)
                else:
                    self.bos_token = processor.decode(processor.bos_token_id)
            self.tokenizer = processor

        if config.dataset_paras is not None:
            print("config.dataset_paras=",config.dataset_paras)
            print(yaml.safe_load(open(config.dataset_paras,"r+")))
            self.probe_paras = yaml.safe_load(open(config.dataset_paras,"r+"))[config.dataset_name]
            print("self.probe_paras",self.probe_paras)
            
            if config.is_vl:
                pass
            self.path_to_dataset = self.probe_paras['path']
            self.colname_of_fixation = self.probe_paras['colname_of_fixation']
            self.ia_idx_start_from = self.probe_paras['ia_idx_start_from']
            self.colname_of_ia_idx = self.probe_paras['colname_of_ia_idx']
            self.colname_of_ia_label = self.probe_paras['colname_of_ia_label']
            self.colname_of_pupil = self.probe_paras['colname_of_pupil']
            self.colname_of_x = self.probe_paras['colname_of_x']
            self.colname_of_ia_left = self.probe_paras['colname_of_ia_left']
            self.colname_of_sentence_id = self.probe_paras['colname_of_sentence_id']
            self.colname_of_subj_id = self.probe_paras['colname_of_subj_id']
            self.colname_of_landing_position = self.probe_paras['colname_of_landing_position']
            self.colname_of_experiment_paradigms = self.probe_paras['colname_of_experiment_paradigms']
            self.colname_of_stimuli_features = self.probe_paras['colname_of_stimuli_features']
            if self.config.path_to_precomputed is not None:
                self.path_to_precomputed = self.config.path_to_precomputed
            else:
                print("fall back to the default path")
                self.path_to_precomputed = self.probe_paras['path_to_precomputed']
            self.path_to_plot_paras = self.probe_paras['plot_paras']
            self.data_is_rm = self.probe_paras['data_is_rm']
            self.is_rtl = self.probe_paras['is_rtl']

            self.reading_measure_columns = []
            self.path_to_output = os.path.join(self.path_to_precomputed, self.config.dataset_name)
            self.filter_by = self.probe_paras['filter_by']
            self.generate_sent_idx = self.probe_paras['generate_sent_idx']
            self.colname_of_fix_id = self.probe_paras['colname_of_fix_id']
            #self.seperate_stimuli_by = self.probe_paras['seperate_stimuli_by']
            print("output path",self.path_to_output)
            os.makedirs(self.path_to_output, exist_ok=True)
    def load_precomputed(self, overwrite = False):
        
        path_to_precomputed_measures = os.path.join(self.path_to_precomputed, self.config.dataset_name, 'reading_measures.csv')
        if os.path.exists(path_to_precomputed_measures) and not overwrite:
            self.df_reading_measures = pd.read_csv(path_to_precomputed_measures, sep = '\t', keep_default_na=False,
            na_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                    '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NaN', 'None', 'n/a',
                    'nan', ''])
        else:
            self.load_reading_measures()
            self.df_reading_measures.to_csv(path_to_precomputed_measures, sep = '\t')
        
        path_to_precomputed_measures_norm = os.path.join(self.path_to_precomputed, self.config.dataset_name, 'reading_measures_norms.csv')
        if os.path.exists(path_to_precomputed_measures_norm) and not overwrite:
            self.df_reading_measures_normalized = pd.read_csv(path_to_precomputed_measures_norm, sep = '\t', keep_default_na=False,
            na_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                    '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NaN', 'None', 'n/a',
                    'nan', ''])
        else:
            self.normalize_between_subjects()
            self.df_reading_measures.to_csv(path_to_precomputed_measures, sep = '\t')
            self.df_reading_measures_normalized.to_csv(path_to_precomputed_measures_norm, sep = '\t')
        

    def processing_logits(self, input_ids, logits, condition_length = 0):

        shift_input_ids = input_ids[:,condition_length:]
        truncate_logits = logits[:,condition_length-1:-1]
        truncate_probs = truncate_logits.softmax(-1)
        selected_logits = torch.gather(
            truncate_logits,
            dim=2,
            index=shift_input_ids.unsqueeze(-1)
        )
        selected_probs = torch.gather(
            truncate_probs,
            dim=2,
            index=shift_input_ids.unsqueeze(-1)
        )
        
        surprisal = torch.log2(selected_probs) * -1
        surprisal = surprisal.squeeze(-1)
        entropy = Categorical(logits=truncate_logits).entropy()
        entropy = entropy.tolist()
        selected_logits = selected_logits.squeeze(-1)
        selected_logits = selected_logits.tolist()
        surprisal = surprisal.tolist()
        return surprisal, entropy, selected_logits
    def gather_ia_feat_from_token_data(self, token_data):
        ia_surprisal_sum = token_data.groupby('word_id')['surprisal'].sum().to_list()
        ia_surprisal_mean = token_data.groupby('word_id')['surprisal'].mean().to_list()
        ia_entropy_sum = token_data.groupby('word_id')['entropy'].sum().to_list()
        ia_entropy_mean = token_data.groupby('word_id')['entropy'].mean().to_list()
        ia_logits_sum = token_data.groupby('word_id')['logits'].sum().to_list()
        ia_logits_mean = token_data.groupby('word_id')['logits'].mean().to_list()
        ia_attention_sum = token_data.groupby('word_id')['attention'].sum().to_list()
        ia_attention_mean = token_data.groupby('word_id')['attention'].mean().to_list()
        ia_label = token_data.groupby('word_id')['token'].sum().to_list()
        idx = token_data.groupby('word_id')['word_id'].first().tolist()
        dist_to_sent_length = [len(idx) - i for i in range(len(idx))]
        ia_attention_sum_uni_norm = [ia_attention_sum[i]/dist_to_sent_length[i] for i in range(len(dist_to_sent_length))]
        ia_data = {
            "ia_surprisal_sum":ia_surprisal_sum,
            "ia_surprisal_mean":ia_surprisal_mean,
            "ia_entropy_sum":ia_entropy_sum,
            "ia_entropy_mean":ia_entropy_mean,
            "ia_logits_sum":ia_logits_sum,
            "ia_logits_mean":ia_logits_mean,
            "ia_attention_sum_uni":ia_attention_sum,
            "ia_attention_mean_uni":ia_attention_mean,
            "ia_label":ia_label,
            "idx":idx,
            "ia_attention_sum_uni_norm":ia_attention_sum_uni_norm
        }
        ia_data = pd.DataFrame(ia_data)
        return ia_data

    def probe_pixel_lm_batch(self, texts, sep = " ", batch_size = 16):
        ia_datas = []
        ia_attention_weights_by_sample = []
        ia_attention_weights_uni_by_sample = []
        for sample_idx, text in enumerate(texts):
            if self.is_spaced:
                base_inputs, pre_encoding = self.engine.prepare_inputs(words = text,
                        manual_mask = []
                        )
                if not self.config.use_force_segmentation:
                    patch_lengths, patch_idx = self.engine.get_patch_index_by_words(words = text.split(sep))
                else:
                    patch_lengths, patch_idx = self.engine.get_patch_index_by_words_without_space(words = text.split(sep), sep = sep)
            else:
                base_inputs, pre_encoding = self.engine.prepare_inputs(words = "".join(x for x in text.split(sep)),
                        manual_mask = []
                        )
                patch_lengths, patch_idx = self.engine.get_patch_index_by_words_without_space(words = text.split(sep), sep = "")
            loss_fn = nn.MSELoss(reduction='none')
            ssim_fn = SSIMLoss(kernel_size=3,sigma=0.1)
            mae_loss_fn = nn.L1Loss(reduction='none')
            ref = self.engine.model.patchify(base_inputs['pixel_values'])

            ias = text.split(sep)
            ia_attention_weights = [0 for _ in range(len(ias))]
            ia_attention_weights_uni = [0 for _ in range(len(ias))]
            batches = [[]]
            device = next(self.engine.model.parameters()).device
            for t in range(len(ias)):
                #print(f"t={t}")
                for patch_id in patch_idx[t]:
                    if 1==2:
                        if self.config.pixel_whole_ia_masking:
                            inputs, pre_encoding = self.engine.prepare_inputs(words = text if self.is_spaced else "".join(x for x in text.split(sep)),
                                                    manual_mask = patch_idx[t],
                                                    )
                        else:
                            inputs, pre_encoding = self.engine.prepare_inputs(words = text if self.is_spaced else "".join(x for x in text.split(sep)),
                                                    manual_mask = [patch_id],
                                                    )
                    else:
                        if self.config.pixel_whole_ia_masking:
                            manual_mask = patch_idx[t]
                        else:
                            manual_mask = [patch_id]
                        
                        patch_mask = self.engine.generate_manual_mask(manual_mask)

                        inputs = {
                            k: v.clone() if torch.is_tensor(v) else v
                            for k, v in base_inputs.items()
                        }
                        inputs['patch_mask'] = patch_mask
                        #for k,v in inputs.items():
                        #    print(f"k={k}")
                        #    print(f"v={v.shape}")
                            
                    if self.config.pixel_lm_right_blur:
                        pixel_values = inputs['pixel_values']
                        blur = GaussianBlur(kernel_size=self.config.pixel_blur_kernel_size, sigma=2.0)
                        blurred = blur(pixel_values)
                        if not self.is_rtl:
                            blur_start_from = min((patch_id + self.config.pixel_preview_window) * 16, inputs['pixel_values'].size(-1))
                            inputs['pixel_values'][:,:,:,blur_start_from:] = blurred[:,:,:,blur_start_from:]
                        else:
                            blur_start_from = max((patch_id-self.config.pixel_preview_window) * 16, 0)
                            #print("blur_start_from",blur_start_from)
                            inputs['pixel_values'][:,:,:,:blur_start_from] = blurred[:,:,:,:blur_start_from]
                    if self.config.pixel_lm_right_cut:
                        if not self.is_rtl:
                            cut_from = min(patch_id + self.config.pixel_preview_window + 1, inputs['attention_mask'].size(-1))
                            inputs['attention_mask'][:, cut_from:] = 0
                        else:
                            cut_from = max(patch_id + self.config.pixel_preview_window - 1, 0)
                            inputs['attention_mask'][:, :cut_from] = 0
                    
                    #print(f"patch_id={patch_id}, t={t}")
                    inputs["t"] = torch.tensor(t, dtype=torch.long)           # shape () scalar
                    inputs["patch_id"] = torch.tensor(patch_id, dtype=torch.long)  # shape () or (N,)


                    if len(batches[-1]) < batch_size:
                        batches[-1].append(inputs)
                    else:
                        batches.append([inputs])
            #for batch in batches:
            #    for sample in batch:
            #        print("sample",sample)
            batch_inputs = [collate_to_batch(batch, self.engine.model.device) for batch in batches]
            
            
            #for batch in batch_inputs:
            #    print(batch['t'])

            for s, cur_inputs in enumerate(batch_inputs):

                filtered_inputs = {k:v for k,v in cur_inputs.items() if k not in ["t", "patch_id"]}
                #for k,v in cur_inputs.items():
                #    print(f"Shape of {k} = {v.shape}")
                #print("input pixel values",filtered_inputs['pixel_values'].size())
                #print("input mask",filtered_inputs['patch_mask'].sum(-1))
                predictions = self.engine.get_predictions(filtered_inputs)
                #print("predictions",predictions['logits'].size())
                #print("rec",predictions['reconstruction'].size())
                #print("predictions.logits",predictions['logits'].shape)
                reconstruction = predictions['reconstruction']
                #for b in range(len(reconstruction)):
                #    plt.imshow(reconstruction[b].sum(0)[:,:600])
                #    cur_t = cur_inputs["t"][b]
               #     plt.savefig(f"examples/s_{s}_{b}_rec_t={cur_t}.png", bbox_inches='tight', pad_inches=0)
                attention_weights = predictions['head_map']
                #双向weight
                for t in range(len(ias)):
                    for patch_id in patch_idx[t]:
                        for b in range(len(attention_weights)):
                            ia_attention_weights[t] += attention_weights[b,patch_id].item()
                            #cur_masked_t = cur_inputs["t"][b]
                            
                            #for i_t in range(cur_masked_t+1):
                            #    ia_attention_weights_uni[i_t] += attention_weights[b,patch_id].item()
                #单向weight
                for b in range(len(attention_weights)):
                    cur_masked_t = cur_inputs["t"][b]
                    for t_ in range(cur_masked_t+1):
                        for patch_id in patch_idx[t_]:
                            ia_attention_weights_uni[t_] +=  attention_weights[b,patch_id].item()
                    
                            #save a sample reconstruction

                
                target_patch_ids = cur_inputs['patch_id'].to(reconstruction.device)#[b]
                #print("target_patch_ids",target_patch_ids.size())
                gather_idx = target_patch_ids.unsqueeze(-1).unsqueeze(-1).repeat(1,1,predictions['logits'].size(-1))
                #print("gather_idx",gather_idx.shape)
                cur_logits = predictions['logits'].gather(dim = 1, index = gather_idx) #[:,target_patch_ids]#  [b,511,768] -> [b,768] [-1,1]
                
                #if 1==2:
                #########Logits需要调整分布范围，因为有一些灰色的地方其实应该是白色的##############
                #cur_logits_raw = cur_logits.clone()
                cur_logits[cur_logits>0] = 1.0


                #print("cur_logits shape",cur_logits.shape)
                #########Logits需要调整分布范围，因为有一些灰色的地方其实应该是白色的##############
                #print("cur_logits",cur_logits.size())
                #print("ref",ref.size())
                ref_logits_repeated = ref.repeat(cur_logits.size(0),1,1)
                #print("ref_logits_repeated",ref_logits_repeated.size())
                ref_logits = ref_logits_repeated.gather(dim = 1, index = gather_idx) #[1,511,768] -> [b,768]
                if self.config.pixel_remove_white:
                    #如果我们不计算白色部分的surprisal
                    #Very effective for persian
                    cur_logits[ref_logits == 1.0] = 1.0
                #cur_logits = clip(cur_logits)
                #ref_logits = clip(ref_logits)
                cur_patches = self.engine.model.unpatchify(cur_logits)#[b,3,16,16]
                #print("cur_patches",cur_patches.shape)
                ref_patches = self.engine.model.unpatchify(ref_logits)#[b,3,16,16]
                #print("ref_patches",ref_patches.shape)
                t_ = cur_inputs['t'][0]
                #plt.imshow(cur_patches[0].permute(1, 2, 0))
                #plt.savefig(f"examples/pred={t_}.png", bbox_inches='tight', pad_inches=0)
                #plt.imshow(ref_patches[0].permute(1, 2, 0))
                #plt.savefig(f"examples/ref={t_}-{ias[t_]}.png", bbox_inches='tight', pad_inches=0)
                #assert 1 == 2
                #print("ref_logits",ref_logits.size())
                #log_scale = torch.full_like(cur_logits, -2.0)
                
                

                surprisal = loss_fn(cur_logits, ref_logits)#-1 * discretized_logistic_likelihood(cur_logits, ref_logits, log_scale)  / (16*16) #loss_fn(clip(cur_logits), clip(ref_logits)) / cur_logits.size(-1) #->[b]
                    
                surprisal = surprisal.mean(dim = -1).reshape(-1) #[B]
                
                surprisal_l1 = mae_loss_fn(cur_logits, ref_logits)
                
                surprisal_l1 = surprisal_l1.mean(dim=-1).reshape(-1)

                #surprisal = surprisal[:,0,:]
                #print("surprisal",surprisal.shape)
                
                surprisal_ssim = ssim_fn.forward(cur_patches, ref_patches)
                #print("surprisal_ssim",surprisal_ssim)
                #print("cur_patches",cur_patches.shape)
                bs = cur_patches.size(0)
                #p = torch.softmax(cur_patches.reshape(bs, -1), dim=1).view_as(cur_patches)
                #eps = 1e-12
                #entropy = -(p.view(bs, -1) * torch.log(p.view(bs, -1) + eps)).sum(dim=1)
                prob = cur_logits.sigmoid()#[50, 80,30] [0.5,0.8,0.3]#[0,1] 16 * 16 = 256
                entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())#.mean(-1).sum().item()#->[b]
                if self.config.pixel_remove_white:
                    entropy = entropy * (ref_logits == 1.0).int()
                else:
                    entropy = entropy[:,0,:]
                
                entropy_gradient = patch_entropy_gradient(cur_patches)
                #print("entropy",entropy.shape)
                t_cur_batch = cur_inputs['t']
                
                for b in range(len(t_cur_batch)):
                    #print(b)
                    ia_data = {
                       "idx":t_cur_batch[b].item(),
                        "ia_surprisal_sum":surprisal[b].sum().item(),
                        "ia_surprisal_ssim":surprisal_ssim[b].item(),
                        "ia_surprisal_l1":surprisal_l1[b].sum().item(),
                        "ia_entropy_sum":entropy[b].sum().item() / (16*16),
                        "ia_entropy_grad":entropy_gradient[b].item(),
                        "ia_label":ias[t_cur_batch[b].item()],
                        "sample_idx":sample_idx,
                    }       
                    ia_datas.append(ia_data)
                #.....
            ia_attention_weights_by_sample += ia_attention_weights
            ia_attention_weights_uni_by_sample += ia_attention_weights_uni
        aggregated_ia_data = pd.DataFrame(ia_datas)
        aggregated_ia_data = aggregated_ia_data.groupby(['idx','sample_idx']).agg({
                    "ia_surprisal_sum": 'sum',
                    #"ia_surprisal_max": 'max',
                    "ia_entropy_sum": 'sum',
                    #"ia_entropy_max": 'max',
                    'ia_surprisal_ssim': 'sum',
                    'ia_surprisal_l1': 'sum',
                    #'ia_surprisal_ssim_max': 'max',
                    'ia_entropy_grad':'sum',
                    #'ia_entropy_grad_max':'max',
                    "ia_label": 'first',
                    'idx':'first',
                    'sample_idx':'first'
                }).reset_index(drop=True)
        if len(aggregated_ia_data) != len(ia_attention_weights_by_sample):
            # 使用
            logger.error("%s has bug", texts)
        aggregated_ia_data['ia_attention_sum'] = ia_attention_weights_by_sample[:len(aggregated_ia_data)]
        aggregated_ia_data['ia_attention_sum_uni'] = ia_attention_weights_uni_by_sample[:len(aggregated_ia_data)]
        aggregated_ia_data = aggregated_ia_data.merge(aggregated_ia_data.groupby('sample_idx')['ia_label'].count(), on ='sample_idx', suffixes=('', '_right'))
        aggregated_ia_data['length'] = aggregated_ia_data['ia_label_right']
        aggregated_ia_data['dist_to_sent_end'] = aggregated_ia_data['length'] - aggregated_ia_data['idx']
        aggregated_ia_data['ia_attention_sum_uni_norm'] = aggregated_ia_data['ia_attention_sum_uni'] / aggregated_ia_data['dist_to_sent_end']
        #aggregated_ia_data['ia_attention_sum_uni_norm'] = aggregated_ia_data['ia_attention_sum_uni'] / len()
        return aggregated_ia_data
    def probe_pixel_lm(self, texts, sep = " "):
        ia_datas = []
        for sample_idx, text in enumerate(texts):
            inputs, pre_encoding = self.engine.prepare_inputs(words = text,
                                    manual_mask = []
                                    )
            patch_lengths, patch_idx = self.engine.get_patch_index_by_words(words = text.split(sep))
            loss_fn = nn.MSELoss(reduction='sum')
            loss_fn_2 = nn.L1Loss(reduction='sum')
            ref = self.engine.model.patchify(inputs['pixel_values'])
            surprisals = []
            entropies = []
            ias = text.split(sep)
            for t in range(len(ias)):
                cur_surprisal = 0
                cur_entropy = 0

                for patch_id in patch_idx[t]:
                    
                    inputs, pre_encoding = self.engine.prepare_inputs(words = text,
                                            manual_mask = [patch_id],
                                            #mask_ratio = mask_ratio,
                                            #masking_max_span_length = masking_max_span_length,
                                            #masking_spacing = masking_spacing,
                                            #masking_cumulative_span_weights = masking_cumulative_span_weights
                                            )
                    
                    if self.config.pixel_lm_right_blur:
                        pixel_values = inputs['pixel_values']
                        blur = GaussianBlur(kernel_size=5, sigma=2.0)
                        blurred = blur(pixel_values)
                        inputs['pixel_values'][:,:,:,patch_id * 16:] = blurred[:,:,:,patch_id * 16:]
                    device = next(self.engine.model.parameters()).device
                    
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()}
                    
                    # Verify the move actually worked
                    #for k, v in inputs.items():
                    #    if isinstance(v, torch.Tensor):
                    #        print(f"  AFTER MOVE: key={k}, device={v.device}")
                    predictions = self.engine.get_predictions(inputs)
                    reconstruction = predictions['reconstruction']
                    plt.imshow(reconstruction.sum(0)[0,:,:600])
                    plt.savefig(f"examples/reconstruction_t{t}.png", bbox_inches='tight', pad_inches=0)
                    cur_logits = predictions['logits'][0,[patch_id]]
                    ref_logits = ref[0,[patch_id]]
                    
                    
                    
                    
                    surprisal = loss_fn(cur_logits, ref_logits) / cur_logits.size(-1) #L2 loss
                    surprisal = surprisal.item()
                    surprisal2 = loss_fn_2(cur_logits, ref_logits) / cur_logits.size(-1) #L1 loss
                    surprisal2 = surprisal2.item()
                    prob = cur_logits.sigmoid()#[50, 80,30] [0.5,0.8,0.3]
                    entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log()).mean(-1).sum().item()
                    cur_surprisal += surprisal
                    cur_entropy += entropy
                surprisals.append(cur_surprisal)
                entropies.append(cur_entropy)
                ia_data = {
                    "ia_label":ias[t],
                    "idx":t,
                    "ia_surprisal_sum":cur_surprisal,
                    "ia_entropy_sum":cur_entropy,
                    "sample_idx":sample_idx
                }
                ia_datas.append(ia_data)
        aggregated_ia_data = pd.DataFrame(ia_datas)
        return aggregated_ia_data
    def get_output_from_bidir_lm(self, inputs):
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        vocab_size = self.model.config.vocab_size

        all_logits = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=self.model.device
        )

        # layer -> batch, heads, seq_len, seq_len
        all_incoming_attn = None

        for pos in range(seq_len):
            masked_input_ids = input_ids.clone()
            masked_input_ids[:, pos] = self.tokenizer.mask_token_id

            masked_inputs = {k: v for k, v in inputs.items()}
            masked_inputs["input_ids"] = masked_input_ids

            with torch.no_grad():
                outputs = self.model(
                    **masked_inputs,
                    output_attentions=True,
                    return_dict=True,
                )

            all_logits[:, pos, :] = outputs.logits[:, pos, :]

            # outputs.attentions is tuple of length num_layers
            # each: (batch, heads, query_pos, key_pos)
            incoming_this_pos = torch.stack(
                [attn[:, :, :, pos] for attn in outputs.attentions],
                dim=1
            )
            # shape: (batch, layers, heads, seq_len)

            if all_incoming_attn is None:
                num_layers = len(outputs.attentions)
                num_heads = outputs.attentions[0].shape[1]
                all_incoming_attn = torch.zeros(
                    batch_size, seq_len, num_layers, num_heads, seq_len,
                    device=self.model.device
                )

            all_incoming_attn[:, pos, :, :, :] = incoming_this_pos

        return all_logits, all_incoming_attn
    def probe_lm(self, texts, remove_postfix = True, ia_labels = None):
        
        hyp_inputs = self.tokenizer([self.bos_token + text for text in texts], return_tensors = 'pt', add_special_tokens = False).to(self.model.device)
        
        with torch.no_grad():
            if self.is_bidir:
                logits, attentions = self.get_output_from_bidir_lm(hyp_inputs)
                attentions = process_incoming_attention(attentions)

            else:
                outputs = self.model(**hyp_inputs, output_attentions = True)
                logits = outputs.logits.detach()
                attentions = get_attention_head_map(outputs.attentions, [-1,-2])
                attentions = attentions.sum(dim = 1)
        batch_token_data = []
        batch_ia_data = []
        for idx in range(len(logits)):
            cur_input_ids = hyp_inputs.input_ids[idx].unsqueeze(0)
            cur_logits = logits[idx].unsqueeze(0)
            surprisal, entropy, selected_logits = self.processing_logits(cur_input_ids, cur_logits, 1)
            shift_input_ids = hyp_inputs.input_ids[idx,1:].unsqueeze(0)
            token_data = self.get_probability_measures(shift_input_ids, selected_logits, entropy, surprisal, ia_labels = ia_labels[idx])
            #if attentions is not None:
            token_data['attention'] = attentions[idx].squeeze().tolist()[1:]#Remove cls
            if remove_postfix:
                if ia_labels[idx] is not None:
                    token_data = token_data[token_data['word_id'] < len(ia_labels[idx])]
                else:
                    if self.config.spaced:
                        token_data = token_data[token_data['word_id'] < len(texts[idx].split(" "))]
                    else:
                        print("Without ia, we cannot decide where is postfix for this language")
                        assert 1 == 2
            ia_data = self.gather_ia_feat_from_token_data(token_data)

            token_data['sample_idx'] = idx
            ia_data['sample_idx'] = idx
            batch_token_data.append(token_data)
            batch_ia_data.append(ia_data)
        aggregated_token_data = pd.concat(batch_token_data, axis = 0)
        aggregated_ia_data = pd.concat(batch_ia_data, axis = 0)
        return aggregated_token_data, aggregated_ia_data
    def get_probability_measures(self, shift_input_ids, selected_logits, entropy, surprisal, ia_labels = None):
        #一次只能塞一条，也就是说batch size一定是1

        seqs = [self.processor.batch_decode(shift_input_ids[:,:i+1]) for i in range(shift_input_ids.size(-1))]#[T, B]
        seqs = [list(row) for row in zip(*seqs)]
        #有些语言会decode出来�，要去掉
        seqs = [[seq.replace("�","") for seq in sample] for sample in seqs]
        tokens = [[seq.replace(sample[i-1],"") if i > 0 else seq for i,seq in enumerate(sample)] for sample in seqs]

        
        if not self.config.is_spaced:
            assert ia_labels is not None
            depth_ids = get_depth_ids_by_ia_labels(self.processor, 
                                                shift_input_ids[0], 
                                                ia_labels, 
                                                sep = '',
                                                decoding_clean_func=lambda x:re.sub(r"(?:\s|\[CLS\])", "", x)
                                                )
        #elif self.config.is_cjk:
        #    assert ia_labels is not None
        #    depth_ids = get_depth_ids_by_ia_labels(self.processor, 
        #                                        shift_input_ids[0], 
        #                                        ia_labels, 
        #                                        sep = '',
        #                                        decoding_clean_func=lambda x:re.sub(r"(?:\s|\[CLS\])", "", x)
        #                                        )
        else:
            depth_ids = get_depth_ids(self.processor, shift_input_ids)
            depth_ids = depth_ids.squeeze(0).tolist()
            ##Flatten Data
        #print("depth_ids",depth_ids)
        #tokens = []
        #for i in range(depth_ids.max().item() + 1):
        #    cur_token_ids = shift_input_ids[depth_ids==i]
        #    reconst = self.processor.tokenizer.decode(cur_token_ids)

        #    tokens.append(reconst.strip())
        #print("depth_ids",depth_ids)
        token_data = {
            "token":tokens[0],
            "logits":selected_logits[0],
            "entropy":entropy[0],
            "surprisal":surprisal[0],
            "word_id": depth_ids,
        
        }
        

        #print("token_data",token_data)
        token_data = pd.DataFrame(token_data)
        return token_data
        
    def process_gaze_events_to_reading_measures(self, trial):
        trial = trial.dropna(subset=[self.colname_of_ia_idx])
        sentence_id = trial[self.colname_of_sentence_id].tolist()[0]
        subj_id = trial[self.colname_of_subj_id].tolist()[0]
        #print(trial[self.colname_of_stimuli_features].values[0])
        stimuli = trial[self.colname_of_stimuli_features].values[0]
        if stimuli is None or type(stimuli) is not str:
            return None
        stimuli = eval(stimuli)
        #print("stimuli",stimuli)
        
        ias = [x[0] for x in stimuli]
        cols = [x[1] for x in stimuli]
        rows = [x[2] for x in stimuli]
        ia_ids_in_stiumli = [x[3] for x in stimuli]
        if self.config.feat_idx_for_sent_idx is not None:
            sent_idx = [x[self.config.feat_idx_for_sent_idx] for x in stimuli]
        else:
            sent_idx = None
        wns = trial[f'{self.colname_of_ia_idx}'] - self.ia_idx_start_from
        wns = wns.tolist()
        #print("wns",wns)
        durs = trial[f'{self.colname_of_fixation}'].tolist()
        if self.colname_of_landing_position:
            lps = trial[f'{self.colname_of_landing_position}']
            lps = lps.tolist()
        elif self.colname_of_x and self.colname_of_ia_left:
            lps = trial[f'{self.colname_of_x}'] - trial[f'{self.colname_of_ia_left}'] #/(example_trial['CURRENT_FIX_INTEREST_AREA_RIGHT'] - example_trial['CURRENT_FIX_INTEREST_AREA_LEFT'])
            lps = lps.tolist()
        else:
            lps = None
        if self.colname_of_pupil:
            pds = trial[f'{self.colname_of_pupil}']
            pds = pds.tolist()
        else:
            pds = None
        ia_measures = {self.colname_of_ia_idx:[i for i in range(len(ias))]}
        ia_measures[self.colname_of_ia_label] = ias
        ia_measures['rows'] = rows
        ia_measures['cols'] = cols
        if sent_idx is not None:
            ia_measures['sent_idx'] = sent_idx
        #print("wns",wns)
        #if self.config.dataset_name == 'lpsimp':
        #    duration_measures = get_durations_and_counts(wns, durs)[0]
        #else:
        #print(stimuli)
        #print("wns",wns)
        #print("max wn",max(wns))
        #print("ias",ias)
        #print("len ias",len(ias))
        #print("ia_ids_in_stiumli",ia_ids_in_stiumli)
        #print("len ia measures",len(ias))
        if self.config.dataset_name in ['lpsimp']:
            raw_duration_measures = get_durations_and_counts(wns, durs, max(ia_ids_in_stiumli))[0]
            
            mask = [1 if i + self.ia_idx_start_from in ia_ids_in_stiumli else 0 for i in range(max(ia_ids_in_stiumli))]
            #print("mask",mask)
            duration_measures = {}
            for k,v in raw_duration_measures.items():
                #print("len raw v",len(v))
                true_v = [v[i] for i in range(len(mask)) if mask[i] == 1]
                #print("len true v",len(true_v))
                duration_measures[k] = true_v
        elif self.config.dataset_name in ['lptrad']:
            corrected_wns = []
            corrected_durs = []
            for wn, dur in zip(wns, durs):
                if wn + self.ia_idx_start_from not in ia_ids_in_stiumli:
                    continue
                else:
                    corrected_wns.append(wn)
                    corrected_durs.append(dur)
            duration_measures = get_durations_and_counts(corrected_wns, corrected_durs, len(ias))[0]
        else:
            duration_measures = get_durations_and_counts(wns, durs, len(ias))[0]
        
        self.reading_measure_columns += list(duration_measures.keys())
        ia_measures.update(duration_measures)
        if lps:
            lp_measures = get_normal_measures(wns, lps, len(ias))[0]
            lp_measures = {k.replace("MEASURE","LANDING_POSITION"):v for k,v in lp_measures.items()}
            self.reading_measure_columns += list(lp_measures.keys())
            ia_measures.update(lp_measures)
        if pds:
            pd_measures = get_normal_measures(wns, pds, len(ias))[0]
            pd_measures = {k.replace("MEASURE","PUPIL_DILATION"):v for k,v in pd_measures.items()}
            self.reading_measure_columns += list(pd_measures.keys())
            ia_measures.update(pd_measures)
        #print("ia_measures",ia_measures)
        #for k,v in ia_measures.items():
        #    print(k)
        #    print(len(v))
        ia_measures = pd.DataFrame(ia_measures)
        ia_measures['sentence_id'] = sentence_id
        ia_measures['subj_id'] = subj_id
        return ia_measures
    def load_reading_measures(self):


        df = pd.read_csv(self.path_to_dataset, sep = "\t", keep_default_na=False,
            na_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                    '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NaN', 'None', 'n/a',
                    'nan', ''])
        print(df.head())

        grouping_factors = [self.colname_of_subj_id, self.colname_of_sentence_id]
        if self.colname_of_experiment_paradigms is not None:
            grouping_factors += [self.colname_of_experiment_paradigms]
        if self.filter_by is not None:
            colname_to_filter = self.filter_by.split("==")[0]
            value_to_filter = self.filter_by.split("==")[1]
            df = df[df[colname_to_filter] == value_to_filter]

        trials = df.groupby(grouping_factors)
        trial_ids = [i for i in range(len(trials))]
        bar = tqdm(trial_ids)
        all_ia_measures = []
        trial_keys = [x for x in trials.groups.keys()]
        for trial_id in bar:
            
            trial_key = trial_keys[trial_id]
            example_trial = trials.get_group(trial_key)
            if self.colname_of_fix_id is not None:
                example_trial = example_trial.sort_values(by = self.colname_of_fix_id)
            if self.data_is_rm:
                ia_measures = example_trial
            else:
                #try:
                ia_measures = self.process_gaze_events_to_reading_measures(example_trial)
                if ia_measures is None:
                    continue
                #except Exception as e:
                #    print(f"Found error for :{trial_id}, {trial_key}. \n Trial is like: {example_trial}")
                #    print(e)
                #    continue
            
            ia_measures['trial_id'] = trial_id
            for i,tk in enumerate(grouping_factors):
                ia_measures[tk] = trial_key[i]
            if self.generate_sent_idx:
                ias = ia_measures[self.colname_of_ia_label].tolist()
                sent_indices = generate_sentence_id(ias)
                ia_measures['sent_idx'] = sent_indices
                sents = []
                for idx in range(min(sent_indices), max(sent_indices) +1):
                    cur_ias = [ias[k] for k in range(len(sent_indices)) if sent_indices[k] == idx]
                    cur_sent = "**".join(x for x in cur_ias)
                    sents.append(cur_sent)
                #print(sents)
                #print("sent_indices",sent_indices)
                ia_measures['sentences'] = [sents[t-min(sent_indices)] for t in sent_indices]
                
            all_ia_measures.append(ia_measures)
        df_reading_measures = pd.concat(all_ia_measures, axis = 0)
        if self.generate_sent_idx:
            df_reading_measures[self.colname_of_sentence_id] = df_reading_measures['sentences'].factorize()[0]
        #if self.seperate_stimuli_by is not None:
        #    df_reading_measures[self.colname_of_sentence_id] = (
        #        df_reading_measures[self.colname_of_sentence_id].astype(str)
        #        + "_"
        #        + df_reading_measures[self.seperate_stimuli_by].astype(str)
        #    )
        self.df_reading_measures = df_reading_measures
    def normalize_between_subjects(self):
        assert self.df_reading_measures is not None
        df = self.df_reading_measures
        reading_measures = self.reading_measure_columns
        cols_to_keep = [x for x in df.columns if x not in reading_measures and x != self.colname_of_subj_id]
        grouping_factors = [self.colname_of_sentence_id, self.colname_of_ia_idx]
        
        if self.colname_of_experiment_paradigms is not None:
            grouping_factors += [self.colname_of_experiment_paradigms]
        
        
        
        df_normed = (
            df
            .groupby(grouping_factors, as_index=False)
            .agg({
                **{col: 'first' for col in cols_to_keep},
                **{col: mean_nonneg for col in reading_measures},
                self.colname_of_subj_id: lambda x:len(x)
                })
                )
                # 1. 按colname_of_sentence_id分组，生成input_id
        # 1. 建立映射字典
        sentence_to_input_id = {}
        counter = 0
        for val in df_normed[self.colname_of_sentence_id]:
            if val not in sentence_to_input_id:
                sentence_to_input_id[val] = counter
                counter += 1

        # 2. 覆盖stimuli_id
        df_normed['stimuli_id'] = [sentence_to_input_id[v] for v in df_normed[self.colname_of_sentence_id]]
        df['stimuli_id'] = [sentence_to_input_id[v] for v in df[self.colname_of_sentence_id]]
        front_cols = ['stimuli_id', self.colname_of_ia_idx, self.colname_of_ia_label]
        df_normed = df_normed[front_cols + [c for c in df_normed.columns if c not in front_cols]]
        df = df[front_cols + [c for c in df_normed.columns if c not in front_cols]]
        self.df_reading_measures = df
        self.df_reading_measures_normalized = df_normed
    def annotate(self, sigma, lm_only = False, batch_size = 5, filter_conditions: list[tuple] = None, group_name = ""):
        if self.normalize_between_subjects:
            df = self.df_reading_measures_normalized
        else:
            df = self.df_reading_measures
        
        if filter_conditions is not None:
            mask = pd.Series(True, index=df.index)

            for field, condition in filter_conditions:
                mask &= df[field].apply(condition)

            df = df[mask].reset_index(drop=True)

        stimuli_frames = df.groupby(self.config.probe_group_by)#['stimuli_id'])
        frame_ids = [i for i in range(len(stimuli_frames))]
        frame_keys = [x for x in stimuli_frames.groups.keys()]
        bar = tqdm(frame_ids)
        token_reports = []
        ia_reports = []
        for frame_id in bar:
            frame_key = frame_keys[frame_id] #frame_key is stimuli_id
            frame = stimuli_frames.get_group(frame_key)
            
            ias = frame[self.colname_of_ia_label].tolist()
            ias = [str(x) for x in ias]
            if self.config.ia_post_process_func is not None:
                
                ias = [self.config.ia_post_process_func(ia) for ia in ias]
            if self.config.is_spaced or self.is_pixel_lm:
                text = ' '.join(ia for ia in ias)
            else:
                #assert 1==2
                text = ''.join(ia for ia in ias)
            cut_off = self.config.cut_off

            alpha_function = lambda t: np.log(t + 1) / np.log(cut_off + 1) if t < cut_off else 1.0
            batches = [[]]
            if self.config.use_official_aoi_positions:
                #stimuli = eval(frame[self.colname_of_stimuli_features].values[0])
                cols = frame['cols'].tolist()
                rows = frame['rows'].tolist()
            else:
                rows = [0 for ia in ias]
                cols = [i for i,_ in enumerate(ias)]
            position_2ds = [(row, col) for row, col in zip(rows, cols)]
            if self.is_rtl:
                ias_to_present = [present_rtl(ia) for ia in ias]
                position_2ds = reshape_position_ids_for_rtl(position_2ds)
            
                
            #print("ias",ias)
            if not lm_only and not self.is_pixel_lm:
                self.plotter.output_path = "cache"
                if self.is_rtl:
                    ia_data = self.plotter.draw_ia_plot(ias_to_present, position_2ds, title = "", draw_box=False, crop=self.config.crop)
                else:
                    ia_data = self.plotter.draw_ia_plot(ias, position_2ds, title = "", draw_box=False, crop=self.config.crop)
            
            
            if self.is_pixel_lm:
                data = {"frame_id":frame_id,
                        "t": None,
                        "pil_img":None,
                        "text":text,
                        "ias":ias,
                        "offset":0
                        }
                batches[-1].append(data)
            elif lm_only:
                data = {"frame_id":frame_id,
                        "t": None,
                        "pil_img":None,
                        "text":text,
                        "ias":ias,
                        "offset":0
                        }
                batches[-1].append(data)
            
            elif sigma == -1:
                mask_idx = [i for i in range(len(ias))]
                pil_img = self.render(ias, 0, mask_idx, alpha_function, )
                data = {"frame_id":frame_id,
                        "t": None,
                        "pil_img":pil_img,
                        "text":text,
                        "ias":ias,
                        "offset":0
                        }
                batches[-1].append(data)
            else:
                for t in range(len(ias)):
                    if self.is_rtl:
                        mask_idx = [i for i in range(len(ias)-1-t, -1, -1)]
                    else:
                        mask_idx = [i for i in range(t,len(ias))]

                    if 0<cut_off < 1:
                        boxes = [(ia['r0'],ia['r1'], ia['c0'], ia['c1']) for ia in self.plotter.get_boxes()]
                        min_box_id = min(mask_idx) if not self.is_rtl else max(mask_idx)
                        cut_off = np.abs(boxes[min_box_id]['c1'] - boxes[min_box_id]['c0'])
                        alpha_function = lambda t: np.log(t + 1) / np.log(cut_off + 1) if t < cut_off else 1.0
                    if self.config.local_window is not None:
                        #print("self.config.local_window",self.config.local_window)
                        left, right = self.config.local_window
                        if self.is_rtl:
                            window_idx = [len(ias) - x for x in list(range(t - right + 1, t + left + 1))]
                        else:
                            window_idx = list(range(t - left, t + right))
                        window_idx = [x for x in window_idx if 0 <= x < len(ias)]
                        
                        pil_img = self.render(ias, sigma, mask_idx, alpha_function, window_idx)
                    else:
                        pil_img = self.render(ias, sigma, mask_idx, alpha_function)

                    pil_img.save(f"examples/example_{t}.png")
                    if self.config.left_truncate_to_picture:
                        offset = max(t - left, 0)
                        trunc_ias = ias[offset:]#left = 2, t = 5, offset = 3, if idx 1, true idx 1 + 3 = 4
                        if self.config.is_spaced or self.is_pixel_lm:
                            trunc_text = ' '.join(ia for ia in trunc_ias)
                        else:
                            #assert 1==2
                            trunc_text = ''.join(ia for ia in trunc_ias)
                        data = {"frame_id":frame_id,
                                "t": t,
                                "pil_img":pil_img,
                                "text":trunc_text,
                                "ias":trunc_ias,
                                "offset":offset
                                }

                    else:
                        data = {"frame_id":frame_id,
                                "t": t,
                                "pil_img":pil_img,
                                "text":text,
                                "ias":ias,
                                "offset":0
                                }
                    if len(batches[-1]) >= batch_size:
                        batches.append([])
                    batches[-1].append(data)
            for batch in tqdm(batches):
                images = [sample['pil_img'] for sample in batch]
                texts = [sample['text'] for sample in batch]
                offsets = [sample['offset'] for sample in batch]

                ts = [sample['t'] for sample in batch]
                ia_labels = [sample['ias'] for sample in batch]
                if self.is_pixel_lm:
                    if self.config.dataset_name in ['ge_uk_p','ge_zu_p']:
                        texts = [re.compile("\s+").sub(" ", x) for x in texts]
                        sample_id_to_sent_ids = []
                        splited_texts = []
                        
                        for k in range(len(texts)):
                            new_texts = split_sentence(texts[k],)
                            for j in range(len(new_texts)):
                                sample_id_to_sent_ids.append(k)
                            splited_texts += new_texts
                                
                        print("splited_texts",splited_texts)
                        print("sample_id_to_sent_ids",sample_id_to_sent_ids)
                        ia_report = self.probe_pixel_lm_batch(splited_texts)
                        ia_report = ia_report.sort_values(['sample_idx','idx'])
                        ia_report['sample_idx'] = ia_report['sample_idx'].apply(lambda x:sample_id_to_sent_ids[x])
                        ia_report['idx'] = ia_report.groupby('sample_idx').cumcount()
                    else:
                        print("texts",texts)
                        if self.config.pixel_whole_ia_masking:
                        
                            ia_report = self.probe_pixel_lm_batch(texts, batch_size = 1)
                        else:
                            ia_report = self.probe_pixel_lm_batch(texts)
                        print("ia_report",ia_report)
                    token_report = ia_report
                elif lm_only:
                    token_report, ia_report = self.probe_lm(texts, ia_labels=ia_labels)
                elif sigma == -1:
                    token_report, ia_report = self.probe(images, texts, remove_image=True, prefix = None, ia_labels=ia_labels, offsets = offsets)
                else:
                    token_report, ia_report = self.probe(images, texts, prefix = None, ia_labels=ia_labels, offsets = offsets)
                token_report['t'] = token_report['sample_idx'].apply(lambda x:ts[x])
                ia_report['t'] = ia_report['sample_idx'].apply(lambda x:ts[x])

                if len(self.config.probe_group_by) == 1:
                    token_report[self.config.probe_group_by[0]] = frame_key
                    ia_report[self.config.probe_group_by[0]] = frame_key
                else:
                    for i,field_name in enumerate(self.config.probe_group_by):
                        token_report[field_name] = frame_key[i]#"-".join(x for x in frame_key)
                        ia_report[field_name] = frame_key[i]#"-".join(x for x in frame_key)
                token_report['sigma'] = sigma
                ia_report['sigma'] = sigma
                token_reports.append(token_report)
                ia_reports.append(ia_report)


        
        df_token_metrics = pd.concat(token_reports, axis=0)
        df_ia_metrics = pd.concat(ia_reports, axis=0)
        front_cols = ['stimuli_id', 'idx', 'ia_label', 'sigma']
        
        #df_token_metrics = df_token_metrics[front_cols + [c for c in df_token_metrics.columns if c not in front_cols]]
        df_ia_metrics = df_ia_metrics[front_cols + [c for c in df_ia_metrics.columns if c not in front_cols]]
        
        if self.config.cut_off > 0:
            fname_ia = f'ia_{group_name}_c{self.config.cut_off}.csv'
            fname_tk = f'token_{group_name}_c{self.config.cut_off}.csv'
        else:
            fname_ia = f'ia_{group_name}.csv'
            fname_tk = f'token_{group_name}.csv'
        
        

        path_to_ia_metrics = os.path.join(self.path_to_precomputed, self.config.dataset_name, fname_ia)
        path_to_token_metrics = os.path.join(self.path_to_precomputed, self.config.dataset_name, fname_tk)
        df_token_metrics.to_csv(path_to_token_metrics, sep = '\t')
        df_ia_metrics.to_csv(path_to_ia_metrics, sep = '\t')
        return df_token_metrics, df_ia_metrics

def mean_nonneg(x):
    x = x.to_numpy()
    x = x[x >= 0]
    return np.mean(x) if x.size else np.nan
        
            
            