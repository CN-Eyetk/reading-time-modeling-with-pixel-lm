import torch
from pixel import PIXELConfig, PIXELForPreTraining
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer
from pixel.data.rendering.arabic_renderer import PangoCairoArabicTextRenderer
from typing import Any, Dict, List, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import torch.nn.functional as F
from pixel import (
    UPOS_LABELS,
    Modality,
    PyGameTextRenderer,
    SpanMaskingGenerator,
    PangoCairoTextRenderer,
    get_transforms,
    get_attention_mask,
    resize_model_embeddings,
    truncate_decoder_pos_embeddings,
    Encoding
)
from process import get_attention_head_map
def clip(x: torch.Tensor):
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x

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
def log_image(img: torch.Tensor, img_name: str, do_clip: bool = True):
    if do_clip:
        img = clip(img)
    #to_pil = T.ToPILImage()
    #img = to_pil(img)
    #display(img)
    #img.save(f"{img_name}.png")
#Image.fromarray(image_to_display.numpy())
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.axis('off') # Hide axes for cleaner image display
    plt.show()
    plt.savefig(f"{img_name}.png")
    plt.close()
    
    #wandb.log({img_name: wandb.Image(img)})
    
def get_best_device():
    if torch.cuda.is_available():
        # Pick GPU with most free memory
        best_gpu = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.mem_get_info(i)[0]  # free memory
        )
        return torch.device(f"cuda:{best_gpu}")
    return torch.device("cpu")

def collate(features, remove_labels = False):
    batch = {}
    for k,v in features[0].items():
        if k == "label_ids" and remove_labels:
            continue
        batch[k] = torch.stack([torch.tensor(feature[k]) for feature in features], dim = 0)#.to(model.device)
    return batch




class PixelLM:
    def __init__(self, path_to_model: str, fallback_fonts_dir = None, is_m4 = True, use_arabic_renderer = False, is_hebrew = False, is_denoiser = False):
        self.is_m4 = is_m4
        self.is_denoiser = is_denoiser
        self.use_arabic_renderer = use_arabic_renderer
        self.is_hebrew = is_hebrew
        if is_m4:
            config_kwargs = {
                    "cache_dir": None,
                    "revision": "56bfcbf71e98f613ee00f8efb7a607bf0058f1e6",
                    "use_auth_token": False,
                    
                }
            if not self.use_arabic_renderer:
                renderer_class = PangoCairoBigramsRenderer
                processor = renderer_class.from_pretrained(path_to_model, fallback_fonts_dir=fallback_fonts_dir, rgb=False, **config_kwargs)
            else:
                renderer_class = PangoCairoArabicTextRenderer
                processor = renderer_class.from_pretrained(path_to_model, fallback_fonts_dir=fallback_fonts_dir, rgb=False,  **config_kwargs)
            
            config = PIXELConfig.from_pretrained(path_to_model, **config_kwargs)

            model_class = PIXELForPreTraining
            model = model_class.from_pretrained(path_to_model, config=config,device_map={"": "cuda:0"},  **config_kwargs).eval()
        else:
            processor = PyGameTextRenderer.from_pretrained(path_to_model)
            config = PIXELConfig.from_pretrained(path_to_model)
            model = PIXELForPreTraining.from_pretrained(path_to_model, config=config, device_map={"": "cuda:0"}).eval()
        resize_model_embeddings(model, processor.max_seq_length)
        truncate_decoder_pos_embeddings(model, processor.max_seq_length)
        #if is_m4:
        #model = model.cuda()
        transforms = get_transforms(
            do_resize=True,
            do_normalize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
        self.processor = processor
        self.config = config
        self.model = model
        print("**********************************************************************")
        print(next(self.model.parameters()).device)  # ← check after .to()
        self.transforms = transforms

    def get_outputs(self, inputs):
        device = next(self.model.parameters()).device          # ← get model's device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}                  # ← move inputs
        with torch.inference_mode():
            outputs = self.model(**inputs, output_attentions = True)
        return outputs
    def txt_to_pixel(self, text):
        encoding = self.processor(text=text)
        attention_mask = get_attention_mask(
            num_text_patches=encoding.num_text_patches, seq_length=self.processor.max_seq_length
        )
        img = Image.fromarray(encoding.pixel_values)
        transformed_pixel_values = self.transforms(img).unsqueeze(0)
        return {
            "attention_mask":attention_mask,
            "encoding":encoding,
            "pixel_values":encoding.pixel_values,
            "transformed_pixel_values":transformed_pixel_values,
            "img":img
        }
    def generate_manual_mask(self, manual_mask):
        #if manual_mask is not None:
        mask_ratio = sum(manual_mask) / self.processor.max_seq_length
        self.config.update({"mask_ratio": mask_ratio})
        self.model.config.mask_ratio = mask_ratio
        mask = torch.zeros(size=(1, self.processor.max_seq_length))
        for idx in range(len(manual_mask)):
            mask[0][manual_mask[idx]] = 1
        return mask
    def extend_to_manual_mask(self, manual_mask):
        #if manual_mask is not None:
        mask_ratio = sum(manual_mask) / self.processor.max_seq_length
        self.config.update({"mask_ratio": mask_ratio})
        self.model.config.mask_ratio = mask_ratio
        mask = torch.zeros(size=(1, self.processor.max_seq_length))
        for idx in range(len(manual_mask)):
            mask[0][idx] = manual_mask[idx]
        return mask
    def prepare_inputs(self, words, 
                    manual_mask = None,
                    mask_ratio = None,
                    masking_max_span_length = None,
                    masking_spacing = None,
                    masking_cumulative_span_weights = None
                    ):
        pre_encoding = self.txt_to_pixel(words)
        inputs = {"pixel_values": pre_encoding['transformed_pixel_values'].float(), 
                "attention_mask": pre_encoding['attention_mask'].unsqueeze(0)}
        if manual_mask is not None:
            manual_mask = self.generate_manual_mask(manual_mask)
            inputs.update({"patch_mask": manual_mask})
        else:
            num_text_patches = pre_encoding['encoding'].num_text_patches
            mask = self.generate_span_mask(num_text_patches = num_text_patches,
                                        mask_ratio = mask_ratio,
                                        masking_max_span_length = masking_max_span_length,
                                        masking_spacing = masking_spacing,
                                        masking_cumulative_span_weights = masking_cumulative_span_weights
                                        )
            
            inputs.update({"patch_mask": mask.unsqueeze(0)})
        return inputs, pre_encoding
    def prepare_inputs_for_cognitive_modeling(self, 
                                            stimuli: Tuple[str, str, str],
                                            mask_right = False,
                                            right_window = 10,
                                            add_space = False
                                            
                                            ):
        assert type(stimuli) == tuple and len(stimuli) == 3
        if add_space:
            sep = " "
        else:
            sep = ""
        pre_encoding = self.txt_to_pixel(text = sep.join(seg for seg in stimuli))
        prefix_encoding = self.txt_to_pixel(text = stimuli[0])
        prefix_with_target_encoding = self.txt_to_pixel(text = stimuli[0] + sep + stimuli[1])
        inputs = {"pixel_values": pre_encoding['transformed_pixel_values'].float(), 
                "attention_mask": pre_encoding['attention_mask'].unsqueeze(0)}
        #sep_patches = pre_encoding['encoding'].sep_patches
        inputs = {"pixel_values": pre_encoding['transformed_pixel_values'].float(), 
                "attention_mask": pre_encoding['attention_mask'].unsqueeze(0)}
        max_active_patch_index = pre_encoding['encoding'].num_text_patches
        if add_space:
            target_start_patch_index = pre_encoding['encoding'].word_starts[prefix_encoding['encoding'].word_patch_map['total_number_of_words']]
            target_end_patch_index = pre_encoding['encoding'].word_starts[prefix_with_target_encoding['encoding'].word_patch_map['total_number_of_words']]
        else:
            target_start_patch_index = prefix_encoding['encoding'].num_text_patches
            target_end_patch_index = prefix_with_target_encoding['encoding'].num_text_patches
        print("target_start_patch_index:",target_start_patch_index)
        print("target_end_patch_index:",target_end_patch_index)
        if mask_right:
            manual_mask = [1 if target_start_patch_index <= i < target_end_patch_index or i > target_end_patch_index + right_window  else 0 for i in range(max_active_patch_index + 1) ]
        else:
            manual_mask = [1 if target_start_patch_index <= i < target_end_patch_index else 0 for i in range(max_active_patch_index + 1) ]
        #print("manual_mask 1",manual_mask)
        manual_mask = self.generate_manual_mask(manual_mask)
        #print("manual_mask 2",manual_mask)
        inputs.update({"patch_mask": manual_mask})
        return inputs, pre_encoding

    def postprocess_outputs(self, inputs, outputs):
        logits = outputs['logits'].detach().cpu()
        attention_weight = [attn.detach().cpu() for attn in outputs['attentions']]
        head_map = get_attention_head_map(attention_weight,[-2,-1]).sum(dim = 1)
        head_map = head_map[:,:outputs["mask"].shape[-1]]#Trunc
        #head_map = head_map.clamp(0, 100)#[b, t=511]
        
        mask = outputs['mask'].detach().cpu()
        #print("mask",mask.sum(dim = -1))
        mask = mask.unsqueeze(-1).repeat(1, 1, self.processor.pixels_per_patch ** 2 * 3)
        mask = self.model.unpatchify(mask)#.squeeze()  # 1 is removing, 0 is keeping
        
        predictions = self.model.unpatchify(logits).detach().cpu()#.squeeze()
        original_img = self.model.unpatchify(self.model.patchify(inputs['pixel_values']))#.squeeze()
        attention_mask = inputs['attention_mask'].unsqueeze(-1).repeat(1, 1, self.processor.pixels_per_patch ** 2 * 3)
        attention_mask = self.model.unpatchify(attention_mask)#.squeeze()
        attention_mask = attention_mask.detach().cpu()
        masked_predictions = predictions * mask * attention_mask
        original_img = original_img.detach().cpu()
        im_masked = original_img * (1 - mask)
        reconstruction = (
            original_img * (1 - (torch.bitwise_and(mask == 1, attention_mask == 1)).long())
            + predictions * mask * attention_mask
        )
        return {"logits":logits,
                "attention_mask":attention_mask,
                "mask":mask,
                "predictions":predictions,
                "original_img":original_img,
                "masked_predictions":masked_predictions,
                "im_masked":im_masked,
                "reconstruction":reconstruction,
                "head_map":head_map
                }
    def get_predictions(self, inputs):
        outputs = self.get_outputs(inputs)
        return self.postprocess_outputs(inputs, outputs)
    def get_patch_index_by_words(self, words):
        if self.is_m4:
            text = " ".join(word for word in words)
        else:
            text = words
        pre_encoding = self.txt_to_pixel(text)
        word_starts = pre_encoding['encoding'].word_starts

        word_ends = word_starts[1:] + [pre_encoding['encoding'].num_text_patches + 1]
        path_lengths = np.array(word_ends) - np.array(word_starts)
        path_lengths = path_lengths[:-1] #Remove patch
        patch_idx = []
        
        for s,e in zip(word_starts, word_ends):
            if self.use_arabic_renderer or self.is_hebrew:
                patch_idx.append([x for x in range(s, e)][::-1])
            else:
                patch_idx.append([x for x in range(s, e)])
        patch_idx = patch_idx[:-1]
        if self.use_arabic_renderer  or self.is_hebrew:
            path_lengths = path_lengths[::-1]
            patch_idx = patch_idx[::-1]
        return path_lengths, patch_idx
    def get_patch_index_by_words_without_space(self, words, sep=""):
        last_patch = 0
        path_lengths = []
        patch_idx = []
        for i in range(len(words)):
            cur_seq = sep.join(x for x in words[:i+1])
            encoding = self.processor(cur_seq)
            pixel_vals = encoding.pixel_values
            cur_sep_patches = encoding.sep_patches[-1]
            path_lengths.append(cur_sep_patches - last_patch)
            patch_idx.append([x for x in range(last_patch, cur_sep_patches)])
            last_patch = cur_sep_patches
        return path_lengths, patch_idx
            
            
def demo():
    engine = PixelLM("/home/lijunlin/pixel/pixel/Team-PIXEL-checkpoints/pixel-base")
    inputs, pre_encoding = engine.prepare_inputs(
    words = ["""The MAE decoder is only used during pre-training to perform the image reconstruction task (only the encoder is used to produce image representations for recognition).
Therefore, the decoder architecture can be flexibly designed."""],
    manual_mask=[1 if x > 10 else 0 for x in range(30)])
    predictions = engine.get_predictions(inputs)
    engine.visualize(predictions['reconstruction'],'reconstruction.png')



class PyGameTextRendererForCognitiveModeling(PyGameTextRenderer):
    def _render_stimuli(self, stimuli: Tuple[str, str, str], compact = True, spacing = True):
        surface = self._get_empty_surface()
        sep_patches = []
        left, target, right = stimuli
        if spacing:
            target = " " + target
            right = " " + right
        offset = 2
        
        surface, left_width = self._render_single_sentence(left, offset, surface)
        
        if not compact:
            offset = self._get_offset_to_next_patch(offset + left_width)
        else:
            offset = offset + left_width

        sep_patches.append(offset // self.pixels_per_patch)
        surface, target_width = self._render_single_sentence(target, offset, surface)
        
        if not compact:
            offset = self._get_offset_to_next_patch(offset + target_width)
        else:
            offset = offset + target_width

        sep_patches.append(offset // self.pixels_per_patch)
        
        surface, right_width = self._render_single_sentence(right, offset, surface)
        offset = self._get_offset_to_next_patch(offset + right_width + 2)
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)
        
        num_text_patches = math.ceil(offset / self.pixels_per_patch)
        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches
        )
        return encoding
    def _render_words_to_surface(self, words: List[str]) -> Encoding:
        """
        Renders a list of words to a surface and keeps track of
        (a) how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches
        and (b) the patch index, starting with 0, where each word begins on the rendered surface

        Args:
            words (`List[str]`):
                The list of words to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered words and metadata
        """

        # Keep track of the patch index at which each new word begins
        word_start_indices = [0]

        # Start with blank surface
        surface = self._get_empty_surface()
        sep_patches = []

        # Pad left with 2px
        offset = 2
        skip_last = False

        # Render each word to the start of the next patch but at least a whitespace width apart
        for word in words[:-1]:

            surface, offset = self._render_single_word(f"{word}", offset, surface)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break

        # Last word is rendered without whitespace
        if not skip_last:
            surface, offset = self._render_single_word(words[-1], offset, surface, is_last=True)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches,
            word_starts=word_start_indices
        )
        return encoding


