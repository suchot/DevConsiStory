from __future__ import annotations

import abc
import cv2
import gc

from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from IPython.display import display
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw, ImageFont
from diffusers.models.attention import Attention
from utils import mask_with_otsu_pytorch

from torchvision.transforms import PILToTensor
import torch.nn as nn
import cupy as cnp
from copy import deepcopy

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = False) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img

class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        self_subject_attention_mask = None
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if self.place_in_unet in ("mid", "up") and not is_cross:
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, dim = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
                
            if self.controller.cur_step != 0:
                key = key.reshape(batch_size * sequence_length, dim).unsqueeze(0).repeat(batch_size, 1, 1)
                value = value.reshape(batch_size * sequence_length, dim).unsqueeze(0).repeat(batch_size, 1, 1)
                
                query = attn.head_to_batch_dim(query) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                key = attn.head_to_batch_dim(key) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                value = attn.head_to_batch_dim(value) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                head_size = attn.heads
                dropout = 0.5
                mask_size = int(sequence_length ** 0.5)
                batch_masks = []
                midpoint = batch_size // 2
                mask_one = torch.ones((query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                mask_zero = torch.zeros((query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                self_subject_attention_mask = _aggregate_attention_ostu_mask(self.controller, res=32, batch_size=batch_size, from_where=("up", "down","mid"), is_cross=True) 
                injection_mask = [[]for i in range(batch_size)]
                for batch_index in range(batch_size):
                    masks = []
                    for idx, sdsa_mask in enumerate(self_subject_attention_mask):       
                        if idx == batch_index:                            
                            sdsa_mask = mask_one
                        elif (idx < midpoint and batch_index >= midpoint) or (idx >= midpoint and batch_index < midpoint):
                            sdsa_mask = mask_zero
                        else:
                            sdsa_mask = mask_with_otsu_pytorch(sdsa_mask)
                            injection_mask[idx].append(sdsa_mask.clone())
                            sdsa_mask = sdsa_mask.unsqueeze(0).unsqueeze(0)
                            mask_size = int(sequence_length ** 0.5)                            
                            sdsa_mask = F.interpolate(sdsa_mask.to(query.device), size=(mask_size, mask_size), mode='bilinear', align_corners=False)
                            sdsa_mask = sdsa_mask.view(-1, sequence_length)
                            sdsa_mask = sdsa_mask.unsqueeze(1).repeat(1, 1, sequence_length, 1)
                            sdsa_mask = sdsa_mask.squeeze(0).squeeze(0)
                            # drop out mask(if pixel value is droped out, it is 0)
                            sdsa_mask = F.dropout(sdsa_mask, p=dropout, training=True)
                        masks.append(sdsa_mask)
                    masks = torch.cat(masks, dim=1)          
                    # print("masks", batch_index, masks.shape)
                    masks = masks.unsqueeze(0).repeat(head_size, 1, 1)
                    batch_masks.append(masks.detach())
                attention_mask = torch.log(torch.cat(batch_masks, dim=0))
            else:
                query = attn.head_to_batch_dim(query) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                key = attn.head_to_batch_dim(key) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                value = attn.head_to_batch_dim(value) # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        else:
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        
        if is_cross and self.place_in_unet in ("mid","down"):
            # one line change
            self.controller(attention_probs.detach(), is_cross, self.place_in_unet)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self_subject_attention_mask is not None and self.controller.dift_feature is not None and sequence_length == 32**2:
            injection_alpha = 0.8
            threshold = 0.3
            ft = self.controller.dift_feature
            # For using big feature map
            ft_width, ft_height=32, 32
            # ft = nn.Upsample(size=(ft_height, ft_width), mode='bilinear')(ft)
            _hidden_states = hidden_states.detach().clone()
            width, height = int(sequence_length**0.5), int(sequence_length**0.5)
            _hidden_states = _hidden_states.permute(0, 2, 1).view(batch_size, dim, height, width)
            raw_hidden_states = _hidden_states.clone()
            index_list = [0,1,2,0,1,2]
            for batch_index in range(midpoint):
                select_index = index_list[batch_index]
                reverse_select_index =index_list[batch_index + 1:batch_index+midpoint] 
                src_ft = ft[select_index].unsqueeze(0)
                trg_ft = ft[reverse_select_index] # N, C, H, W
                num_channel = ft.size(1)
                trg_vec = trg_ft.view(midpoint-1, num_channel, -1)
                trg_vec = F.normalize(trg_vec, dim=2) # N, C, HW
                _injection_mask = mask_with_otsu_pytorch(self_subject_attention_mask[select_index])
                # _injection_mask = _injection_mask.unsqueeze(0).unsqueeze(0)
                # _injection_mask = F.interpolate(_injection_mask.to(query.device), size=(height, width), mode='bilinear', align_corners=False)
                # _injection_mask = _injection_mask.squeeze(0).squeeze(0)
                y_coords, x_coords = torch.meshgrid(torch.arange(height).to(query.device), torch.arange(width).to(query.device), indexing='ij')
                y_coords = y_coords.flatten()
                x_coords = x_coords.flatten()
                
                valid_positions = _injection_mask[y_coords, x_coords] != 0
                y_coords = y_coords[valid_positions]
                x_coords = x_coords[valid_positions]

                if len(y_coords) == 0: 
                    continue

                y_factors = (ft_height * y_coords / height).long()
                x_factors = (ft_width * x_coords / width).long()
                
                src_vecs = src_ft[0, :, y_factors, x_factors].view(len(y_coords), num_channel)
                src_vecs = F.normalize(src_vecs, dim=1)
                # TODO: looks like it should be calculated before inference.
                cos_maps = torch.einsum('ij,ajk->iak', src_vecs, trg_vec).view(len(y_coords), len(reverse_select_index), ft_height, ft_width)
                # cos_maps = torch.matmul(src_vecs.unsqueeze(1), trg_vec).view(len(y_coords), len(reverse_select_index), ft_height, ft_width)
                
                max_values, max_indices = cos_maps.view(len(y_coords), len(reverse_select_index), -1).max(dim=2)
                
                max_y_coords = (max_indices // ft_width).long()
                max_x_coords = (max_indices % ft_width).long()
                
                max_coords = torch.stack((max_y_coords, max_x_coords), dim=-1)
                valid_max_coords = max_values >= threshold
                
                best_indices = max_values.argmax(dim=1)
                
                for idx in range(len(y_coords)):
                    if valid_max_coords[idx].any():
                        best_index = best_indices[idx].item()
                        best_y, best_x = max_coords[idx, best_index]
                        best_y = best_y * height // ft_height
                        best_x = best_x * width // ft_width
                        trg_index = reverse_select_index[best_index]
                        curr_y, curr_x = y_coords[idx], x_coords[idx]
                        _hidden_states[select_index, :, curr_y, curr_x] = raw_hidden_states[trg_index, :, best_y, best_x]
                        _hidden_states[select_index + midpoint, :, curr_y, curr_x] = raw_hidden_states[trg_index + midpoint, :, best_y, best_x]
                    else:
                        curr_y, curr_x = y_coords[idx], x_coords[idx]
                        _hidden_states[select_index, :, curr_y, curr_x] = raw_hidden_states[select_index, :, curr_y, curr_x]
                        _hidden_states[select_index + midpoint, :, curr_y, curr_x] = raw_hidden_states[select_index + midpoint, :, curr_y, curr_x]

            _hidden_states = _hidden_states.view(batch_size, dim, -1).permute(0, 2, 1)
            hidden_states = _hidden_states * injection_alpha + hidden_states * (1 - injection_alpha)
        
            del attention_mask
        # linear proj
        
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def create_controller(
    prompts: List[str], cross_attention_kwargs: Dict, num_inference_steps: int, tokenizer, device, attn_res
) -> AttentionControl:
    edit_type = cross_attention_kwargs.get("edit_type", None)
    local_blend_words = cross_attention_kwargs.get("local_blend_words", None)
    equalizer_words = cross_attention_kwargs.get("equalizer_words", None)
    equalizer_strengths = cross_attention_kwargs.get("equalizer_strengths", None)
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)

    # only replace
    if edit_type == "replace" and local_blend_words is None:
        return AttentionReplace(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # replace + localblend
    if edit_type == "replace" and local_blend_words is not None:
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        return AttentionReplace(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, lb, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # only refine
    if edit_type == "refine" and local_blend_words is None:
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # refine + localblend
    if edit_type == "refine" and local_blend_words is not None:
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, lb, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # only reweight
    if edit_type == "reweight" and local_blend_words is None:
        assert (
            equalizer_words is not None and equalizer_strengths is not None
        ), "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(
            equalizer_strengths
        ), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
        )

    # reweight and localblend
    if edit_type == "reweight" and local_blend_words:
        assert (
            equalizer_words is not None and equalizer_strengths is not None
        ), "To use reweight edit, please specify equalizer_words and equalizer_strengths."
        assert len(equalizer_words) == len(
            equalizer_strengths
        ), "equalizer_words and equalizer_strengths must be of same length."
        equalizer = get_equalizer(prompts[1], equalizer_words, equalizer_strengths, tokenizer=tokenizer)
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        return AttentionReweight(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            tokenizer=tokenizer,
            device=device,
            equalizer=equalizer,
            attn_res=attn_res,
            local_blend=lb,
        )

    raise ValueError(f"Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.")

@torch.no_grad()
def _up_sample_attn(x, fix_hw, method='bicubic'):
    # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
    # x shape: (heads, height * width, tokens)
    """
    Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
    Diffusion model.

    Args:
        x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
        value (`torch.Tensor`): the value tensor.
        method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

    Returns:
        `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
    """
    
    h_fix = w_fix = fix_hw
    heads, seq, tokens = x.shape
    x = x.permute(0, 2, 1)
    h,w  = int(seq**0.5), int(seq**0.5)
    x = x.view(heads, tokens, h, w)
    upsample_x = F.interpolate(x, size=(h_fix, w_fix), mode='bicubic')
    upsample_x = upsample_x.view(heads,tokens, -1)
    upsample_x = upsample_x.permute(0, 2, 1)
    return upsample_x
    

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, attn_res=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.attn_res = attn_res

class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i] 
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=None):
        super(AttentionStore, self).__init__(attn_res)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
class AttentionConsiStore(AttentionStore):
    def __init__(self, attn_res=None, token_positions=-1):
        super(AttentionConsiStore, self).__init__(attn_res)
        self.token_positions = token_positions if token_positions <=76 else 76
            
    
class LocalBlend:
    def __call__(self, x_t, attention_store):
        # note that this code works on the latent level!
        k = 1
        # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]  # These are the numbers because we want to take layers that are 256 x 256, I think this can be changed to something smarter...like, get all attentions where thesecond dim is self.attn_res[0] * self.attn_res[1] in up and down cross.
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] +  attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, self.attn_res[0], self.attn_res[1], self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1) # since alpha_layers is all 0s except where we edit, the product zeroes out all but what we change. Then, the sum adds the values of the original and what we edit. Then, we average across dim=1, which is the number of layers.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)

        mask = mask[:1] + mask[1:]
        mask = mask.to(torch.float16)

        x_t = x_t[:1] + mask * (x_t - x_t[:1]) # x_t[:1] is the original image. mask*(x_t - x_t[:1]) zeroes out the original image and removes the difference between the original and each image we are generating (mostly just one). Then, it applies the mask on the image. That is, it's only keeping the cells we want to generate.
        return x_t

    def __init__(
        self, prompts: List[str], words: [List[List[str]]], tokenizer, device, threshold=0.3, attn_res=None
    ):
        self.max_num_words = 77
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device) # a one-hot vector where the 1s are the words we modify (source and target)
        self.threshold = threshold


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= self.attn_res[0]**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words
                    + (1 - alpha_words) * attn_replace
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device,
        attn_res=None,
    ):
        super(AttentionControlEdit, self).__init__(attn_res=attn_res)
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, self.tokenizer
        ).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
    ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None
    ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
    ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller


### util functions for all Edits
def update_alpha_time_word(
    alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor] = None
):
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts, num_steps, cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]], tokenizer, max_num_words=77
):
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


### util functions for LocalBlend and ReplacementEdit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


### util functions for ReplacementEdit
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")
    words_y = y.split(" ")
    if len(words_x) != len(words_y):
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length"
            f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    # return torch.from_numpy(mapper).float()
    return torch.from_numpy(mapper).to(torch.float16)


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


### util functions for ReweightEdit
def get_equalizer(
    text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer
):
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = torch.FloatTensor(values[i])
    return equalizer


### util functions for RefinementEdit
class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)

# Normalize the last two dimensions
def normalize_last_two_dims(tensor):
    # Min and max values along the last two dimensions for each channel
    min_val = tensor.amin(dim=(-1, -2), keepdim=True)
    max_val = tensor.amax(dim=(-1, -2), keepdim=True)
    
    # Normalize the tensor
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-9)  # Adding epsilon to avoid division by zero
    
    return normalized_tensor

def _aggregate_attention_ostu_mask(attention_store: AttentionStore,
                        res: int,
                        batch_size: int,
                        from_where: List[str],
                        is_cross: bool,
                        ) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    token_pos = attention_store.token_positions
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if (item is not None) and item.shape[1] == num_pixels:
                cross_maps = item.reshape(batch_size, -1, res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]
    out = out[:,:,:,token_pos]
    norm_out = normalize_last_two_dims(out)
    norm_out = torch.unbind(norm_out, dim=0)
    norm_out = list(norm_out)  # uncond + cond

    return norm_out

def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        prompts: List[str],
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()
    
def show_cross_attention(attention_store: AttentionStore, tokenizer, prompts: list[str], res: int, from_where: List[str], select: int = 0, display: bool = False):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store=attention_store, res=res, from_where=from_where, is_cross=True, select=select)
    
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = (image - image.min())/(image.max()-image.min())
        image = 255 * image
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    if display:
        view_images(np.stack(images, axis=0), display_image=display)
    else:
        return view_images(np.stack(images, axis=0), display_image=display)
    