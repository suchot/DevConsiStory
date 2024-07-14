import os
import torch
import yaml

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from consistory_sdxl_pipeline import ConsistoryXLPipeline, main
from utils import mask_with_otsu_pytorch, extract_details, concept_pprint
from dift_sd import SDFeaturizer

seed = 864
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

consistory_pipe = ConsistoryXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                            torch_dtype=torch.float16, use_safetensors=True).to(device)
sdxl_pipe = StableDiffusionXLPipeline(**consistory_pipe.components)
dift = SDFeaturizer("stabilityai/stable-diffusion-2-1") # have tested sdxl, not well, have bugs

consistory_pipe.enable_xformers_memory_efficient_attention()
consistory_pipe.enable_vae_slicing()
consistory_pipe.enable_vae_tiling()

with open('test_consistory_prompts.yaml', 'r') as file:
    data = yaml.safe_load(file)

details = []
pipe_encode, pipe_decode = consistory_pipe.tokenizer.encode, consistory_pipe.tokenizer.decode
extract_details(data, details, pipe_encode, pipe_decode)
del pipe_encode, pipe_decode
consistory_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.4)
sdxl_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.4)

for detail in details:
    concept_pprint(detail)
    batch_images = main(detail,consistory_pipe, sdxl_pipe, dift, seed)
    for idx, img in enumerate(batch_images):
        img.save(f"{result_dir}/{detail['index']}_{detail['concept_token']}_{detail['token_positions']}_{idx}.png")
