# DevConsiStory
This repository is an experimental implementation of [Consistory](https://arxiv.org/abs/2402.03286). **This is an unofficial implementation based on SDXL.**  


 The experimental code ~~will~~ be published,  with the same results as below.

## My Experimental Code Qualitative Results

### Date:20240710 Results

#### w/o Feature Injection

| Key word  |                                                              |                                                              |                                                              |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| gentleman | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425099.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425090.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425075.png) |
| woman     | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425108.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425115.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101425122.png) |
| gentleman | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424288.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424281.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424272.png) |
| vase      | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424296.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424304.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424311.png) |
| dragon    | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424331.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424325.png) | ![](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407101424318.png) |

#### With Feature Injection

| Key word |                                                              |                                                              |                                                              |
| :------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|          |                                                              |                                                              |                                                              |
|          | ![122_vase_9_2_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407120935010.png) | ![122_vase_9_1_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407120936846.png) | ![122_vase_9_0_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407120937509.png) |
|          | ![94_unicorn_3_0_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121000085.png) | ![94_unicorn_3_1_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121000198.png) | ![94_unicorn_3_2_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121000147.png) |
|          | ![102_leprechaun_-1_0_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121001481.png) | ![102_leprechaun_-1_1_norm_mid_up_freeu_cfg5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121001774.png) | ![102_leprechaun_-1_2_norm_mid_up_freeu_cfg5_](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121002145.png) |
|          | ![103_werewolf_2_0_norm_mid_up_freeu_cfg5_v5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121006206.png) | ![103_werewolf_2_1_norm_mid_up_freeu_cfg5_v5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121006787.png) | ![103_werewolf_2_2_norm_mid_up_freeu_cfg5_v5](https://raw.githubusercontent.com/suchot/blog-pic-bed/master/202407121006414.png) |

## TODO:

- [x] Subject Driven Self-Attention
  - [ ] Vanilla Query Features
  - [x] Subject Localization
  - [x] Self-Attention Dropout
- [x] Feature injection
- [ ] Anchor images & reusable subjects


## Inference:
```python
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
```

## Env: 
  * A800
  * Driver Version: 470.161.03   
  * CUDA Version: 12.2

## Reference:

* RoyiRa/prompt-to-prompt-with-sdxl: https://github.com/RoyiRa/prompt-to-prompt-with-sdxl 


* kousw/experimental-consistory: https://github.com/kousw/experimental-consistory


