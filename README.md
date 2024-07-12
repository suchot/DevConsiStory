---

---

# DevConsiStory
This repository is an experimental implementation of [Consistory](https://arxiv.org/abs/2402.03286). **This is an unofficial implementation based on SDXL.**  

will release the experimental code,  which get the same results like the following.

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

## TODO

- [x] Subject Driven Self-Attention
  - [ ] Vanilla Query Features
  - [x] Subject Localization
  - [x] Self-Attention Dropout
- [x] Feature injection
- [ ] Anchor images & reusable subjects

