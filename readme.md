# SDFlow

Official code for `Semantic Latent Decomposition with Normalizing Flows for Face Editing`,

> In recent years, navigating in the latent space of StyleGAN has achieved significant progress for face editing. However, these methods usually encounter challenges in complicated navigation due to the entanglement between different attributes in the latent space of StyleGAN. This paper proposes a novel framework, termed SDFlow, with a semantic decomposition in original latent space using continuous conditional normalizing flows. Specifically, SDFlow decomposes the original latent code into different irrelevant variables by jointly optimizing two components: (i) a semantic estimator produces semantic variables from input faces and (ii) a flow-based transformation module maps the latent code to semantic-irrelevant variable in Gaussian distribution, conditioned on the learned semantic variables. To eliminate the entanglement between variables, we further employ a disentangled learning strategy under a mutual information framework, thereby providing precise manipulation controls. Experimental results demonstrate that SDFlow outperforms existing face editing methods both qualitatively and quantitatively.

> ***Index terms*** : Face Editing, Disentangle Learning, Generative Adversarial Network


![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg?style=plastic) ![pytorch 1.11.0](https://img.shields.io/badge/Pytorch-1.11.0-green.svg?style=plastic) ![torchdiffeq 0.0.1](https://img.shields.io/badge/Torchdiffeq-0.0.1-green.svg?style=plastic) ![streamlit 1.26.0](https://img.shields.io/badge/streamlit-1.26.0-green.svg?style=plastic)


**The Training code, colab demo and pretrained models will be released soon.**


**Example results**

![1](assert/showcases.png)

**Framework**
![2](assert/framework.png)



## Demo


or run the demo locally. The web ui demo looks like this:
![img](assert/demo_720p.gif)


We additionally support

- (1) Align the faces to FFHQ template
- (2) Project the input images into $W+$ space for in-the-wild editing
- (3) Showing the facial attributes by the pretrained classifier.


## Citation

**We appreciate for any citations or stars if you found this work is helpful**


## Acknowledgments

This implementation builds upon the awesome work done by Zhizhong Huang et al. (**[AdaTrans](https://github.com/Hzzone/AdaTrans)**), Karras et al. (**[StyleGAN2](https://github.com/NVlabs/stylegan2)**) and Chen et al. (**[torchdiffeq](https://github.com/rtqichen/torchdiffeq)**).  They provides inspiration and examples that played a crucial role in the implements of this project.