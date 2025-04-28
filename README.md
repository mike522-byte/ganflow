
# GANFlow: Unsupervised Generative Adversarial Learning for Optical Flow

## Overview
**GANFlow** proposes an unsupervised optical flow learning framework that incorporates **Generative Adversarial Networks (GANs)** to enhance the realism of reconstructed images and improve flow prediction accuracy. Built upon the **UFlow** generator backbone, GANFlow introduces an image-based discriminator that guides the generator without requiring any ground-truth optical flow labels.

## Architecture
![image](https://github.com/user-attachments/assets/d4a966a0-f510-4451-af5b-3f9c0fe64d5c)
*Figure 1. Network architecture of our proposed GANFlow*

The architecture consists of:
- **Generator**: Predicts optical flow between two frames via a hierarchical feature extraction and warping mechanism.
- **Discriminator**: Takes real and reconstructed images, along with occlusion masks, to output a probability of authenticity, guiding the generator via adversarial training.

The generator and discriminator are trained alternately in an adversarial manner, where the generator is optimized using a combination of photometric consistency loss **\( \mathcal{L}_{census} \)**, smoothness loss **\( \mathcal{L}_{smooth} \)**, self-supervision loss **\( \mathcal{L}_{self} \)**, and adversarial generator loss **\( \mathcal{L}_{Gg} \)**, while the discriminator is optimized independently using the adversarial discriminator loss **\( \mathcal{L}_{Gd} \)**.


## Ablation Studies
![image](https://github.com/user-attachments/assets/38229fac-1b0b-4a1b-b000-a773a68c07ff)
*Figure 2. Different image discriminator architectures*

We explored four different discriminator architectures to assess how feature extraction and fusion impact adversarial learning. These variants are summarized in Figure 2:
- **Structure (A)**: Separate feature extraction for real and reconstructed images followed by independent probability predictions.
- **Structure (B)**: Concatenated real and reconstructed images at the input stage, enabling joint feature extraction.
- **Structure (C)**: Introduction of the second frame \( \mathbf{X}_{t+1} \) and use of a cost volume to guide the probability prediction.
- **Structure (D)**: Feature extraction for each input image followed by feature-level concatenation prior to probability prediction.

Results show **Structure (D)** consistently achieves the best performance across datasets. This structure facilitates richer interactions between features from real and reconstructed images.

### Occlusion Handling

Since occluded regions violate photometric assumptions, we investigated different occlusion handling methods. Directly masking occluded pixels with zeros led to suboptimal performance due to introducing unnatural boundaries. In contrast, adding **Gaussian noise** to occluded regions provided smoother gradients and more stable training, improving both flow accuracy and adversarial convergence.

### Discriminator Output Format

We compared two types of discriminator outputs:
- A single scalar value per image pair (global decision).
- A dense \( H \times W \) probability map (pixel-wise decision).

Our results indicate that using **pixel-wise outputs** yields superior performance, as it enables fine-grained spatial supervision. This allows the discriminator to focus on local inconsistencies, providing stronger learning signals to the generator.

### Image Reconstruction Strategy

Classical reconstruction methods warp the second frame using forward flow to reconstruct the first frame. We proposed a **novel reconstruction** technique: synthesizing the first frame by combining both forward and backward flows. This method compensates for inconsistencies between frames and mitigates bias introduced by single-direction warping, leading to improved photometric losses and better optical flow predictions.



## Training Results
We follow the standard practice in optical flow research, utilizing the **Flying Chairs**, **Sintel**, and **KITTI 2015** datasets for training and evaluation. Specifically, we first conduct pre-training on the Flying Chairs dataset, followed by fine-tuning separately on the Sintel and KITTI datasets.

GANFlow achieves state-of-the-art on train results compared to existing unsupervised methods:

*Table I. comparison to SOTA*

| Method | Sintel Clean (EPE) | Sintel Final (EPE) | KITTI 2015 (EPE) | KITTI 2015 (ER%) |
|:------:|:------------------:|:-----------------:|:----------------:|:---------------:|
| DSTFlow | 6.16 | 7.38 | 16.79 | 36.00 |
| OAFlow  | 4.03 | 5.95 | 8.88 | - |
| UFlow-test  | 3.01 | 4.09 | 2.84 | 9.39 |
| **GANFlow-test (ours)** | **2.94** | **3.82** | **2.76** | **9.67** |



## Contributions
This project is collaboratively conducted by **Shuaiqi Ren**, **Lim Ming Jie**, and **Guangming Wang**,  and is supervised by **Professor Hesheng Wang** from Shanghai Jiao Tong University.
