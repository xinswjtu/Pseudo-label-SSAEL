# Pseudo-label-SSAEL

This is the impletement of ***[Pseudo-label assisted semi-supervised adversarial enhancement learning for intelligent fault diagnosis of industrial gearbox degradation](https://doi.org/10.1016/j.ymssp.2024.112108)*** .

## Usage

### Requirements

* Python: 3.8.16
* PyTorch: 2.0.0+cu118
* Torchvision: 0.15.0+cu118
* CUDA: 11.8
* CUDNN: 8700
* NumPy: 1.23.4
* PIL: 10.0.1

### Data preparation

The data utilized in this study is proprietary; however, anyone can prepare their own dataset by adhering to the following directory structure

```
./datasets/Gear/real_images/
├── test
│   ├── 0-0.jpg
│   ├── 1-0.jpg
...
│   ├── 100-1.jpg
...
├── train
│   ├── 0-0.jpg
│   ├── 1-0.jpg
...
│   ├── 200-1.jpg
...
└── valid
    ├── 0-0.jpg
    ├── 1-0.jpg
    ...
```

### Training

To run `train_sgan.py`

```bash
$ bash train.sh
```

OR：

```bash
$python train_sgan.py --n_train 30 --n_label 10 --add_supconloss True --temperature 0.3 --sup_wt 0.5 --df_aux_wt 0.5 --lr_g 0.0006 --lr_d 0.0001 --n_critic 2
```

## Citation

If you find our work is relevant to your research, please cite:

```
X. Chen, Z. Chen, L. Guo, W. Zhai, Pseudo-label assisted semi-supervised adversarial enhancement learning for fault diagnosis of gearbox degradation with limited data, Mechanical Systems and Signal Processing, 224 (2025) 112108.
```

## Acknowledgements

1. [[2109.02235\] Gradient Normalization for Generative Adversarial Networks](https://arxiv.org/abs/2109.02235) 
1. [[1606.01583\] Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583)
1. [[1606.03498\] Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
1. [[2004.11362\] Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
1. [[2002.05709\] A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
1. [Conditional Contrastive Domain Generalization for Fault Diagnosis | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9721021)

## Contact

If you have any questions, please do not hesitate to contact us：

* Xin Chen
* chenxinnpu@gmail.com
