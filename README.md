# AE-KAN: Autoencoders using KAN 

AE-KAN introduces a autoencoder model using KAN for high-performance image classification and leveraging polynomial transformations for enhanced feature detection.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction
Deep learning models have revolutionized various domains, with Multi-Layer Perceptrons (MLPs) being a cornerstone for tasks like image classification. However, recent advancements have introduced Kolmogorov-Arnold Networks (KANs) as promising alternatives to MLPs, leveraging activation functions placed on edges rather than nodes. This structural shift aligns KANs closely with the Kolmogorov-Arnold representation theorem, potentially enhancing both model accuracy and interpretability. In this study, we explore the efficacy of KANs in the context of data representation via autoencoders, comparing their performance with traditional Convolutional Neural Networks (CNNs) on the MNIST dataset. Our results demonstrate that KAN-based autoencoders not only achieve competitive performance in terms of reconstruction accuracy but also offer insights into the interpretability of learned representations, thereby suggesting their viability as effective tools in data analysis tasks. Inspired by a range of sources, this first implementation in KAN in torchkan.py achieves over 97% accuracy with an evaluation time of 0.6 seconds. The quantized model further reduces this to under 0.55 seconds on the MNIST dataset within 8 epochs, utilizing an Nvidia RTX 4090 on Ubuntu 22.04.

We are conducting large-scale analysis to investigate how AE-KANs can be made more efficient.

Note: As the model is still being researched, further explorations into its full potential are ongoing. Contributions, questions, and critiques are welcome. I appreciate constructive feedback and contributions. Merge requests will be processed promptly, with a clear outline of the issue, the solution, and its effectiveness.

Note: The PyPI pipeline is currently deprecated and will be stabilized following the release of Version 1.

## Model Architecture

Convolutional Feature Extraction: The model begins with two convolutional layers, each paired with ReLU activation and max-pooling. The first layer employs 16 filters of size 3x3, while the second increases the feature maps to 32 channels.

Polynomial Feature Transformation: Post feature extraction, the model applies polynomial transformations up to the n'th order to the flattened convolutional outputs, enhancing its ability to discern non-linear relationships.

How Monomials Work: In the context of this model, monomials are polynomial powers of the input features. By computing monomials up to a specified order, the model can capture non-linear interactions between the features, potentially leading to richer and more informative representations for downstream tasks.

For a given input image, the monomials of its flattened pixel values are computed, which are then used to adjust the output of linear layers before activation. This approach introduces an additional dimension of feature interaction, allowing the network to learn more complex patterns in the data.

Note KANvolver uses polynomials which are distinct from the original KANs[1].

## Key Features

Polynomial Order: Utilizes Legendre polynomials up to a specific order for each input normalization, capturing nonlinear relationships more efficiently than simpler polynomial approximations.
Efficient Computations: By leveraging functools.lru_cache, the network avoids redundant computations, enhancing the forward pass's speed.
Activation Function: Employs the SiLU (Sigmoid Linear Unit) for improved performance in deeper networks due to its non-monotonic nature.
Layer Normalization: Stabilizes each layer's output using layer normalization, enhancing training stability and convergence speed.
Design and Initialization
Weight Initialization: Weights are initialized using the Kaiming uniform distribution, optimized for linear nonlinearity, ensuring a robust start for training.
Dynamic Weight and Normalization Management: Manages weights for base transformations and polynomial expansions dynamically, scaling with input features and polynomial order.
Advantages Over Splines (Pending Rigorous Empirical Testing)
Flexibility in High-Dimensional Spaces: Legendre polynomials offer a more systematic approach to capturing interactions in high-dimensional data compared to splines, which often require manual knot placement and struggle with dimensionality issues.
Analytical Efficiency: The caching and recurrence relations in Legendre polynomial computations minimize the computational overhead associated with spline evaluations, especially in high dimensions.
Generalization: The orthogonal properties of Legendre polynomials typically lead to better generalization in machine learning model fitting, avoiding common overfitting issues with higher-degree splines.
Performance Metrics
Accuracy: KAL_Net achieved a remarkable 97.8% accuracy on the MNIST dataset, showcasing its ability to handle complex patterns in image data.
Efficiency: The average forward pass takes only 500 microseconds, illustrating the computational efficiency brought by caching Legendre polynomials and optimizing tensor operations in PyTorch.

## Prerequisites
Ensure the following are installed on your system:

* Python (version 3.9 or higher)
* CUDA Toolkit (compatible with your PyTorch installation's CUDA version)
* cuDNN (compatible with your installed CUDA Toolkit)

## Installation
Tested on MacOS and Linux.

### 1. Clone the Repository
Clone the torchkan repository and set up the project environment:

git clone https://github.com/1ssb/torchkan.git
cd torchkan
pip install -r requirements.txt

### 2. Configure CUDA Environment Variables if Unset:
```python
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
3. Configure Weights & Biases (wandb)
To monitor experiments and model performance with wandb:

## Cite this Project
If this project is used in your research or referenced for baseline results, please use the following BibTeX entry.
```bibtex
@misc{torchkan,
  author = {Moradi, Mohammadamin and Panahi, Shirin and Lai, Ying-Cheng},
  title = {Kolmogorov-Arnold Network (KAN) Autoencoders},
  year = {2024},
  publisher = {arxiv}
}
```

## Contributions
Contributions are welcome. Please raise issues as needed. Maintained solely by @AminMoradiXL.

## References
* [0] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
* [1] https://github.com/KindXiaoming/pykan
* [2] https://github.com/Blealtan/efficient-kan
