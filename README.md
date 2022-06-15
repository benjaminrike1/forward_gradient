# forward_gradient

This repository contains our implementation and experiements with using _forward gradients_ for optimizing neural networks as part of a project in the course CS-439 Optimization for Machine Learning at EPFL. The _Forward gradient_ is an unbiased estimator of the real gradient suggested by Baydian et al. in the paper [_Gradients without backpropagation_](https://arxiv.org/abs/2202.08587). Information about our experiments can be found in report.pdf, but we include the abstract here for simplicity:

__Abstract:__

Training deep natural networks is a resource- and time-consuming process. The networks are usually trained using backpropagation, a case of reverse-mode automatic differentiation that differentiates functions using a forward and backward pass. Baydin et al. propose using forward-mode automatic differentiation to eliminate the backward pass and calculate an unbiased estimator of the actual gradient, the _forward gradient_. The authors' results show that the forward gradient trains twice as fast as backpropagation while reaching the same or better performance. In this paper, we investigate the authors' results, test whether they scale to more complex tasks, and explore the performance of the Adam optimizer and SGD with Nesterov momentum on the forward gradient. We find that the original paper uses too small learning rates for backpropagation, leading to unfair comparisons. Furthermore, our results indicate that backpropagation outperforms forward gradients and that forward gradient does not scale to increased complexity, regardless of the optimizer used.


__How to reproduce the experiements:__

The code used for reproducing experiements can be found in the files: `MNIST.ipynb`and `CIFAR.ipynb`. The notebooks are made to be run on GPU in Google Colab, but can with minor modifications be run locally. The notebooks require `functorch` to run the forward mode autodifferentiation used in the forward gradients and this must be installed if run locally. As a disclaimer, we encountered trouble running `functorch`on Windows and Mac with M1 chip, and therefore reccomend using Google Colab.

__Implementation:__

The implementation of the forward gradient optimization algorithms can be found in `optimizers.py`. We have implemented SGD and SGD with Nesterov momentum in the ForwardSGD class and Adam in the ForwardAdam class. The models used can be found in `models.py`. There are three nets, 1) Net, the vanilla neural network used on MNIST, 2) ConvNet, the convolutional net used on MNIST, 3) CifarNet, the convolutional net used on CIFAR-10. 

__Contributors:__

- Benjamin Rike
- Olav Førland
