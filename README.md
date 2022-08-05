# vmc_pde
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RehMoritz/vmc_pde/blob/main/vmc_pde_MinimalDemo.ipynb)


Code for the paper "Variational Monte Carlo Approach to Partial Differential Equations with Neural Networks" (https://arxiv.org/abs/2206.01927).

The code is localized in the folder "vmc_fluids".
The data is localized in "vmc_fluids/paper_plot/".


The code should run as is for the following package versions:

```
>>> import jax
>>> import flax
>>> import jaxlib
>>> jax.__version__
'0.2.18'
>>> flax.__version__
'0.3.6'
>>> jaxlib.__version__
'0.1.74'
```
A minimal working example can be found by clicking the Colab badge.
