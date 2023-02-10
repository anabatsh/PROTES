# PROTES


## Description

Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization of the multidimensional arrays and  discretized multivariable functions based on the tensor train (TT) format.


## Installation

The package can be installed via pip: `pip install protes` (it requires the [Python](https://www.python.org) programming language of the version >= 3.6). The [jax](https://github.com/google/jax) and [optax](https://github.com/deepmind/optax) libraries should be manually installed for successful operation of jax version of the code. Alternatively, an equivalent pytorch version can be used (it is currently very slow), in this case, please, install manually [pytorch](https://pytorch.org/) library.


## Documentation and examples

Please see the documentation for function `protes` with a detailed description of all optimizer parameters. Examples are presented in the `demo` folder. A simple demo can be run in the console with a command `python demo/demo_func.py` (to run the pytorch version, please, specify the appropriate argument: `python demo/demo_func.py tor`).


## Authors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{batsheva2023protes,
    author    = {Batsheva, Anastasia and Chertkov, Andrei  and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {PROTES: Probabilistic Optimization with Tensor Sampling},
    journal   = {arXiv preprint arXiv:2301.12162},
    url       = {https://arxiv.org/pdf/2301.12162.pdf}
}
```
