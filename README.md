# PROTES


## Description

Method **PROTES** (**PR**obability **O**ptimizer with **TE**nsor **S**ampling) for derivative-free optimization of the multidimensional arrays and  discretized multivariate functions based on the tensor train (TT) format.


## Installation

To use this package, please install manually first the [python](https://www.python.org) programming language of the version >= 3.8, then, the package can be installed via pip: `pip install protes`.

> Required python packages ["jax[cpu]"](https://github.com/google/jax) (0.4.6+), [matplotlib](https://matplotlib.org/) (3.7.0+), [numpy](https://numpy.org) (1.22+), [optax](https://github.com/deepmind/optax) (0.1.5+) and [seaborn](https://seaborn.pydata.org/) (1.22+) will be automatically installed during the installation of the main software product.

To run scripts from the `calc` folder (numerical experiments for various benchmarks and comparison with various baselines), please install also additional dependencies as `pip install -r requirements_calc.txt`.


## Usage

Let's say we want to find the minimum of a `d = 10` dimensional array (tensor) that has `n = 5` elements (indices) in each mode. Let an arbitrary array element `y` related to the d-dimensional multi-index `i` is defined by the function `y = f(i) = sum(i)`. In this case, the optimizer may be launched in the following way:

```python
import jax.numpy as jnp
from protes import protes
f_batch = lambda I: jnp.sum(I, axis=1)
i_opt, y_opt = protes(f=f_batch, d=10, n=5, m=5.E+3, log=True)
```

The function `f_batch` takes a set of multi-indices `I` (jax array having a size `samples x d`; it can be easily converted to a numpy array if needed) and returns a list or numpy array of the corresponding array values; `m` is is the computational budget (the allowed number of requested array elements). Returned values `i_opt` (jax array) and `y_opt` (float) are the found multi-index and the value in this multi-index for the approximate optimum, respectively.

Note that the code runs orders of magnitude faster if the array's mode size (`n`) is constant, as shown in the example above. If you need to optimize an array with discriminating mode sizes, then you should use the slow `protes_general` method (i.e., `from protes import protes_general`). In this case, instead of two parameters `d` and `n`, one parameter `n` should be passed, which is a list of the length `d` corresponding to the mode sizes in each dimension.

Please, see also:

- The `demo` folder contains several examples of using the PROTES method for real tasks (a simple demo can be run in the console with a command `python demo/demo_func.py` and `python demo/demo_qubo.py`).

- The `calc` folder contains test runs of the PROTES method for various model problems (20 benchmarks) and compare it with other optimization methods (7 baselines). Additional dependencies from the `requirements_calc.txt` file should be installed to run the related script `calc/calc.py`.

- The `animation` folder contains a script `animate.py` for building an animation of the 2D optimization process.


## Parameters of the `protes` function

**Mandatory arguments**:

- `f` (function) - the target function `f(I)`, where input `I` is a 2D array of the shape `[samples, d]` (`d` is a number of dimensions of the function's input and `samples` is a batch size of requested multi-indices). The function should return 1D list or numpy array or jax array of the length equals to `samples` (the values of the target function for all provided multi-indices).
- `d` (int) - number of array dimensions.
- `n` (int) - mode size for each array's dimension.
- `m` (int) - the number of allowed requests to the objective function.

**Optional arguments**:

- `k` (int) - the batch size for optimization (the default value is `100`).
- `k_top` (int) - number of selected candidates in the batch (it should be `< k`; the default value is `10`).
- `k_gd` (int) - number of gradient lifting iterations for each batch (the default value is `1`).
- `lr` (float): learning rate for gradient lifting iterations (the default value is `5.E-2`).
- `r` (int): TT-rank of the constructed probability TT-tensor (the default value is `5`).
- `seed` (int): parameter for jax random generator (the default value is `0`).
- `is_max` (bool): if flag is set, then maximization rather than minimization will be performed.
- `log` (bool): if flag is set, then the information about the progress of the algorithm will be printed after each improvement of the optimization result and at the end of the algorithm's work.
- `log_ind` (bool): if flag is set and `log` is True, then the current optimal multi-index will be printed every step.
- `info` (dict): optional dictionary, which will be filled with reference information about the process of the algorithm operation.
- `P` (list): optional initial probability tensor in the TT-format. If this parameter is not set, then a random initial TT-tensor will be generated. Note that this tensor will be changed inplace.


## Notes

- You can use the outer cache for the values requested by the optimizer (that is, for each requested batch, check if any of the multi-indices have already been calculated), this can in some cases reduce the number of requests to the objective function.

- For a number of tasks, performance can be improved by switching to increased precision in the representation of numbers in `jax`; for this, at the beginning of the script, you should specify the code:
    ```python
    from jax.config import config
    config.update('jax_enable_x64', True)
    ```

- If there is a GPU, the jax optimizer code will be automatically executed on it, however, the current version of the code works better on the CPU. To use CPU, you should specify the following code at the beginning of the executable script:
    ```python
    import os
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    ```


## Authors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{batsheva2023protes,
    author    = {Batsheva, Anastasia and Chertkov, Andrei and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {PROTES: Probabilistic Optimization with Tensor Sampling},
    journal   = {arXiv preprint arXiv:2301.12162},
    url       = {https://arxiv.org/pdf/2301.12162.pdf}
}
```

> ✭ 🚂 The stars that you give to **PROTES**, motivate us to develop faster and add new interesting features to the code 😃
