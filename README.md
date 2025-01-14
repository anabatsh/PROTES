# PROTES


## Description

Method **PROTES** (**PR**obabilistic **O**ptimizer with **TE**nsor **S**ampling) for derivative-free optimization of the multidimensional arrays and discretized multivariate functions based on the tensor train (TT) format.


## Installation

To use this package, please install manually first the [python](https://www.python.org) programming language of the version `3.8+`, then, the package can be installed via pip:
```bash
pip install protes==0.3.11
```

> To ensure version stability, we recommend working in a virtual environment, as described in the `workflow.md`. Also note that `requirements.txt` contains `jax[cpu]`; if you need the GPU version of the `jax`, please install it yourself.


## Usage

Let's say we want to find the minimum of a `d = 10` dimensional array (tensor) that has `n = 5` elements (indices) in each mode. Let an arbitrary tensor element `y` related to the d-dimensional multi-index `i` is defined by the function `y = f(i) = sum(i)`. In this case, the optimizer may be launched in the following way:

```python
import numpy as np
from protes import protes
f_batch = lambda I: np.sum(I, axis=1)
i_opt, y_opt = protes(f=f_batch, d=10, n=5, m=5.E+3, log=True)
```

The function `f_batch` takes a set of multi-indices `I` (`jax` array having a size `samples x d`) and returns a list (or `jax` array or `numpy` array) of the corresponding tensor values; `m` is is the computational budget (the allowed number of requested tensor elements). Returned values `i_opt` (`jax` array) and `y_opt` (float) are the found multi-index and the value in this multi-index for the approximate optimum (in our case, minimum, since the flag `is_max` is not set) of the target tensor, respectively.

> The input `I` can be easily converted to a `numpy` array if needed (i.e., `import numpy as np; I = np.array(I)`), so you you can use the ordinary `numpy` inside the taget function and do not use the `jax` at all.

> Note that the code runs orders of magnitude faster if the tensor's mode size (`n`) is the same for all modes, as was in the example above. If you need to optimize a tensor with discriminating mode sizes, then you should use the slow `protes_general` function (i.e., `from protes import protes_general`). In this case, instead of two parameters `d` and `n`, one parameter `n` should be passed, which is a list of the length `d` corresponding to the mode sizes in each dimension (all other parameters for the function `protes_general` are the same as for the function `protes`, which are detailed in the next section).

Please, see also the `demo` folder, which contains several examples of using the `PROTES` method for real problems (a simple demo can be run in the console with a command `python demo/demo_func.py` and `python demo/demo_qubo.py`).


## Parameters of the `protes` function

**Mandatory arguments**:

- `f` (function) - the target function `f(I)`, where input `I` is a 2D jax array of the shape `[samples, d]` (`d` is a number of dimensions of the function's input and `samples` is a batch size of requested multi-indices). The function should return 1D list or jax array or numpy array of the length equals to `samples` (the values of the target function for all provided multi-indices, i.e., the values of the optimized tensor).
    > If the function returns `None`, the optimizer will immediately terminate normally (and the currently found approximation to the optimum will be returned). If the function returns an empty list or an empty array, the current iteration will be skipped and the function will be called again with a new batch of samples on the next iteration.
- `d` (int) - number of tensor dimensions.
    > For the slow function `protes_general` this argument is missing (instead, the dimension is determined by the length of argument `n`).
- `n` (int) - mode size for each tensor's dimension.
    > If the dimensions differ for different modes, then the slow function `protes_general` should be used, in which case this argument should be a list.

**Optional arguments**:

- `m` (int) - the number of allowed requests to the objective function (the default value is `None`). If this parameter is not set, then the optimization process will continue until the objective function returns `None` instead of a list of values.
- `k` (int) - the batch size for optimization (the default value is `100`).
- `k_top` (int) - number of selected candidates in the batch (it should be `< k`; the default value is `10`).
- `k_gd` (int) - number of gradient lifting iterations for each batch (the default value is `1`. Please note that this value ensures the maximum performance of the method, however, for a number of problems, a more accurate result is obtained by increasing this parameter, for example to `100`).
- `lr` (float): learning rate for gradient lifting iterations (the default value is `5.E-2`. Please note that this value must be correlated with the parameter `k_gd`).
- `r` (int): TT-rank of the constructed probability TT-tensor (the default value is `5`. Please note that we have not yet found problems where changing this value would lead to an improvement in the result).
- `seed` (int): parameter for jax random generator (the default value is `0`).
- `is_max` (bool): if flag is set, then maximization rather than minimization will be performed.
- `log` (bool): if flag is set, then the information about the progress of the algorithm will be printed after each improvement of the optimization result and at the end of the algorithm's work. Not that this argument may be also a function, in this case it will be used instead of ordinary `print` (i.e., `log(text) will be launched`).
- `info` (dict): optional dictionary, which will be filled with reference information about the process of the algorithm operation.
- `P` (list): optional initial probability tensor in the TT-format (represented as a list of jax arrays, where all non-edge TT-cores are merged into one array; see the function `_generate_initial` in `protes.py` for details). If this parameter is not set, then a random initial TT-tensor will be generated. Note that this tensor will be changed inplace.
    > If the slow function `protes_general` is used, this parameter must be (if specified) a list of ordinary 3D TT-cores of the length `d`.

**Other arguments**:

there are also a few more arguments (not documented) that we use for special applications.


## Notes

- You can use the outer cache for the values requested by the optimizer (that is, for each requested batch, check if any of the multi-indices have already been calculated), this can in some cases reduce the number of requests to the objective function. In this case, it is convenient not to set limits on the number of requests (parameter `m`), but instead, when the budget is exhausted (including the cache), return the `None` value from the target function (`f`), in which case the optimization algorithm will be automatically terminated.

- For a number of tasks, performance can be improved by switching to increased precision in the representation of numbers in `jax`; for this, at the beginning of the script, you should specify the code:
    ```python
    import jax
    jax.config.update('jax_enable_x64', True)
    ```

- If there is a GPU, the `jax` optimizer code will be automatically executed on it, however, the current version of the code works better on the CPU. To use CPU on a device with available GPU, you should specify the following code at the beginning of the executable script:
    ```python
    import jax
    import logging
    # To remove jax warnings about GPU:
    logger = logging.getLogger('jax._src.xla_bridge')
    logger.setLevel(logging.ERROR)
    # To use CPU instead of GPU:
    jax.config.update('jax_platform_name', 'cpu')
    jax.default_device(jax.devices('cpu')[0]);
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
    title     = {{PROTES}: {P}robabilistic optimization with tensor sampling},
    journal   = {Advances in Neural Information Processing Systems},
    url       = {https://arxiv.org/pdf/2301.12162.pdf}
}
```

> ✭ 🚂 The stars that you give to **PROTES**, motivate us to develop faster and add new interesting features to the code 😃


## License

If you plan to use this software for commercial purposes, please contact [Ivan Oseledets](https://github.com/oseledets) for details. There are no restrictions on the use for scientific purposes when citing 😊