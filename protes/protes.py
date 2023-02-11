def protes(f, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, P=None, seed=42, info={}, i_ref=None, is_max=False, log=False, log_ind=False, mod='jax', device='cpu'):
    """Tensor optimization based on sampling from the probability TT-tensor.

    Method PROTES (PRobability Optimizer with TEnsor Sampling) for optimization
    of the multidimensional arrays and  discretized multivariable functions
    based on the tensor train (TT) format.

    Args:
        f (function): the target function "f(I)", where input "I" is a 2D
            array of the shape "[samples, d]" ("d" is a number of dimensions
            of the function's input and "samples" is a batch size of requested
            multi-indices). The function should return 1D array on the CPU or
            GPU of the length equals to "samples" (the values of the target
            function for all provided multi-indices).
        n (list of int): tensor size for each dimension.
        m (int): the number of allowed requests to the objective function.
        k (int): the batch size for optimization.
        k_top (int): number of selected candidates for all batches (< k).
        k_gd (int): number of GD iterations for each batch.
        lr (float): learning rate for GD.
        r (int): TT-rank of the constructed probability TT-tensor.
        P (list): optional initial probability tensor in the TT-format. It
            should be a list of TT-cores (3D arrays). If the parameter is not
            set, then a random initial TT-tensor will be generated.
        seed (int): parameter for random generator.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
        i_ref (list of int): optional multi-index, in which the values of the
            probabilistic tensor will be stored during iterations (the result
            will be available in the info dictionary in the 'p_ref_list' field).
            If this parameter is set, then the values of the probability tensor
            in the current optimum ("p_opt_ref_list") and in the current best
            sample from the batch ("p_top_ref_list") will also be saved.
        is_max (bool): if is True, then maximization will be performed.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed every step.
        log_ind (bool): if flag is set and "log" is True, then the current
            optimal multi-index will be printed every step.
        mod (str): the type of optimizer to use. Currently available values are:
            "jax" (jax code; default value) and "tor" (torch code).
        device (str): device for computations ('cpu' or 'cuda'). This parameter
            is used only if the "mod" is "tor" (jax itself selects the best
            available device).

    Returns:
        tuple: multi-index "i_opt" (list of the length "d") corresponding to
        the found optimum of the tensor and the related value "y_opt" (float).

    """
    if mod == 'jax':
        try:
            from .protes_jax import protes_jax
            from .protes_jax_fast import protes_jax_fast
        except Exception as e:
            msg = 'For "jax" version "jax" and "optax" packages are required.'
            raise ValueError(msg)

        if len(n) >= 3 and len(set(n)) == 1:
            return protes_jax_fast(f, n, m, k, k_top, k_gd, lr, r, P, seed,
                info, i_ref, is_max, log, log_ind, mod)
        else:
            return protes_jax(f, n, m, k, k_top, k_gd, lr, r, P, seed,
                info, i_ref, is_max, log, log_ind, mod)

    elif mod == 'tor':
        try:
            from .protes_tor import protes_tor
        except Exception as e:
            msg = 'For "tor" version "torch" package is required.'
            raise ValueError(msg)

        return protes_tor(f, n, m, k, k_top, k_gd, lr, r, P, seed, info,
            i_ref, is_max, log, log_ind, mod, device)

    else:
        raise ValueError('Invalid "mod" parameter.')
