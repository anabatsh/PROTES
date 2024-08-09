# workflow

> Workflow instructions for `protes` developers.


## How to install the current local version

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org);

2. Create and activate a virtual environment:
    ```bash
    conda create --name PROTES python=3.10 -y && conda activate PROTES
    ```

3. Install special dependencies (for developers only):
    ```bash
    pip install jupyterlab twine
    ```

4. Install `protes` from the source:
    ```bash
    python setup.py install
    ```
    > You may also use the command `pip install --no-cache-dir protes` to install the current public version.

5. Reinstall `protes` from the source (after updates of the code):
    ```bash
    clear && pip uninstall protes -y && python setup.py install
    ```

6. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name PROTES --all -y
    ```


## How to update the package version

1. Reinstall the package from the source and run the demo scripts to check the correctness of the code:
    ```bash
    pip uninstall protes -y && python setup.py install && clear && python demo/demo_func.py && python demo/demo_qubo.py && python demo/demo_func_general.py
    ```

2. Update version (like `0.3.X`) in `protes/__init__.py` and `README.md` files, where `X` is a new subversion number

3. Do commit like `Update version (0.3.X)` and push

4. Upload new version to `pypi` (login: AndreiChertkov)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall the package from `pypi` and check that the installed version is new:
    ```bash
    pip uninstall protes -y && pip install --no-cache-dir --upgrade protes
    ```