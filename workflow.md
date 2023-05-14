# workflow

> Workflow instructions for `protes` developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name protes python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate protes
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab
    ```

5. Install protes:
    ```bash
    python setup.py install
    ```

6. Install additional packages for `calc`:
    ```bash
    pip install -r requirements_calc.txt
    ```

7. Reinstall protes (after updates of the code):
    ```bash
    clear && pip uninstall protes -y && python setup.py install
    ```

8. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name protes --all -y
    ```


## How to update the package version

1. Run the demo script for Ackley function and check the result:
    ```bash
    clear && python demo/demo_func.py
    ```

2. Run the demo script for QUBO problem and check the result:
    ```bash
    clear && python demo/demo_qubo.py
    ```

3. Update version (like `0.2.X`) in the file `protes/__init__.py`

    > For breaking changes we should increase the major index (`2`), for non-breaking changes we should increase the minor index (`X`)

4. Do commit `Update version (0.2.X)` and push

5. Upload new version to `pypi` (login: AndreiChertkov)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

6. Reinstall and check that installed version is new
    ```bash
    pip install --no-cache-dir --upgrade protes
    ```
