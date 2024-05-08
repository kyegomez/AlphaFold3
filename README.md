[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Python Package Template
A easy, reliable, fluid template for python packages complete with docs, testing suites, readme's, github workflows, linting and much much more


## Installation

You can install the package using pip

```bash
pip install -e .
```

# Usage
```python
print("hello world")

```



### Code Quality 🧹

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`

### Tests 🧪

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.

### Publish on PyPi 🚀

**Important**: Before publishing, edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.

```
poetry build
poetry publish
```

### CI/CD 🤖

We use [GitHub actions](https://github.com/features/actions) to automatically run tests and check code quality when a new PR is done on `main`.

On any pull request, we will check the code quality and tests.

When a new release is created, we will try to push the new code to PyPi. We use [`twine`](https://twine.readthedocs.io/en/stable/) to make our life easier. 

The **correct steps** to create a new realease are the following:
- edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.
- create a new [`tag`](https://git-scm.com/docs/git-tag) with the release name, e.g. `git tag v0.0.1 && git push origin v0.0.1` or from the GitHub UI.
- create a new release from GitHub UI

The CI will run when you create the new release.

# Docs
We use MK docs. This repo comes with the zeta docs. All the docs configurations are already here along with the readthedocs configs.



# License
MIT
