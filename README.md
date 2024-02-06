# llm-wrapper

___
## About

`llm-wrapper` is a versatile and powerful library designed to streamline the process of querying large language models, offering a user-friendly experience.
The` llm-wrapper` package is designed to simplify interactions with the underlying models by providing the following features:

* **Simple and User-Friendly Interface**: The module offers an intuitive and easy-to-use interface, making it straightforward to work with the model.

* **Asynchronous Querying**: Requests to the model are processed asynchronously by default, ensuring efficient and non-blocking interactions.

* **Automatic Retrying Mechanism**: The module includes an automatic retrying mechanism, which helps handle transient errors and ensures that queries to the model are robust.

* **Error Handling and Management**: Errors that may occur during interactions with the model are handled and managed gracefully, providing informative error messages and potential recovery options.

* **Output Parsing**: The module simplifies the process of defining the model's output format as well as parsing and working with it, allowing you to easily extract the information you need.

___

## Documentation

Are you interested in using `llm-wrapper` in your project? Consult the [Official Documentation](# TODO: open-source)!

___

## Local Development

### Installation

We assume that you have python `3.10.*` installed on your machine. 
You can set it up using [pyenv](https://github.com/pyenv/pyenv#installationbrew) 
([How to install pyenv on MacOS](https://jordanthomasg.medium.com/python-development-on-macos-with-pyenv-2509c694a808)). To install `llm-wrapper` env locally:

* Activate your pyenv;
* Install Poetry via:

```bash
make install-poetry
```

* Install `llm-wrapper` dependencies with the command:

```bash
make install-env
```

* Now you can use this venv for development. You can activate it in your shell by running:

```bash
make activate-env # or simply, poetry shell
```

### Tests

In order to execute tests, run:

```bash
make tests
```

### Make a new release

When a new version of `llm-wrapper` is ready to be released, do the following operations:

1. **Merge to master** the dev branch in which the new version has been specified:
    1. In this branch, `version` under `[tool.poetry]` section in `pyproject.toml` should be updated, e.g `0.1.0`;
    2. Update the CHANGELOG, specifying the new release.

2. **Tag the new master** with the name of the newest version:
    1. e.g `v0.1.0`.

3. **Publish package to PyPI**:
    1. Go to _Actions_ → _Manual Publish To PyPI_;
    2. Select "master" as branch and click _Run workflow_;
    3. If successful, you will find the package under # TODO: open-source.

4. **Make a GitHub release**:
    1. Go to _Releases_ → _Draft a new release_;
    2. Select the recently created tag in _Choose a tag_ window;
    3. Copy/paste all the content present in the CHANGELOG under the version you are about to release;
    4. Upload `llm_wrapper-<NEW-VERSION>.whl` and `llm_wrapper-<NEW-VERSION>.tar.gz` as assets;
    5. Click `Publish release`.