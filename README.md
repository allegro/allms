# llm-wrapper

___
## About

llm-wrapper is a versatile and powerful library designed to streamline the process of querying Large Language Models
(LLMs) 🤖💬

Developed by the Allegro engineers, llm-wrapper is based on popular libraries like transformers, pydantic, and langchain. It takes care 
of the boring boiler-plate code you write around your LLM applications, quickly enabling you to prototype ideas, and eventually helping you to scale up 
for production use-cases!

Among the llm-wrapper most notable features, you will find:

* **😊 Simple and User-Friendly Interface**: The module offers an intuitive and easy-to-use interface, making it straightforward to work with the model.

* **🔀 Asynchronous Querying**: Requests to the model are processed asynchronously by default, ensuring efficient and non-blocking interactions.

* **🔄 Automatic Retrying Mechanism** : The module includes an automatic retrying mechanism, which helps handle transient errors and ensures that queries to the model are robust.

* **🛠️ Error Handling and Management**: Errors that may occur during interactions with the model are handled and managed gracefully, providing informative error messages and potential recovery options.

* **⚙️ Output Parsing**: The module simplifies the process of defining the model's output format as well as parsing and working with it, allowing you to easily extract the information you need.

___

## Documentation

Full documentation available at **[llm-wrapper.allegro.tech](https://llm-wrapper.allegro.tech/)**

Get familiar with llm-wrapper 🚀: [introductory jupyter notebook](https://github.com/allegro/llm-wrapper/blob/main/examples/introduction.ipynb)

___

## Quickstart

Install the package via pip:

```
pip install llm-wrapper
```

Configure endpoint credentials and start querying the model!

```python
from llm_wrapper.models import AzureOpenAIModel
from llm_wrapper.domain.configuration import AzureOpenAIConfiguration

configuration = AzureOpenAIConfiguration(
    api_key="<OPENAI_API_KEY>",
    base_url="<OPENAI_API_BASE>",
    api_version="<OPENAI_API_VERSION>",
    deployment="<OPENAI_API_DEPLOYMENT_NAME>",
    model_name="<OPENAI_API_MODEL_NAME>"
)

gpt_model = AzureOpenAIModel(config=configuration)
gpt_response = gpt_model.generate("Plan me a 3-day holiday trip to Italy")
```
___

## Local Development

### Installation from the source

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

### Updating the documentation

Run `mkdocs serve` to serve a local instance of the documentation.

Modify the content of `docs` directory to update the documentation. The updated content will be deployed
via the github action `.github/workflows/docs.yml`

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