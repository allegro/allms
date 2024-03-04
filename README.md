# allms

___
## About

allms is a versatile and powerful library designed to streamline the process of querying Large Language Models
(LLMs) 🤖💬

Developed by the Allegro engineers, allms is based on popular libraries like transformers, pydantic, and langchain. It takes care 
of the boring boiler-plate code you write around your LLM applications, quickly enabling you to prototype ideas, and eventually helping you to scale up 
for production use-cases!

Among the allms most notable features, you will find:

* **😊 Simple and User-Friendly Interface**: The module offers an intuitive and easy-to-use interface, making it straightforward to work with the model.

* **🔀 Asynchronous Querying**: Requests to the model are processed asynchronously by default, ensuring efficient and non-blocking interactions.

* **🔄 Automatic Retrying Mechanism** : The module includes an automatic retrying mechanism, which helps handle transient errors and ensures that queries to the model are robust.

* **🛠️ Error Handling and Management**: Errors that may occur during interactions with the model are handled and managed gracefully, providing informative error messages and potential recovery options.

* **⚙️ Output Parsing**: The module simplifies the process of defining the model's output format as well as parsing and working with it, allowing you to easily extract the information you need.

___

## Supported Models

| LLM Family  | Hosting             | Supported LLMs                          |
|-------------|---------------------|-----------------------------------------|
| GPT(s)      | OpenAI endpoint     | `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo` |
| Google LLMs | VertexAI deployment | `text-bison@001`, `gemini-pro`          |
| Llama2      | Azure deployment    | `llama2-7b`, `llama2-13b`, `llama2-70b` |
| Mistral     | Azure deployment    | `Mistral-7b`, `Mixtral-7bx8`            |
| Gemma       | GCP deployment      | `gemma`                                 |

* Do you already have a subscription to a Cloud Provider for any the models above? Configure
the model using your credentials and start querying!
* Are you interested in knowing how to self-deploy open-source models in Azure and GCP?
Consult our [guide](https://allms.allegro.tech/usage/deploy_open_source_models/)

___

## Documentation

Full documentation available at **[allms.allegro.tech](https://allms.allegro.tech/)**

Get familiar with allms 🚀: [introductory jupyter notebook](https://github.com/allegro/allms/blob/main/examples/introduction.ipynb)

___

## Quickstart

### Installation 🚧

Install the package via pip:

```
pip install allms
```

### Basic Usage ⭐

Configure endpoint credentials and start querying the model with any prompt:

```python
from allms.models import AzureOpenAIModel
from allms.domain.configuration import AzureOpenAIConfiguration

configuration = AzureOpenAIConfiguration(
   api_key="your-secret-api-key",
   base_url="https://endpoint.openai.azure.com/",
   api_version="2023-03-15-preview",
   deployment="gpt-35-turbo",
   model_name="gpt-3.5-turbo"
)

gpt_model = AzureOpenAIModel(config=configuration)
gpt_response = gpt_model.generate(prompt="Plan me a 3-day holiday trip to Italy")
```

You can pass also a system prompt:

```python
gpt_response = gpt_model.generate(
    system_prompt="You are an AI assistant acting as a trip planner",
    prompt="Plan me a 3-day holiday trip to Italy"
)
```

### Advanced Usage 🔥

### Batch Querying and Symbolic Variables

If you want to generate responses for a batch of examples, you can achieve this by preparing a prompt with symbolic
variables and providing input data that will be injected into the prompt. Symbolic variables can be more than one.

```python
positive_review_0 = "Very good coffee, lightly roasted, with good aroma and taste. The taste of sourness is barely noticeable (which is good because I don't like sour coffees). After grinding, the aroma spreads throughout the room. I recommend it to all those who do not like strongly roasted and pitch-black coffees. A very good solution is to close the package with string, which allows you to preserve the aroma and freshness."
positive_review_1 = "Delicious coffee!! Delicate, just the way I like it, and the smell after opening is amazing. It smells freshly roasted. Faithful to Lavazza coffee for years, I decided to look for other flavors. Based on the reviews, I blindly bought it and it was a 10-shot, it outperformed Lavazze in taste. For me the best."
negative_review = "Marketing is doing its job and I was tempted too, but this coffee is nothing above the level of coffees from the supermarket. And the method of brewing or grinding does not help here. The coffee is simply weak - both in terms of strength and taste. I do not recommend."

prompt = "You'll be provided with a review of a coffe. Decide if the review is positive or negative. Review: {review}"
input_data = [
    InputData(input_mappings={"review": positive_review_0}, id="0"),
    InputData(input_mappings={"review": positive_review_1}, id="1"),
    InputData(input_mappings={"review": negative_review}, id="2")
]

responses = model.generate(prompt=prompt, input_data=input_data)

# >>> {f"review_id={response.input_data.id}": response.response for response in responses}
# {
#     'review_id=0': 'The review is positive.',
#     'review_id=1': 'The review is positive.',
#     'review_id=2': 'The review is negative.'
# }
```

### Forcing Structured Output Format

Through pydantic integration, in allms you can pass an output dataclass and force the LLM to provide
you the response in a structured way.

```python
from pydantic import BaseModel, Field

class ReviewOutputDataModel(BaseModel):
    summary: str = Field(description="Summary of a product description")
    should_buy: bool = Field(description="Recommendation whether I should buy the product or not")
    brand_name: str = Field(description="Brand of the coffee")
    aroma:str = Field(description="Description of the coffee aroma")
    cons: list[str] = Field(description="List of cons of the coffee")


review = "Marketing is doing its job and I was tempted too, but this Blue Orca coffee is nothing above the level of coffees from the supermarket. And the method of brewing or grinding does not help here. The coffee is simply weak - both in terms of strength and taste. I do not recommend."
    
prompt = "Summarize review of the coffee. Review: {review}"
input_data = [InputData(input_mappings={"review": review}, id="0")]
    
responses = model.generate(
    prompt=prompt, 
    input_data=input_data,
    output_data_model_class=ReviewOutputDataModel
)
response = responses[0].response

# >>> type(response)
# ReviewOutputDataModel
# 
# >>> response.should_buy
# False
# 
# >>> response.brand_name
# "Blue Orca"
# 
# >>> response.aroma
# "Not mentioned in the review"
# 
# >>> response.cons
# ['Weak in terms of strength', 'Weak in terms of taste']
```
___

## Local Development 🛠️

### Installation from the source

We assume that you have python `3.10.*` installed on your machine. 
You can set it up using [pyenv](https://github.com/pyenv/pyenv#installationbrew) 
([How to install pyenv on MacOS](https://jordanthomasg.medium.com/python-development-on-macos-with-pyenv-2509c694a808)). To install allms env locally:

* Activate your pyenv;
* Install Poetry via:

```bash
make install-poetry
```

* Install allms dependencies with the command:

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

When a new version of allms is ready to be released, do the following operations:

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
    4. Upload `allms-<NEW-VERSION>.whl` and `allms-<NEW-VERSION>.tar.gz` as assets;
    5. Click `Publish release`.