---
layout: default
title: AzureOpenAIModel
parent: Models
grand_parent: API
nav_order: 1
---

## `class llm_wrapper.models.AzureOpenAIModel` API
### Methods
```python
__init__(
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    request_timeout_s: int = 60,
    model_total_max_tokens: int = 4096,
    max_concurrency: int = 1000,
    max_retries: int = 8
)
```
#### Parameters
- `temperature` (`float`): The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more
   random, while lower values like 0.2 will make it more focused and deterministic. Default: `0.0`.
- `max_output_tokens` (`int`): The maximum number of tokens to generate by the model. The total length of input tokens 
   and generated tokens is limited by the model's context length. Default: `512`.
- `request_timeout_s` (`int`): Timeout for requests to the model. Default: `60`.
- `model_total_max_tokens` (`int`): Context length of the model - maximum number of input plus generated tokens.
   Default: `4096`.
- `max_concurrency` (`int`): Maximum number of concurrent requests. Default: `1000`.
- `max_retries` (`int`): Maximum number of retries if a request fails. Default: `8`.

---

```python
generate(
    prompt: str,
    input_data: typing.Optional[typing.List[InputData]] = None,
    output_data_model_class: typing.Optional[typing.Type[BaseModel]] = None
) -> typing.List[ResponseData]:
```
#### Parameters
- `prompt` (`str`): Prompt to use to query the model.
- `input_data` (`Optional[List[InputData]]`): If prompt contains symbolic variables you can use this parameter to
   generate model responses for batch of examples. Each symbolic variable from the prompt should have mapping provided
   in the `input_mappings` of `InputData`.
- `output_data_model_class` (`Optional[Type[BaseModel]]`): If provided forces the model to generate output in the
  format defined by the passed class. Generated response is automatically parsed to this class.

#### Returns
`List[ResponseData]`: Each `ResponseData` contains the response for a single example from `input_data`. If `input_data`
is not provided, the length of this list is equal 1, and the first element is the response for the raw prompt. 

---

```python
AzureOpenAIModel.setup_environment(
    openai_api_key: str,
    openai_api_base: str,
    openai_api_version: str,
    openai_api_deployment_name: str,
    openai_api_type: str = "azure",
    model_name: str = "gpt-3.5-turbo",
)
```
Sets up the environment for the `AzureOpenAIModel` model.
#### Parameters
- `openai_api_key` (`str`):  The API key for your Azure OpenAI resource. You can find this in the Azure portal under
   your Azure OpenAI resource.
- `openai_api_base` (`str`): The base URL for your Azure OpenAI resource. You can find this in the Azure portal under
   your Azure OpenAI resource. 
- `openai_api_version` (`str`): The API version.
- `openai_api_deployment_name` (`str`): The name under which the model was deployed.
- `openai_api_type` (`str`): Default: `"azure"`.
- `model_name` (`str`): Model name to use. Default: `"gpt-3.5-turbo"`.

---

### Example usage
```python
from llm_wrapper.models import AzureOpenAIModel
from llm_wrapped.domain.configuration import AzureOpenAIConfiguration

configuration = AzureOpenAIConfiguration(
    api_key="<OPENAI_API_KEY>",
    base_url="<OPENAI_API_BASE>",
    api_version="<OPENAI_API_VERSION>",
    deployment="<OPENAI_API_DEPLOYMENT_NAME>",
    model_name="<OPENAI_API_MODEL_NAME>"
)

gpt_model = AzureOpenAIModel(config=configuration)
gpt_response = gpt_model.generate("2+2 is?")
```
