## `class allms.models.AzureLlama2Model` API
### Methods
```python
__init__(
    config: AzureSelfDeployedConfiguration,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
    model_total_max_tokens: int = 4096,
    max_concurrency: int = 1000,
    max_retries: int = 8
)
```
#### Parameters
- `config` (`AzureSelfDeployedConfiguration`): an instance of `AzureSelfDeployedConfiguration` class
- `temperature` (`float`): The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more
   random, while lower values like 0.2 will make it more focused and deterministic. Default: `0.0`.
- `top_p` (`float`): Default: `1.0`.
- `max_output_tokens` (`int`): The maximum number of tokens to generate by the model. The total length of input tokens 
   and generated tokens is limited by the model's context length. Default: `512`.
- `model_total_max_tokens` (`int`): Context length of the model - maximum number of input plus generated tokens.
   Default: `4096`.
- `max_concurrency` (`int`): Maximum number of concurrent requests. Default: `1000`.
- `max_retries` (`int`): Maximum number of retries if a request fails. Default: `8`.

---

```python
generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    input_data: typing.Optional[typing.List[InputData]] = None,
    output_data_model_class: typing.Optional[typing.Type[BaseModel]] = None
) -> typing.List[ResponseData]:
```
#### Parameters
- `prompt` (`str`): Prompt to use to query the model.
- `system_prompt` (`Optional[str]`): System prompt that will be used by the model.
- `input_data` (`Optional[List[InputData]]`): If prompt contains symbolic variables you can use this parameter to
   generate model responses for batch of examples. Each symbolic variable from the prompt should have mapping provided
   in the `input_mappings` of `InputData`.
- `output_data_model_class` (`Optional[Type[BaseModel]]`): Generated response is automatically parsed to this class. WARNING: You need to manually provide the JSON format instructions in the prompt, they are not injected for this model.

#### Returns
`List[ResponseData]`: Each `ResponseData` contains the response for a single example from `input_data`. If `input_data`
is not provided, the length of this list is equal 1, and the first element is the response for the raw prompt. 

---

## `class allms.domain.configuration.AzureSelfDeployedConfiguration` API
```python
AzureSelfDeployedConfiguration(
    api_key: str,
    endpoint_url: str,
    deployment: str
)
```
#### Parameters
- `api_key` (`str`): Authentication key for the endpoint.
- `endpoint_url` (`str`): URL of pre-existing endpoint.
- `deployment` (`str`): The name under which the model was deployed.

---

### Example usage

```python
from allms.models import AzureLlama2Model
from allms.domain.configuration import AzureSelfDeployedConfiguration

configuration = AzureSelfDeployedConfiguration(
    api_key="<AZURE_API_KEY>",
    endpoint_url="<AZURE_ENDPOINT_URL>",
    deployment="<AZURE_DEPLOYMENT_NAME>"
)

llama_model = AzureLlama2Model(config=configuration)
llama_response = llama_model.generate("2+2 is?")
```