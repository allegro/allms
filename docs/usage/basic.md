# Basic Usage

## Single Query 

In the simplest approach you just need to pass a prompt and the model will provide a response for it.  

```python
from llm_wrapper.models import AzureOpenAIModel
from llm_wrapper.domain.configuration import AzureOpenAIConfiguration
from llm_wrapper.domain.response import ResponseData

configuration = AzureOpenAIConfiguration(
    api_key="<OPENAI_API_KEY>",
    base_url="<OPENAI_API_BASE>",
    api_version="<OPENAI_API_VERSION>",
    deployment="<OPENAI_API_DEPLOYMENT_NAME>",
    model_name="<OPENAI_API_MODEL_NAME>"
)

model = AzureOpenAIModel(config=configuration)

response = model.generate("What is the capital of Poland?")
print(response)

#[ResponseData(response='The capital of Poland is Warsaw.', input_data=None, number_of_prompt_tokens=7, number_of_generated_tokens=7, error=None)]
```

As a response you'll get `List[ResponseData]`, where the first element will contain response from the model in the
`ResponseData.response` field and also information about number of prompt and generated tokens. If any error occurred
also `ResponseData.error` field will be filled with the actual exception.

## Single Query with System Prompt

A System Prompt can be passed along with a standard prompt. Please note that adding a system prompt will increase the 
prompt token count for your query, increasing costs and latency.

```python
response = model.generate(
    system_prompt="You are an AI agent answering questions like a student during an exam. Answer the question in Polish.",
    prompt="What is the capital of Poland?"
)
print(response)
# Stolica Polski to Warszawa.
```

