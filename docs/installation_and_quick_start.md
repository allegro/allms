---
layout: default
title: Installation & Quick start
nav_order: 1
---

# Installation
Install the package via pip:

```bash
pip install llm-wrapper
```

# Quick Start 

To use our package, you must have access to the credentials of the endpoint with the deployed model.
Each of the supported models have a different set of credentials
that must be passed in the corresponding configuration object. Below is a brief overview of how to use each of these models.

## Simple usage

### Azure GPT

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
gpt_response = gpt_model.generate("2+2 is?")
```

* `<OPENAI_API_KEY>`: The API key for your Azure OpenAI resource. You can find this in the Azure portal under your
   Azure OpenAI resource.
* `<OPENAI_API_BASE>`: The base URL for your Azure OpenAI resource. You can find this in the Azure portal under your
   Azure OpenAI resource.
* `<OPENAI_API_VERSION>`: The API version.
* `<OPENAI_API_DEPLOYMENT_NAME>`: The name under which the model was deployed.
* `<OPENAI_API_MODEL_NAME>`: The underlying model's name.

### Google PaLM

```python
from llm_wrapper.models import VertexAIPalmModel 
from llm_wrapper.domain.configuration import VertexAIConfiguration

configuration = VertexAIConfiguration(
    cloud_project="<GCP_PROJECT_ID>",
    cloud_location="<MODEL_REGION>"
)

palm_model = VertexAIPalmModel(config=configuration)
palm_response = palm_model.generate("2+2 is?")
```

* `<GCP_PROJECT_ID>`: The GCP project in which you have access to the PALM model.
* `<MODEL_REGION>`: The region where the model is deployed.

### Google Gemini

```python
from llm_wrapper.models import VertexAIGeminiModel 
from llm_wrapper.domain.configuration import VertexAIConfiguration

configuration = VertexAIConfiguration(
    cloud_project="<GCP_PROJECT_ID>",
    cloud_location="<MODEL_REGION>"
)

gemini_model = VertexAIGeminiModel(config=configuration)
gemini_response = gemini_model.generate("2+2 is?")
```

* `<GCP_PROJECT_ID>`: The GCP project in which you have access to the PALM model.
* `<MODEL_REGION>`: The region where the model is deployed.

### Azure LLaMA 2

```python
from llm_wrapper.models import AzureLlama2Model 
from llm_wrapper.domain.configuration import AzureSelfDeployedConfiguration

configuration = AzureSelfDeployedConfiguration(
    api_key="<AZURE_API_KEY>",
    endpoint_url="<AZURE_ENDPOINT_URL>",
    deployment="<AZURE_DEPLOYMENT_NAME>"
)

llama_model = AzureLlama2Model(config=configuration)
llama_response = llama_model.generate("2+2 is?")
```

* `<AZURE_API_KEY>`: Authentication key for the endpoint.
* `<AZURE_ENDPOINT_URL>`: URL of pre-existing endpoint.
* `<AZURE_DEPLOYMENT_NAME>`: The name under which the model was deployed.

### Azure Mistral

```python
from llm_wrapper.models.azure_mistral import AzureMistralModel
from llm_wrapper.domain.configuration import AzureSelfDeployedConfiguration

configuration = AzureSelfDeployedConfiguration(
    api_key="<AZURE_API_KEY>",
    endpoint_url="<AZURE_ENDPOINT_URL>",
    deployment="<AZURE_DEPLOYMENT_NAME>"
)

mistral_model = AzureMistralModel(config=configuration)
mistral_response = mistral_model.generate("2+2 is?")
```

* `<AZURE_API_KEY>`: Authentication key for the endpoint.
* `<AZURE_ENDPOINT_URL>`: URL of pre-existing endpoint.
* `<AZURE_DEPLOYMENT_NAME>`: The name under which the model was deployed.
