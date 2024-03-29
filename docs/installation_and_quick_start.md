# Installation
Install the package via pip:

```bash
pip install allms
```

# Quick Start 

To use our package, you must have access to the credentials of the endpoint with the deployed model.
Each of the supported models have a different set of credentials
that must be passed in the corresponding configuration object. Below is a brief overview of how to use each of these models.

## Simple usage

### Azure GPT

```python
from allms.models import AzureOpenAIModel
from allms.domain.configuration import AzureOpenAIConfiguration

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

### VertexAI PaLM

```python
from allms.models import VertexAIPalmModel
from allms.domain.configuration import VertexAIConfiguration

configuration = VertexAIConfiguration(
    cloud_project="<GCP_PROJECT_ID>",
    cloud_location="<MODEL_REGION>"
)

palm_model = VertexAIPalmModel(config=configuration)
palm_response = palm_model.generate("2+2 is?")
```

* `<GCP_PROJECT_ID>`: The GCP project in which you have access to the PALM model.
* `<MODEL_REGION>`: The region where the model is deployed.

### VertexAI Gemini

```python
from allms.models import VertexAIGeminiModel
from allms.domain.configuration import VertexAIConfiguration

configuration = VertexAIConfiguration(
    cloud_project="<GCP_PROJECT_ID>",
    cloud_location="<MODEL_REGION>"
)

gemini_model = VertexAIGeminiModel(config=configuration)
gemini_response = gemini_model.generate("2+2 is?")
```

* `<GCP_PROJECT_ID>`: The GCP project in which you have access to the PALM model.
* `<MODEL_REGION>`: The region where the model is deployed.

### VertexAI Gemma

```python
from allms.models import VertexAIGemmaModel
from allms.domain.configuration import VertexAIModelGardenConfiguration

configuration = VertexAIModelGardenConfiguration(
    cloud_project="<GCP_PROJECT_ID>",
    cloud_location="<MODEL_REGION>",
    endpoint_id="<ENDPOINT_ID>"
)

gemini_model = VertexAIGemmaModel(config=configuration)
gemini_response = gemini_model.generate("2+2 is?")
```

* `<GCP_PROJECT_ID>`: The GCP project in which you have access to the PALM model.
* `<MODEL_REGION>`: The region where the model is deployed.
* `<ENDPOINT_ID>`: ID of an endpoint where the model has been deployed.

### Azure LLaMA 2

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

* `<AZURE_API_KEY>`: Authentication key for the endpoint.
* `<AZURE_ENDPOINT_URL>`: URL of pre-existing endpoint.
* `<AZURE_DEPLOYMENT_NAME>`: The name under which the model was deployed.

### Azure Mistral

```python
from allms.models.azure_mistral import AzureMistralModel
from allms.domain.configuration import AzureSelfDeployedConfiguration

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
