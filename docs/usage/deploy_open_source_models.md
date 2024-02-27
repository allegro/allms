# How to Deploy Azure Open-source LLMs?


To use Open-source models like Llama or Mistral with allms, first you have to deploy it on your own on Azure as a ML Online Endpoint. 
Here's how to do it:

1. Go to [ml.azure.com](https://ml.azure.com/) and use a subscription with a workspace that has access to the
   `Model catalog`.
2. On the left click `Model catalog`, then under `Introducing Llama 2` click `View models`.
3. Click the model you want to deploy.
4. Click `Deploy -> Real-time endpoint`.
5. Select `Skip Azure AI Content Safety` and click `Proceed`.
6. Select a virtual machine and click `Deploy`. You must have sufficient quota to deploy the models. 
7. In the menu on the left, click `Endpoints` and select the endpoint you've just created.
8. After the deployment is complete, you'll see `Consume` tab where the endpoint URL and authentication key will be
   provided.
9. Now you can start using the model by configuring it as in the example below:

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


In case of any problems with deployment, you can review this guide on the Azure blog: 
[Introducing Llama 2 on Azure](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/introducing-llama-2-on-azure/ba-p/3881233)

