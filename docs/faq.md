
# Frequently Asked Questions

### 1. How to use the allms in a python notebook?
When using the `allms` library, which utilizes asynchronous programming under the hood, you must install the `nest-asyncio` library to use it in a Jupyter notebook environment.

To ensure proper functionality, execute the following code at the beginning of your notebook:
```jupyterpython
!pip install nest-asyncio
import nest_asyncio
nest_asyncio.apply()
```



### 2. How can I estimate the cost of my queries?

The model provides information for each record about the count of tokens in the prompt and the count of generated tokens.
In most cases, pricing for Language Models (LLMs) is determined based on the total number of tokens processed, which encompasses both prompt tokens and generated tokens. It is essential to familiarize yourself with the pricing details offered by your service provider to understand the associated costs. An example pricing for AzureOpenAI can be found [here](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/).




