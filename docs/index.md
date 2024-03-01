<p align="center">
<img src="assets/images/logo.png" alt="aLLMs Logo"/>
</p>

# Introduction

`allms` is a versatile and powerful library designed to streamline the process of querying large language models, offering a user-friendly experience.  The `allms` module is designed to simplify interactions with the underlying model by providing the following features:

- **Simple and User-Friendly Interface**: The module offers an intuitive and easy-to-use interface, making it straightforward to work with the model.

- **Asynchronous Querying (Default)**: Requests to the model are processed asynchronously by default, ensuring efficient and non-blocking interactions.

- **Automatic Retrying Mechanism**: The module includes an automatic retrying mechanism, which helps handle transient errors and ensures that queries to the model are robust.

- **Error Handling and Management**: Errors that may occur during interactions with the model are handled and managed gracefully, providing informative error messages and potential recovery options.

- **Simple Output Parsing**: The module simplifies the process of parsing and working with the model's output, allowing you to easily extract the information you need.



### Supported Models

Currently, the library supports:

* OpenAI models hosted on Microsoft Azure (`gpt-3.5-turbo`, `gpt4`, `gpt4-turbo`);
* Google Cloud Platform VertexAI models (`PaLM2`, `Gemini`);
* Open-source models `Llama2` and `Mistral` self-deployed on Azure.

