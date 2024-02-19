---
layout: default
title: Tutorial
nav_order: 3
has_children: true
---

## Unleash the Library's Potential

Dive into our tutorial to fully grasp the library's capabilities. From beginners to experts, it's designed to accommodate all levels of experience.

### What You'll Gain

- **Discover Core Features**: Learn how to harness the library's power for your projects.
- **Efficient Learning**: Save time with hands-on examples, tips, and best practices.
- **Hidden Gems**: Unlock advanced functions that can enhance your work.
- **Troubleshooting**: Find solutions to common issues.
- **Community Support**: Join a helpful user community.

### Access the Tutorial

Visit our documentation to access the tutorial. It's your key to maximizing the potential of the library, whether you're using it for research, development, or any other application. Start your journey now and witness the limitless possibilities it offers!


## Before You Begin

All the examples presented in the tutorials assume that you pass a configuration object to the model. How to do
it is described in detail in the Quick Start section. For example, for the Azure GPT model it's done like this:
```python
from llm_wrapper.models.azure_openai import AzureOpenAIModel
from llm_wrapper.domain.configuration import AzureOpenAIConfiguration

configuration = AzureOpenAIConfiguration(
    api_key="<OPENAI_API_KEY>",
    base_url="<OPENAI_API_BASE>",
    api_version="<OPENAI_API_VERSION>",
    deployment="<OPENAI_API_DEPLOYMENT_NAME>",
    model_name="<OPENAI_API_MODEL_NAME>"
)

model = AzureOpenAIModel(config=configuration)
```

-------
<sub>_*Documentation Powered by GPT-3.5-turbo_</sub>