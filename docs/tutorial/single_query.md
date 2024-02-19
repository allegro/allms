---
layout: default
title: Single query
nav_order: 0
parent: Tutorial
---

In the simplest approach you just need to pass a prompt and the model will provide a response for it.  

```python
>>> model.generate("What is the capital of Poland?")
[ResponseData(response='The capital of Poland is Warsaw.', input_data=None, number_of_prompt_tokens=7, number_of_generated_tokens=7, error=None)]
```

As a response you'll get `List[ResponseData]`, where the first element will contain response from the model in the
`ResponseData.response` field and also information about number of prompt and generated tokens. If any error occurred
also `ResponseData.error` field will be filled with the actual exception. 
