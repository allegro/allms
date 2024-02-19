---
layout: default
title: Batch query
nav_order: 1
parent: Tutorial
---

If you want to generate responses for a batch of examples, you can achieve this by preparing a prompt with symbolic
variables and providing input data that will be injected into this prompt. `llm-wrapper` will automatically make these
requests in an async mode and retry them in case of any API error.

Let's say we want to classify reviews of coffee as positive or negative. Here's how to do it:
```python
positive_review_0 = "Very good coffee, lightly roasted, with good aroma and taste. The taste of sourness is barely noticeable (which is good because I don't like sour coffees). After grinding, the aroma spreads throughout the room. I recommend it to all those who do not like strongly roasted and pitch-black coffees. A very good solution is to close the package with string, which allows you to preserve the aroma and freshness."
positive_review_1 = "Delicious coffee!! Delicate, just the way I like it, and the smell after opening is amazing. It smells freshly roasted. Faithful to Lavazza coffee for years, I decided to look for other flavors. Based on the reviews, I blindly bought it and it was a 10-shot, it outperformed Lavazze in taste. For me the best."
negative_review = "Marketing is doing its job and I was tempted too, but this coffee is nothing above the level of coffees from the supermarket. And the method of brewing or grinding does not help here. The coffee is simply weak - both in terms of strength and taste. I do not recommend."

prompt = "You'll be provided with a review of a coffe. Decide if the review is positive or negative. Review: {review}"
input_data = [
    InputData(input_mappings={"review": positive_review_0}, id="0"),
    InputData(input_mappings={"review": positive_review_1}, id="1"),
    InputData(input_mappings={"review": negative_review}, id="2")
]

responses = model.generate(prompt=prompt, input_data=input_data)
```

As an output we'll get `List[ResponseData]` where each `ResponseData` will contain response for a single example from
`input_data`. The requests are performed in an async mode, so remember that the order of the `responses` is not the same
as the order of the `input_data`. That's why together with the response, we pass also the `ResponseData.input_data` to
the output.

So let's see the responses:
```python
>>> {f"review_id={response.input_data.id}": response.response for response in responses}
{
    'review_id=0': 'The review is positive.',
    'review_id=1': 'The review is positive.',
    'review_id=2': 'The review is negative.'
}

```

## Multiple symbolic variables
You can also define prompt with multiple symbolic variables. The rule is that each symbolic variable from the prompt
should have mapping provided in the `input_mappings` of `InputData`. Let's say we want to provide two reviews in one 
prompt and let the model decide which one of them is positive. Here's how to do it:

```python
prompt = """You'll be provided with two reviews of a coffee. Decide which one is positive.

First review: {first_review}
Second review: {second_review}"""
input_data = [
    InputData(input_mappings={"first_review": positive_review_0, "second_review": negative_review}, id="0"),
    InputData(input_mappings={"first_review": negative_review, "second_review": positive_review_1}, id="1"),
]

responses = model.generate(prompt=prompt, input_data=input_data)
```

And the results:
```python
>>> {f"example_id={response.input_data.id}": response.response for response in responses}
{
    'example_id=0': 'The first review is positive.',
    'example_id=1': 'The second review is positive.'
}
```

## Control the number of concurrent requests
As it's written above, `llm-wrapper` automatically makes requests in an async mode. By default, the maximum number of 
concurrent requests is set to 1000. You can control this value by setting the `max_concurrency` parameter when
initializing the model. Set it to a value that is appropriate for your model endpoint.

## Use a common asyncio event loop
By default, each model instance has its own event loop for handling the execution of async tasks. If you want to use
a common loop for multiple models or to have a custom loop, it's possible to specify it in the model constructor:

```python
import asyncio

from llm_wrapper.models import AzureOpenAIModel
from llm_wrapper.domain.configuration import AzureOpenAIConfiguration


custom_event_loop = asyncio.new_event_loop()

configuration = AzureOpenAIConfiguration(
    api_key="<OPENAI_API_KEY>",
    base_url="<OPENAI_API_BASE>",
    api_version="<OPENAI_API_VERSION>",
    deployment="<OPENAI_API_DEPLOYMENT_NAME>",
    model_name="<OPENAI_API_MODEL_NAME>"
)

model = AzureOpenAIModel(
    config=configuration,
    event_loop=custom_event_loop
)
``` 