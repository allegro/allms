---
layout: default
title: Forcing model response format
nav_order: 2
parent: Tutorial
---

If you want to force the model to output the response in a given JSON schema, `llm-wrapper` provides an easy way to do 
it. You just need to provide a data model that describes the desired output format and the package does all the rest. 
As an output you get string already parsed to a provided data model class.

Here's how to use this functionality step by step:
1. Define the desired output data model class. It needs to inherit from pydantic `BaseModel`. Each field should have
type defined and a description provided in `Field()` which should describe what given field means. By providing accurate
description, you make it easier for the model to generate proper response.
    ```python
    import typing
    
    from pydantic import BaseModel, Field
    
    class ReviewOutputDataModel(BaseModel):
        summary: str = Field(description="Summary of a product description")
        should_buy: bool = Field(description="Recommendation whether I should buy the product or not")
        brand_name: str = Field(description="Brand of the coffee")
        aroma:str = Field(description="Description of the coffee aroma")
        cons: typing.List[str] = Field(description="List of cons of the coffee")
    ```

2. Provide the data model class together with prompt and input data to the `.generate()` method. `llm-wrapper` will 
automatically force the model to output the data in the provided format and will parse the string returned from the
model to the provided data model class.

    ```python
    review = "Marketing is doing its job and I was tempted too, but this Blue Orca coffee is nothing above the level of coffees from the supermarket. And the method of brewing or grinding does not help here. The coffee is simply weak - both in terms of strength and taste. I do not recommend."
    
    prompt = "Summarize review of the coffee. Review: {review}"
    input_data = [
        InputData(input_mappings={"review": review}, id="0")
    ]
    
    responses = model.generate(
        prompt=prompt, 
        input_data=input_data,
        output_data_model_class=ReviewOutputDataModel
    )
    response = responses[0].response
    ```

Now we can check the response:
```python
>>> type(response)
ReviewOutputDataModel

>>> response.should_buy
False

>>> response.brand_name
"Blue Orca"

>>> response.aroma
"Not mentioned in the review"

>>> response.cons
['Weak in terms of strength', 'Weak in terms of taste']
```

### What to do when output formatting doesn't work?

The feature described above works best with advanced proprietary models like GPT and PaLM/Gemini. Less capable models like Llama2 or Mistral
may not able to understand instructions passed as output_dataclasses, and in most cases the returned response won't be compatible
with the defined format, resulting in an unexpected response.

In such cases, we recommend to address the issue by specifying in the prompt how the response should look like. Using
few-shot learning techniques is also advisable. In the case of JSON-like output, use double curly brackets to escape them in order
to use them in the JSON example.

### How forcing response format works under the hood?
To force the model to provide output in a desired format, under the hood `llm-wrapper` automatically adds a description
of the desired output format. For example, for the `ReviewOutputDataModel` the description looks like this:
````text
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"summary": {"title": "Summary", "description": "Summary of a product description", "type": "string"}, "should_buy": {"title": "Should Buy", "description": "Recommendation whether I should buy the product or not", "type": "boolean"}, "brand_name": {"title": "Brand Name", "description": "Brand of the coffee", "type": "string"}, "aroma": {"title": "Aroma", "description": "Description of the coffee aroma", "type": "string"}, "cons": {"title": "Cons", "description": "List of cons of the coffee", "type": "array", "items": {"type": "string"}}}, "required": ["summary", "should_buy", "brand_name", "aroma", "cons"]}
```
````

This feature is really helpful, but you have to bear in mind that by using it you increase the number or prompt tokens
so it'll make the requests more costly (if you're using model with per token pricing)

If the model will return an output that doesn't comform to the defined data model, raw model response will be returned
in `ResponseData.response` and `ResponseData.error` will be `OutputParserException`.