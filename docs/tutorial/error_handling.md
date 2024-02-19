---
layout: default
title: Error handling
nav_order: 3
parent: Tutorial
---


## Too long prompt
Each LLM has its own context size defined. This is the maximum number of input plus output tokens that the model is able
to consume. `llm-wrapper` before sending the request to the model automatically checks if your input data will fit into
the model's context size and if not it'll either:
- raise `ValueError` saying that your prompt is too long if the prompt alone has already more tokens than the allowed
  maximum context size of the model
- log warning saying that number of prompt tokens plus generated tokens may exceed the max allowed number of tokens of 
  the model if the number of tokens in the prompt plus the `max_output_tokens` you set for the model is longer than the 
  allowed maximum context size of the model

In the first case, the only solution is to truncate the input data, so that it'll fit into the context size of the
model.

The second case is just a warning, because the model will be able to start the generation, but it may fail randomly
if the number of generated tokens will be long enough to exceed the model maximum context size. In this case you have
two options. You can either truncate the input data or lower the `max_output_tokens` so that they added together won't 
exceed the max context size.

In the future releases, we plan to add automatic long sequences handling. Then the package will be able to automatically
split the whole input into shorter chunks, process them separately and combine the outputs. But it's not there yet.


## Output parsing errors
If you use the [Forcing model response format](forcing_response_format.md) functionality, sometimes the model can 
generate a response that actually doesn't comform to the provided output data schema. In this case, `llm-wrapper` won't
be able to parse the output to the provided output data model class. So as a response you'll get a `ResponseData` where
`ResponseData.response` will be a raw, unparsed response from the model, and the `ResponseData.error` will be
`OutputParserException`.


## API errors
`llm-wrapper` automatically retries failed requests. But even with this feature, the model can fail to return a response
more times than the maximum number of retries (which is currently set to 8) or some other unexpected errors may occur.
In all of these cases, `ResponseData.error` will contain the exception that occurred. So a good rule of thumb is to 
first check the `ResponseData.error` and only if it's empty move to processing the response of the model.