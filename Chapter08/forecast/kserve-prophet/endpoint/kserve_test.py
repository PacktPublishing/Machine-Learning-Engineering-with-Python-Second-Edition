from typing import List
from kserve import KServe
from kserve.model import KModel


def my_custom_model(inputs: List[float]) -> List[float]:
    # Custom model logic
    outputs = [x * 2 for x in inputs]
    return outputs


model = KModel(model_name="my-custom-model", model_fn=my_custom_model)

serve = KServe()
serve.create_or_update(model)
serve.wait_ready(model.model_name)

# At this point, your custom model is deployed and ready to serve requests
