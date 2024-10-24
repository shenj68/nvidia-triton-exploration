import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import Tensor
from pytriton.triton import Triton
from pytriton.client import ModelClient


@batch
def infer_fn(data):
    result = data * np.array([[-1]], dtype=np.float32)  # Process inputs and produce result
    return [result] 


triton = Triton()
triton.bind(
    model_name="Linear",
    infer_func=infer_fn,
    inputs=[Tensor(name="data", dtype=np.float32, shape=(-1,)),],
    outputs=[Tensor(name="result", dtype=np.float32, shape=(-1,)),],
)
triton.run()

client = ModelClient("localhost", "Linear")
data = np.array([1, 2, ], dtype=np.float32)
print(client.infer_sample(data=data))

client.close()
triton.stop()