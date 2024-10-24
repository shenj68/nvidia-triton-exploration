import logging

import numpy as np
import torch  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = torch.nn.Linear(20, 30).to(DEVICE).eval()


@batch
def _infer_fn(**inputs):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to(DEVICE)
    output1_batch_tensor = MODEL(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("examples.linear_random_pytorch.server")

with Triton() as triton:
    logger.info("Loading Linear model.")
    triton.bind(
        model_name="Linear",
        infer_func=_infer_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    logger.info("Serving models")
    triton.serve()