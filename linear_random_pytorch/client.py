"""Client for linear_random example."""

import argparse
import logging

import torch  # pytype: disable=import-error

from pytriton.client import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("examples.linear_random_pytorch.client")

    input1_batch = torch.randn(128, 20).cpu().detach().numpy()
    
    logger.info(f"Input: {input1_batch.tolist()}")

    with ModelClient(args.url, "Linear") as client:
        logger.info("Sending request")
        result_dict = client.infer_batch(input1_batch)

    for output_name, output_batch in result_dict.items():
        logger.info(f"{output_name}: {output_batch.tolist()}")


if __name__ == "__main__":"""Client for linear_random example."""

import argparse
import logging

import torch  # pytype: disable=import-error

from pytriton.client import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("examples.linear_random_pytorch.client")

    input1_batch = torch.randn(128, 20).cpu().detach().numpy()

    logger.info(f"Input: {input1_batch.tolist()}")

    with ModelClient(args.url, "Linear") as client:
        logger.info("Sending request")
        result_dict = client.infer_batch(input1_batch)

    with open("output.txt", "w") as f:
        for output_name, output_batch in result_dict.items():
            output_str = f"{output_name}: {output_batch.tolist()}\n"
            logger.info(output_str)
            f.write(output_str)


if __name__ == "__main__":
    main()
    