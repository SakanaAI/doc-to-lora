# (c) Meta Platforms, Inc. and affiliates.
import logging
import socket
from datetime import datetime

import torch
from torch.autograd.profiler import record_function
from torchvision import models

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def run_resnet50(num_iters=5, device="cuda:0"):
    model = models.resnet50().to(device=device)
    inputs = torch.randn(1, 3, 224, 224, device=device)
    labels = torch.rand_like(model(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iters):
            prof.step()
            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # Warm up
    run_resnet50()
    # Run the resnet50 model
    run_resnet50()
