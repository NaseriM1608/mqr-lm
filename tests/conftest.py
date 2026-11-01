import torch


def pytest_sessionstart(session):
    # Small correctness fixtures are slower with an oversubscribed BLAS pool.
    torch.set_num_threads(2)
