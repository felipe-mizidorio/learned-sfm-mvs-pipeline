import pytest


def _cuda_available() -> bool:
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional `learned` group
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.gpu`` tests when no CUDA device is present."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
