import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent.parent.parent / "examples"
