import pytest
from typing import List, Tuple
from unittest.mock import patch, MagicMock

import pandas as pd
from pandas import DataFrame

from utils import (
    url_decode_cols,
    url_encode_cols,
    pascal_case,
    get_client_class,
    get_memory_usage,
    clean_feature_names,
)


@pytest.fixture
def sample_dataframe() -> DataFrame:
    """Fixture for creating a sample Pandas DataFrame."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


@pytest.fixture
def columns_encoded_decoded() -> List[Tuple[str, str]]:
    """Fixture for providing both encoded and decoded column names."""
    return [("hello%20world", "hello world"), ("python%3A3", "python:3")]


def test_url_decode_cols(columns_encoded_decoded: List[Tuple[str, str]]) -> None:
    """Test for url_decode_cols function."""
    columns_encoded = [col[0] for col in columns_encoded_decoded]
    columns_decoded = [col[1] for col in columns_encoded_decoded]
    assert url_decode_cols(columns_encoded) == columns_decoded


def test_url_encode_cols(dataframe: DataFrame) -> None:
    """Test for url_encode_cols function."""
    df = dataframe.copy()
    df.columns = ["hello world", "python:3"]
    encoded_df = url_encode_cols(df)
    assert list(encoded_df.columns) == ["hello%20world", "python%3A3"]


def test_pascal_case() -> None:
    """Test for pascal_case function."""
    assert pascal_case("hello-world") == "HelloWorld"
    assert pascal_case("hello_world") == "HelloWorld"
    assert pascal_case("hello world") == "HelloWorld"
    assert pascal_case("hello   world") == "HelloWorld"
    assert pascal_case("hello\tworld") == "HelloWorld"
    assert pascal_case("hello*world") == "HelloWorld"


@patch("utils.import_module")
@patch("utils.log_message")
def test_get_client_class_not_found(
    mock_log_message: MagicMock, mock_import_module: MagicMock
) -> None:
    """Test get_client_class when client module is not found."""
    mock_import_module.side_effect = ModuleNotFoundError
    result = get_client_class("nonexistent_client")
    mock_log_message.assert_called_once()
    assert result.__name__ == "Base"


def test_get_memory_usage(dataframe) -> str:
    """Test for get_memory_usage function."""
    assert "MB" in get_memory_usage(dataframe)


def test_clean_feature_names() -> None:
    """Test for clean_feature_names function."""
    assert clean_feature_names(["hello world!", "python:3", "C++ programming"]) == [
        "hello_world",
        "python3",
        "C_programming",
    ]


if __name__ == "__main__":
    pytest.main()
