
import numpy as np
import pytest

from sklearn.datasets import load_digits

from src.predict import load_model, predict_digit


@pytest.fixture(scope="module")
def model():
    """Load the trained model once for this test module."""
    return load_model()


@pytest.fixture(scope="module")
def sample():
    """Return one real sample with 64 raw pixel features."""
    digits = load_digits()
    return digits.data[0]


def test_model_loads_successfully(model):
    assert model is not None


def test_valid_input_returns_prediction(model, sample):
    result = predict_digit(sample, model)

    assert "predicted_digit" in result
    assert "confidence" in result

    assert isinstance(result["predicted_digit"], int)
    assert 0 <= result["predicted_digit"] <= 9

    assert 0.0 <= result["confidence"] <= 1.0


def test_invalid_input_shape_is_rejected(model):
    invalid_sample = np.zeros(63)

    with pytest.raises(ValueError, match="Expected 64 features"):
        predict_digit(invalid_sample, model)


def test_non_finite_input_is_rejected(model):
    invalid_sample = np.zeros(64)
    invalid_sample[0] = np.nan

    with pytest.raises(ValueError, match="finite numbers"):
        predict_digit(invalid_sample, model)


@pytest.mark.parametrize("invalid_value", [-1, 17])
def test_out_of_range_pixel_values_are_rejected(
    model, invalid_value
):
    invalid_sample = np.zeros(64)
    invalid_sample[0] = invalid_value

    with pytest.raises(ValueError, match="between 0 and 16"):
        predict_digit(invalid_sample, model)
