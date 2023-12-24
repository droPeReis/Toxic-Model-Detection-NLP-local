import pytest
import torch
import numpy as np
from transformers.trainer_utils import PredictionOutput
from src.ml.inference import predict

TESTS = [
    (
        np.array(
            [
                [
                    -0.1586,
                    0.2220,
                    -0.0367,
                    -0.2293,
                    0.3756,
                    -0.3352,
                    -0.3230,
                    -0.0227,
                    -0.0658,
                    0.1569,
                ]
            ]
        ),
        np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]]),
        "multi-label",
    ),
    (
        PredictionOutput(
            predictions=np.array(
                [
                    [
                        -0.1586,
                        0.2220,
                        -0.0367,
                        -0.2293,
                        0.3756,
                        -0.3352,
                        -0.3230,
                        -0.0227,
                        -0.0658,
                        0.1569,
                    ]
                ]
            ),
            label_ids=np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]]),
            metrics={},
        ),
        np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]]),
        "multi-label",
    ),
    (
        np.array(
            [
                [
                    -0.1576,
                    0.1573,
                    -0.0312,
                    -0.2690,
                    0.3792,
                    -0.3307,
                    -0.3826,
                    0.0392,
                    -0.1152,
                    0.1577,
                ]
            ]
        ),
        np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]),
        "multi-label",
    ),
    (
        PredictionOutput(
            predictions=np.array(
                [
                    [
                        -0.1576,
                        0.1573,
                        -0.0312,
                        -0.2690,
                        0.3792,
                        -0.3307,
                        -0.3826,
                        0.0392,
                        -0.1152,
                        0.1577,
                    ]
                ]
            ),
            label_ids=np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]),
            metrics={},
        ),
        np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]),
        "multi-label",
    ),
    (
        PredictionOutput(
            predictions=np.array(
                [
                    [0.3416589, 0.00159595],
                    [0.16833752, 0.03792517],
                    [0.21325374, -0.03534849],
                    [0.18841237, -0.03214778],
                    [0.37990564, -0.15444584],
                    [0.28276327, -0.04099831],
                    [0.4790431, -0.18628305],
                    [0.33323058, -0.1398493],
                    [0.3088707, -0.09429711],
                    [0.31372625, -0.08971833],
                    [0.2573364, -0.04797133],
                    [0.30788502, -0.12525819],
                    [0.31722286, -0.10512252],
                    [0.4186611, -0.19944343],
                    [0.24844109, -0.10448106],
                    [0.25360614, -0.02878625],
                    [0.29938588, -0.15905927],
                    [0.31121805, -0.08164844],
                    [0.36362365, -0.11202919],
                    [0.3316734, -0.13064188],
                    [0.43251607, -0.16324174],
                    [0.22488856, -0.02969664],
                    [0.261734, -0.12085498],
                    [0.33951756, -0.1320796],
                    [0.41219848, -0.26228532],
                    [0.28913674, -0.06983679],
                    [0.2548892, -0.02173016],
                    [0.37318906, -0.14351502],
                    [0.19286662, -0.05843454],
                    [0.28177795, -0.08032534],
                    [0.34407595, -0.14566495],
                    [0.38159952, -0.14447926],
                    [0.30558395, -0.09957042],
                    [0.39386562, -0.12204813],
                    [0.3280886, -0.15307719],
                    [0.358768, -0.18881796],
                    [0.15525346, -0.001724],
                    [0.1948127, -0.06672906],
                    [0.4632586, -0.3392518],
                    [0.37497327, -0.12811454],
                    [0.32120684, -0.04155049],
                    [0.25476477, -0.01910269],
                    [0.19781362, -0.02477918],
                    [0.38447687, -0.14513063],
                    [0.22305596, -0.04620795],
                    [0.2102885, -0.02880435],
                    [0.2639521, -0.07673061],
                    [0.04903182, 0.079817],
                    [0.20634188, -0.00123839],
                    [0.2668473, -0.11547986],
                ]
            ),
            label_ids=np.array(
                [
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                ]
            ),
            metrics={
                "test_loss": 0.5716606378555298,
                "test_runtime": 0.7568,
                "test_samples_per_second": 66.072,
                "test_steps_per_second": 9.25,
            },
        ),
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ]
        ),
        "binary",
    ),
]


@pytest.mark.parametrize("predictions, expected, problem_type", TESTS)
def test_predict(predictions, expected, problem_type):
    assert np.array_equal(
        predict(predictions, problem_type=problem_type), expected
    ), "The predictions are not correct."
    if problem_type == "multi-label":
        assert (
            type(
                predict(
                    predictions, return_proba=True, problem_type=problem_type
                )
            )
            == torch.Tensor
        ), "Predictions should be a numpy array."
