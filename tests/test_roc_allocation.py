import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from verl.protocol import DataProto


@pytest.fixture(scope="module")
def resample():
    # Load the sampling package without importing the GPU agent/tool registrations.
    path = Path(__file__).parents[1] / "rstar2_agent/down_sample/__init__.py"
    name = "roc_sampling_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module.resample_of_correct
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


@pytest.mark.parametrize(
    "target,min_negative,min_positive,expected",
    [
        (3, 1, 1, [0, 1, 6]),
        (5, 1, 1, [0, 1, 5, 6, 7]),
        (7, 1, 1, [0, 1, 2, 3, 5, 6, 7]),
        (6, 1, 1, [0, 1, 2, 5, 6, 7]),
        (5, 3, 1, [0, 1, 2, 6, 7]),
        (5, 1, 4, [0, 4, 5, 6, 7]),
        (8, 1, 1, list(range(8))),
    ],
)
def test_roc_keeps_the_target_count_and_preferred_positive_traces(
    resample, target, min_negative, min_positive, expected
):
    # Four failures and four successes. Successful traces have penalties 1, .5, 0, 0.
    responses = ["<tool_response>error</tool_response>"] * 5 + [
        "no tool call",
        "<tool_response>2</tool_response>",
        "<tool_response>3</tool_response>",
    ]
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(8).unsqueeze(-1),
            "token_level_scores": torch.tensor([[0.0]] * 4 + [[1.0]] * 4),
        },
        non_tensors={
            "uid": np.array(["question"] * 8, dtype=object),
            "prompt_text": np.array(["question"] * 8, dtype=object),
            "response_text": np.array(responses, dtype=object),
        },
    )
    config = {
        "roc_error_ratio": True,
        "roc_answer_format": False,
        "min_zero_reward_trace_num": min_negative,
        "min_non_zero_reward_trace_num": min_positive,
        "down_sample_to_n": target,
    }

    selected, metrics = resample(batch, None, config, world_size=1)

    assert len(selected) == target
    assert selected.batch["input_ids"].flatten().tolist() == expected
    rewards = selected.batch["token_level_scores"].sum(-1)
    assert (rewards <= 0).sum() >= min_negative
    assert (rewards > 0).sum() >= min_positive
    assert metrics["roc_error_ratio/global_err_ratio"] == pytest.approx(5 / 7)
    assert len(batch) == 8
