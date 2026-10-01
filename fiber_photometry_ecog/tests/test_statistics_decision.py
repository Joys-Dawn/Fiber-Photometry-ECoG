import numpy as np

from fiber_photometry_ecog.statistics.decision import should_use_parametric


def test_parametric_requires_every_group_to_pass_shapiro() -> None:
    normal = np.array([
        -1.8, -1.4, -1.1, -0.9, -0.6, -0.4, -0.2, -0.1,
        0.0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.1, 1.4, 1.8,
    ])
    shapiro_only_failure = np.array([
        15.412414, 21.755562, 24.406371, 27.684712,
        37.668511, 39.182467, 150.240446,
    ])

    assert should_use_parametric([normal])
    assert not should_use_parametric([normal, shapiro_only_failure])


def test_shapiro_p_equal_to_alpha_does_not_fail(monkeypatch) -> None:
    class Result:
        pvalue = 0.05

    monkeypatch.setattr(
        "fiber_photometry_ecog.statistics.decision.shapiro",
        lambda _values: Result(),
    )

    assert should_use_parametric([np.array([-1.0, 0.0, 1.0])])
