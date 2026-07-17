import pytest

from sbx_h.metrics import voltage_tracking_equity


def test_equal_area_tracking_burdens_have_zero_gini() -> None:
    metric = voltage_tracking_equity({
        1: (0.001, -0.001),
        2: (0.001,),
        3: (-0.001,),
    })

    assert metric.area_rmse_mpu == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
    assert metric.mean_rmse_mpu == pytest.approx(1.0)
    assert metric.worst_rmse_mpu == pytest.approx(1.0)
    assert metric.worst_area == 1
    assert metric.gini == pytest.approx(0.0)
    assert metric.fairness == pytest.approx(1.0)


def test_normalized_gini_quantifies_unequal_tracking_burden() -> None:
    metric = voltage_tracking_equity({
        1: (0.001,),
        2: (0.003,),
        3: (0.008,),
    })

    assert metric.mean_rmse_mpu == pytest.approx(4.0)
    assert metric.worst_rmse_mpu == pytest.approx(8.0)
    assert metric.worst_area == 3
    assert metric.gini == pytest.approx(7.0 / 12.0)
    assert metric.fairness == pytest.approx(5.0 / 12.0)


def test_all_zero_tracking_error_is_defined_as_perfectly_equal() -> None:
    metric = voltage_tracking_equity({1: (0.0,), 2: (0.0,), 3: (0.0,)})

    assert metric.mean_rmse_mpu == 0.0
    assert metric.worst_rmse_mpu == 0.0
    assert metric.gini == 0.0
    assert metric.fairness == 1.0
