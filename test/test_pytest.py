import pytest
from src import preprocessing as pp


def test_mean():
    assert pp.mean([2, 4, 6, 8]) == 5


def test_std():
    assert pp.std([2, 4, 6, 8]) == pytest.approx(2.2360679, rel=1e-6)


def test_min_max_scale():
    assert pp.min_max_scale([2, 4, 6, 8]) == pytest.approx([0, 1/3, 2/3, 1])


def test_z_score():
    z = pp.z_score([2, 4, 6, 8])
    assert pp.mean(z) == pytest.approx(0)
    assert pp.std(z) == pytest.approx(1)


def test_train_test_split():
    train, test = pp.train_test_split(list(range(1, 11)), 0.2)
    assert train == [1, 2, 3, 4, 5, 6, 7, 8]
    assert test == [9, 10]


def test_empty_raises():
    with pytest.raises(ValueError):
        pp.mean([])


def test_load_csv_column(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("age,salary\n25,50000\n35,70000\n")
    assert pp.load_csv_column(f, "age") == [25.0, 35.0]


@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3], 2),
    ([10], 10),
    ([-5, 5], 0),
])
def test_mean_param(values, expected):
    assert pp.mean(values) == expected
