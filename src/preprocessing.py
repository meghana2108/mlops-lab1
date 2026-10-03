import csv
import math


def mean(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def min_max_scale(values):
    if not values:
        raise ValueError("values must not be empty")
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0.0 for _ in values]
    return [(v - lo) / hi for v in values]


def z_score(values):
    m, s = mean(values), std(values)
    if s == 0:
        return [0.0 for _ in values]
    return [(v - m) / s for v in values]


def train_test_split(data, test_ratio=0.2):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


def load_csv_column(path, column):
    with open(path, newline="") as f:
        return [float(row[column]) for row in csv.DictReader(f)]
