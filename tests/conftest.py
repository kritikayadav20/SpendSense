"""Synthetic fixtures are confined to tests and temporary directories."""
import pandas as pd
import pytest

from src.preprocessing import clean_dataset, split_dataset
from src.train import fit_models


@pytest.fixture
def raw_fixture():
    rows = []
    for label, words in [('Meals', 'cafe lunch sandwich'), ('Transit', 'bus metro ticket'), ('Books', 'library textbook paper')]:
        for i in range(12):
            rows.append({'description': f'{words} location{i}', 'category': label})
    return pd.DataFrame(rows)


@pytest.fixture(scope='session')
def trained():
    rows = [{'description': f'{text} variant{i}', 'category': label}
            for label, text in [('Meals', 'cafe lunch sandwich'), ('Transit', 'bus metro ticket'), ('Books', 'library textbook paper')]
            for i in range(12)]
    clean, _ = clean_dataset(pd.DataFrame(rows), 'description', 'category')
    train, test = split_dataset(clean)
    models, info = fit_models(train)
    return models, info, train, test
