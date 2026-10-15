import pandas as pd
import pytest
from src.preprocessing import clean_dataset, normalize_text, split_dataset, validate_description


def test_normalization():
    assert normalize_text('  Café\tUPI-42  lunch\n') == 'Café UPI-42 lunch'
    assert normalize_text(None) == ''
    assert normalize_text(42) == ''
    assert normalize_text('ＰＡＹ') == 'PAY'


@pytest.mark.parametrize('value', ['', '  ', None, 5, '!!!', 'x' * 1001])
def test_invalid_descriptions(value):
    with pytest.raises(ValueError):
        validate_description(value)


def test_missing_conflicts_and_duplicates():
    frame = pd.DataFrame({'d': ['Cafe', ' CAFE ', 'bus', 'BUS', None, '  ', 'book', '!!'],
                          'y': ['Food', 'Food', 'Transit', 'Other', 'Food', 'Food', None, 'Food']})
    clean, audit = clean_dataset(frame, 'd', 'y')
    assert clean.description.tolist() == ['Cafe']
    assert audit['exclusion_counts'] == {'missing_or_invalid_text_or_label': 4, 'conflicting_labels': 2, 'duplicate_description': 1}


def test_leakage_safe_split(raw_fixture):
    clean, _ = clean_dataset(raw_fixture, 'description', 'category')
    train, test = split_dataset(clean)
    assert not set(train.description_key) & set(test.description_key)
    assert not set(train.source_row) & set(test.source_row)
    assert set(train.category) == set(test.category)
    assert len(train) + len(test) == len(clean)
    assert test.equals(split_dataset(clean)[1])
    with pytest.raises(ValueError, match='five unique'):
        split_dataset(clean.iloc[:3])
