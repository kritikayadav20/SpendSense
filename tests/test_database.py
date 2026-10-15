from io import BytesIO, StringIO
import pandas as pd
import pytest
from src.database import Database
from src.importer import prepare_import


def record(**overrides):
    return {'date': '2025-01-02', 'description': 'Cafe lunch', 'amount': 120.25, 'category': 'Meals', **overrides}


def test_crud_correction_budget_export(tmp_path):
    db = Database(tmp_path / 'local.sqlite3')
    identifier = db.add_transaction(record(predicted_category='Other', needs_review=True))
    db.correct_category(identifier, 'Meals')
    data = db.transactions('2025-01')
    assert data.iloc[0].amount == 120.25
    assert data.iloc[0].predicted_category == 'Other'
    assert not data.iloc[0].needs_review
    db.set_budget('2025-01', 1000)
    db.set_budget('2025-01', 1500)
    db.set_budget('2025-01', 300, 'Meals')
    assert db.budgets('2025-01') == {'': 1500, 'Meals': 300}
    exported = pd.read_csv(BytesIO(db.export_csv()))
    assert exported.iloc[0]['category'] == 'Meals'
    assert exported.iloc[0].amount == 120.25
    db.update_transaction(identifier, record(amount=150, description='Updated lunch'))
    assert db.transactions().iloc[0].amount == 150
    assert db.transactions().iloc[0].predicted_category == 'Other'
    db.delete_transaction(identifier)
    assert db.transactions().empty
    with pytest.raises(ValueError):
        db.correct_category(identifier, 'Meals')


def test_atomic_import_duplicate_handling(tmp_path):
    db = Database(tmp_path / 'local.sqlite3')
    assert db.import_transactions([record(), record()]) == {'inserted': 1, 'duplicates': 1}
    db.correct_category(1, 'Dining')
    assert db.import_transactions([record()])['duplicates'] == 1
    with pytest.raises(ValueError):
        db.import_transactions([record(description='New purchase'), record(amount=-5)])
    assert len(db.transactions()) == 1
    # Manual identical purchases remain valid.
    db.add_transaction(record())
    assert len(db.transactions()) == 2


@pytest.mark.parametrize('amount', [float('nan'), float('inf'), -1, 0, True, 'bad', 1e12])
def test_amount_validation(tmp_path, amount):
    with pytest.raises(ValueError):
        Database(tmp_path / 'db').add_transaction(record(amount=amount))


def test_sql_parameterization_and_safe_export(tmp_path):
    db = Database(tmp_path / 'db')
    db.add_transaction(record(description="=1+1", category="Meals'); DROP TABLE transactions;--"))
    exported = pd.read_csv(BytesIO(db.export_csv()))
    assert exported.iloc[0].description == "'=1+1"
    assert len(db.transactions()) == 1


def test_csv_validation_and_preview(trained):
    rows, errors = prepare_import(StringIO('date,description,amount,category\n2025-01-02,Cafe lunch,25,Meals\n2025-01-03,Bus,-5,Transit\n'))
    assert len(rows) == 1 and errors[0]['csv_row'] == 3
    with pytest.raises(ValueError, match='requires columns'):
        prepare_import(StringIO('wrong\nvalue\n'))
    source = 'date,description,amount\n2025-01-02,Cafe lunch,20\n'
    assert prepare_import(StringIO(source))[1]
    rows, errors = prepare_import(StringIO(source), trained[0]['logistic_regression'])
    assert not errors and rows[0]['predicted_category']
