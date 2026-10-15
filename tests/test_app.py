from datetime import date
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.database import Database


def test_all_pages_and_manual_save(tmp_path, monkeypatch):
    db_path = tmp_path / 'app.sqlite3'
    monkeypatch.setenv('SPENDSENSE_DB', str(db_path))
    # Empty-state coverage must not depend on the developer's active model.
    import src.data_loader
    monkeypatch.setattr(src.data_loader, 'ROOT', tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    for page in ['Add & Import', 'Transactions & Analytics', 'Budget Coach', 'ML Lab', 'Methodology', 'Overview']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio[0].set_value('Add & Import').run()
    next(x for x in app.text_input if x.label == 'Transaction description').set_value('Test lunch')
    next(x for x in app.text_input if x.label == 'Personal category').set_value('Meals')
    next(x for x in app.button if x.label == 'Save expense').click().run()
    assert not app.exception
    assert len(Database(db_path).transactions()) == 1
    Database(db_path).set_budget(date.today().strftime('%Y-%m'), 1000)
    for page in ['Overview', 'Transactions & Analytics', 'Budget Coach']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
