import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.data_loader import inspect_csv
from src.evaluate import active_run, evaluate_run, metrics_for_model
from src.train import MODEL_NAMES, train_benchmark


def test_four_models_metrics_and_leakage(trained):
    models, info, train, test = trained
    assert set(models) == set(MODEL_NAMES)
    expected = {'logistic_regression': 'LogisticRegression', 'multinomial_nb': 'MultinomialNB',
                'linear_svm': 'LinearSVC', 'decision_tree': 'DecisionTreeClassifier'}
    for name, model in models.items():
        assert type(model.named_steps['classifier']).__name__ == expected[name]
        assert model.named_steps['tfidf'].transform(train.description).min() >= 0
        vocab = model.named_steps['tfidf'].vocabulary_
        training_words = set(' '.join(train.description).split())
        test_only = set(' '.join(test.description).split()) - training_words
        assert not test_only & set(vocab)
        metrics, predicted = metrics_for_model(model, test)
        assert len(predicted) == len(test)
        assert set(['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall',
                    'weighted_precision', 'weighted_recall', 'prediction_ms_per_record']) <= set(metrics)
        assert 0 <= metrics['macro_f1'] <= 1
        assert info[name]['cv_folds'] >= 2


def test_real_artifact_workflow_only_in_temporary_directory(tmp_path, raw_fixture, monkeypatch):
    # Explicit synthetic test fixture, never written into the project's benchmark directories.
    # Six merchant-like template groups per class, two numeric variants each.
    raw_fixture['description'] = [text + ' vendor' + chr(97 + (i % 12) // 2) for i, text in enumerate(raw_fixture.description)]
    csv = tmp_path / 'unit_fixture.csv'
    raw_fixture.to_csv(csv, index=False)
    manifest = {'file': csv.name, 'description_column': 'description', 'target_column': 'category',
        'source_url': 'unit-test-only', 'license': 'test-only', 'license_evidence': 'fixture in tests',
        'provenance': 'synthetic', 'provenance_notes': 'Unit test, not reported benchmark',
        'taxonomy': ['Meals', 'Transit', 'Books'], 'taxonomy_evidence': 'test fixture',
        'feature_definitions': {'description': 'synthetic words'}, 'verified': True,
        'sha256': inspect_csv(csv)['sha256']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    result = train_benchmark(path, root=tmp_path)
    run = result['run_id']
    assert active_run(tmp_path) == run
    assert result['grouped_evaluation']['status'] == 'completed'
    from src.train import template_key
    grouped = result['grouped_evaluation']
    tr_keys = {template_key(raw_fixture.iloc[i - 2].description) for i in grouped['train_source_rows']}
    te_keys = {template_key(raw_fixture.iloc[i - 2].description) for i in grouped['test_source_rows']}
    assert not tr_keys & te_keys
    assert set(grouped['models']) == set(MODEL_NAMES)
    assert result['selected_model'] == max(result['models'], key=lambda name: result['models'][name]['cv_macro_f1'])
    comparison = evaluate_run(tmp_path)
    assert len(comparison) == 4
    assert len(list((tmp_path / 'models' / run).glob('*.joblib'))) == 4
    for name in MODEL_NAMES:
        details = json.loads((tmp_path / 'reports' / 'metrics' / run / 'details.json').read_text())[name]
        assert np.asarray(details['confusion_matrix']).sum() == result['split']['test_rows']
        assert (tmp_path / 'reports' / 'figures' / run / f'{name}_confusion.png').is_file()
        errors = pd.read_csv(tmp_path / 'reports' / 'metrics' / run / f'{name}_errors.csv')
        assert errors.empty or set(errors.description) == {'[withheld for privacy]'}
    from pathlib import Path
    from src.report import build_report
    project = Path(__file__).resolve().parents[1]
    report = build_report(tmp_path, docs_root=project / 'docs').read_text()
    assert 'Results pending' not in report and run in report
    # Verify the app against this completed test run without publishing fixture
    # artifacts or ledger records into the real project.
    import src.data_loader
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(src.data_loader, 'ROOT', tmp_path)
    monkeypatch.setenv('SPENDSENSE_DB', str(tmp_path / 'ui.sqlite3'))
    app = AppTest.from_file(str(project / 'app.py'), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value('ML Lab').run()
    assert not app.exception
    assert any('Selected:' in message.value for message in app.success)
    assert any('Synthetic-data demonstration' in message.value for message in app.info)
    app.sidebar.radio[0].set_value('Add & Import').run()
    next(x for x in app.text_input if x.label == 'Transaction description').set_value('cafe lunch sandwich')
    next(x for x in app.button if x.label == 'Suggest category').click().run()
    assert not app.exception
    assert any('probability' in metric.label.lower() or 'margin' in metric.label.lower() for metric in app.metric)
    test_csv = tmp_path / 'data' / 'processed' / run / 'test.csv'
    test_csv.write_text(test_csv.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        evaluate_run(tmp_path)


def test_missing_evaluation(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_run(tmp_path)
