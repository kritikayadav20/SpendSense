import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.predict import load_model, map_features, predict_expense


def test_persistence_and_confidence(trained, tmp_path):
    models, _, _, _ = trained
    for name, model in models.items():
        path = tmp_path / f'{name}.joblib'
        joblib.dump(model, path)
        loaded = load_model(path)
        prediction = predict_expense(loaded, 'cafe sandwich lunch')
        assert prediction['category'] in model.classes_
        assert prediction['classes'] == list(model.classes_)
        if name == 'linear_svm':
            assert prediction['probability'] is None
            assert prediction['decision_margin'] is not None
            assert 'not probability' in prediction['score_label']
        else:
            assert 0 <= prediction['probability'] <= 1
            assert sum(prediction['class_probabilities'].values()) == pytest.approx(1)
            if name == 'decision_tree':
                assert 'poorly calibrated' in prediction['explanation']
        unseen = predict_expense(loaded, 'zzzzquux noveltoken')
        assert unseen['needs_review']
        assert not unseen['recognized_terms']
        assert loaded.predict(['metro ticket'])[0] == model.predict(['metro ticket'])[0]
        with pytest.raises(ValueError):
            predict_expense(loaded, '')


def test_feature_mapping_and_missing_model(tmp_path):
    assert map_features(' cafe  lunch ') == ['cafe lunch']
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / 'missing.joblib')


def test_binary_svm_margin():
    model = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LinearSVC(dual='auto'))])
    model.fit(['bus ticket', 'train ticket', 'cafe meal', 'lunch meal'], ['Travel', 'Travel', 'Food', 'Food'])
    result = predict_expense(model, 'bus ticket')
    assert result['probability'] is None
    assert result['decision_margin'] >= 0
