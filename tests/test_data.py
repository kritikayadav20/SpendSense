import json
import pandas as pd
import pytest
from src.data_loader import inspect_csv, load_verified_dataset, read_csv, validate_schema


def test_schema(raw_fixture):
    validate_schema(raw_fixture, 'description', 'category')
    with pytest.raises(ValueError, match='Missing'):
        validate_schema(raw_fixture, 'merchant', 'category')
    with pytest.raises(ValueError, match='different'):
        validate_schema(raw_fixture, 'category', 'category')
    with pytest.raises(ValueError, match='no records'):
        validate_schema(raw_fixture.iloc[:0], 'description', 'category')


def test_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / 'missing.csv')
    with pytest.raises(FileNotFoundError):
        load_verified_dataset(tmp_path / 'missing.json')


def test_inspection_and_manifest(tmp_path, raw_fixture):
    csv = tmp_path / 'fixture.csv'
    raw_fixture.to_csv(csv, index=False)
    profile = inspect_csv(csv, 'description', 'category')
    assert profile['records'] == 36
    assert profile['headers'] == ['description', 'category']
    manifest = {'file': csv.name, 'description_column': 'description', 'target_column': 'category',
                'source_url': 'unit-test-fixture', 'license': 'test-only', 'license_evidence': 'pytest fixture',
                'provenance': 'synthetic', 'provenance_notes': 'NOT a benchmark', 'taxonomy': ['Meals', 'Transit', 'Books'],
                'taxonomy_evidence': 'fixture', 'feature_definitions': {'description': 'test words'},
                'verified': False, 'sha256': profile['sha256']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Verify source'):
        load_verified_dataset(path)
    manifest['verified'] = True
    path.write_text(json.dumps(manifest))
    frame, _, _ = load_verified_dataset(path)
    assert len(frame) == 36
    manifest['taxonomy'] = ['Wrong', 'Labels']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Undocumented'):
        load_verified_dataset(path)
    csv.write_text(csv.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        load_verified_dataset(path)
