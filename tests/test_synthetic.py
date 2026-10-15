from src.generate_synthetic import generate_records, write_dataset, COUNTS
from src.data_loader import load_verified_dataset
from src.preprocessing import clean_dataset
from src.train import template_key


def test_synthetic_recipe_reproducible_and_label_safe():
    counts = {category: 10 for category in COUNTS}
    rows, summary = generate_records(seed=42, counts=counts)
    assert (rows, summary) == generate_records(seed=42, counts=counts)
    assert rows != generate_records(seed=43, counts=counts)[0]
    assert len(rows) == sum(counts.values())
    assert set(rows[0]) == {'description', 'category'}
    assert set(row['category'] for row in rows) == set(counts)
    assert all('category:' not in row['description'].casefold() for row in rows)
    assert len({template_key(row['description']) for row in rows}) < len(rows)


def test_synthetic_manifest_and_activation(tmp_path):
    path, manifest = write_dataset(tmp_path)
    assert not (tmp_path / 'data' / 'dataset.json').exists()
    frame, loaded, profile = load_verified_dataset(path)
    assert loaded['provenance'] == 'synthetic'
    assert profile['records'] == sum(COUNTS.values())
    assert sum(profile['missing_values'].values()) == 0
    cleaned, audit = clean_dataset(frame, 'description', 'category')
    assert len(cleaned) == len(frame)
    assert not audit['exclusions']
    write_dataset(tmp_path, activate=True)
    assert (tmp_path / 'data' / 'dataset.json').read_text() == path.read_text()
