from pathlib import Path
from src.report import build_report


def test_pending_report(tmp_path):
    source_docs = Path(__file__).resolve().parents[1] / 'docs'
    report = build_report(tmp_path, docs_root=source_docs)
    text = report.read_text()
    assert 'Results pending' in text
    assert 'Scalable and Weakly Supervised' in text
    assert '| Linear SVM | Pending |' in text
