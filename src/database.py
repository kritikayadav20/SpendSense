"""Local SQLite persistence. Integer cents, parameterized SQL and explicit transactions."""
from contextlib import contextmanager
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from src.budget import money_to_cents, validate_month
from src.preprocessing import normalize_text, validate_description


def validate_category(value):
    category = normalize_text(value)
    if not category or len(category) > 120:
        raise ValueError("Category must contain 1–120 characters.")
    return category


def validate_transaction(record):
    try:
        raw_date = str(record["date"])
        when = date.fromisoformat(raw_date)
        if when.isoformat() != raw_date:
            raise ValueError("Use YYYY-MM-DD dates.")
        if when > date.today():
            raise ValueError("Record completed expenses only; future dates are not supported.")
        description = validate_description(record["description"])
        amount = money_to_cents(record["amount"])
        category = validate_category(record["category"])
    except KeyError as exc:
        raise ValueError(f"Missing transaction field: {exc.args[0]}") from None
    return {"date": when.isoformat(), "description": description, "amount_cents": amount,
            "category": category, "predicted_category": record.get("predicted_category"),
            "needs_review": bool(record.get("needs_review", False)), "model_name": record.get("model_name")}


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY, date TEXT NOT NULL, description TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL CHECK(amount_cents > 0), category TEXT NOT NULL,
                    predicted_category TEXT, needs_review INTEGER NOT NULL DEFAULT 0,
                    model_name TEXT, import_key TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS transactions_date ON transactions(date);
                CREATE TABLE IF NOT EXISTS budgets (
                    month TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',
                    amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
                    PRIMARY KEY(month, category)
                );
            """)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _insert(con, record, imported=False):
        key = None
        if imported:
            key = hashlib.sha256(json.dumps([record["date"], record["description"].casefold(), record["amount_cents"]]).encode()).hexdigest()
        cursor = con.execute("""INSERT INTO transactions
            (date, description, amount_cents, category, predicted_category, needs_review, model_name, import_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(import_key) DO NOTHING""",
            (record["date"], record["description"], record["amount_cents"], record["category"],
             record["predicted_category"], int(record["needs_review"]), record["model_name"], key))
        return cursor.lastrowid if cursor.rowcount else None

    def add_transaction(self, record):
        valid = validate_transaction(record)
        with self.connection() as con:
            return self._insert(con, valid)

    def import_transactions(self, records):
        # Validate whole batch first: one invalid row causes zero writes.
        valid = [validate_transaction(record) for record in records]
        inserted = 0
        with self.connection() as con:
            for record in valid:
                inserted += self._insert(con, record, imported=True) is not None
        return {"inserted": inserted, "duplicates": len(valid) - inserted}

    def transactions(self, month=None):
        query, params = "SELECT * FROM transactions", []
        if month is not None:
            validate_month(month)
            query += " WHERE substr(date, 1, 7) = ?"
            params = [month]
        query += " ORDER BY date DESC, id DESC"
        with self.connection() as con:
            frame = pd.read_sql_query(query, con, params=params)
        frame["amount"] = frame.amount_cents / 100
        frame["needs_review"] = frame.needs_review.astype(bool)
        return frame

    def correct_category(self, transaction_id, category, needs_review=False):
        category = validate_category(category)
        with self.connection() as con:
            cursor = con.execute("UPDATE transactions SET category=?, needs_review=? WHERE id=?",
                                 (category, int(needs_review), int(transaction_id)))
            if not cursor.rowcount:
                raise ValueError("Transaction does not exist.")

    def update_transaction(self, transaction_id, record):
        valid = validate_transaction(record)
        with self.connection() as con:
            cursor = con.execute("""UPDATE transactions SET date=?, description=?, amount_cents=?,
                category=?, needs_review=? WHERE id=?""", (valid["date"], valid["description"],
                valid["amount_cents"], valid["category"], int(valid["needs_review"]), int(transaction_id)))
            if not cursor.rowcount:
                raise ValueError("Transaction does not exist.")

    def delete_transaction(self, transaction_id):
        with self.connection() as con:
            cursor = con.execute("DELETE FROM transactions WHERE id=?", (int(transaction_id),))
            if not cursor.rowcount:
                raise ValueError("Transaction does not exist.")

    def set_budget(self, month, amount, category=""):
        validate_month(month)
        cents = money_to_cents(amount, allow_zero=True)
        if category:
            category = validate_category(category)
        with self.connection() as con:
            con.execute("""INSERT INTO budgets(month, category, amount_cents) VALUES (?, ?, ?)
                ON CONFLICT(month, category) DO UPDATE SET amount_cents=excluded.amount_cents""", (month, category, cents))

    def budgets(self, month):
        validate_month(month)
        with self.connection() as con:
            return {category: cents / 100 for category, cents in con.execute(
                "SELECT category, amount_cents FROM budgets WHERE month=?", (month,))}

    def export_csv(self, frame=None):
        data = self.transactions() if frame is None else frame.copy()
        data = data[["date", "description", "amount", "category", "needs_review"]].copy()
        # Avoid spreadsheet formula execution. Export is for viewing; prefixed strings
        # need deliberate unescaping before a lossless re-import.
        for col in ["description", "category"]:
            data[col] = data[col].map(lambda x: "'" + x if x.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else x)
        return data.to_csv(index=False).encode("utf-8-sig")
