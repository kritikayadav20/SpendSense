"""SpendSense Campus — launch with streamlit run app.py."""
import calendar
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.budget import budget_summary
from src.data_loader import ROOT
from src.database import Database
from src.importer import prepare_import
from src.predict import load_model, predict_expense
from src.train import MODEL_NAMES

st.set_page_config(page_title="SpendSense Campus", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width:1280px;padding-top:2.5rem;padding-bottom:3rem}
h1 {letter-spacing:-1.4px;font-weight:750!important} h2,h3 {letter-spacing:-.5px}
[data-testid="stMetric"] {background:white;border:1px solid #DCE5EE;border-radius:14px;padding:18px}
[data-testid="stMetricLabel"] {color:#526579}
[data-testid="stSidebar"] {border-right:1px solid #DCE5EE}
div.stButton>button {border-radius:9px}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#147D73;margin-bottom:6px}
.intro {color:#526579;font-size:17px;max-width:760px;margin-bottom:28px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path, modified):
    return load_model(path)


@st.cache_data
def cached_json(path, modified):
    return json.loads(Path(path).read_text())


def read_metadata(path):
    return cached_json(str(path), path.stat().st_mtime_ns)


def readiness():
    pointer = ROOT / "models" / "active.json"
    if not pointer.exists():
        return None, None, "Dataset and model setup pending"
    try:
        run = read_metadata(pointer)["run_id"]
        metadata = read_metadata(ROOT / "models" / run / "metadata.json")
        path = ROOT / "models" / run / f"{metadata['selected_model']}.joblib"
        status = "Synthetic demo model ready" if metadata["dataset"]["provenance"] == "synthetic" else "Trained model ready"
        return metadata, cached_model(str(path), path.stat().st_mtime_ns), status
    except Exception:
        return None, None, "Model artifacts could not be loaded; rerun the documented training workflow"


db = Database(os.environ.get("SPENDSENSE_DB", str(ROOT / "data" / "spendsense.sqlite3")))
metadata, model, status = readiness()
today = date.today()
with st.sidebar:
    st.markdown("## ◈ SpendSense")
    st.caption("CAMPUS · YOUR MONEY, IN FOCUS")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Add & Import", "Transactions & Analytics", "Budget Coach", "ML Lab", "Methodology"], label_visibility="collapsed")
    st.divider()
    month_date = st.date_input("Reporting month", value=today.replace(day=1), help="Any date selects its calendar month.")
    month = month_date.strftime("%Y-%m")
    st.caption(f"Selected: {month_date.strftime('%B %Y')}")
    if model is not None:
        st.success(status)
    else:
        st.info(status)
    st.caption("Local SQLite storage · INR only\n\nNo bank connection. Keep account numbers and personal identifiers out of descriptions.")

if metadata and metadata["dataset"]["provenance"] == "synthetic":
    st.info("Synthetic-data demonstration: categorization uses generated examples. Benchmark scores measure this synthetic corpus.")

all_transactions = db.transactions()
monthly = all_transactions[all_transactions.date.str.startswith(month)]
budgets = db.budgets(month)
categories = sorted(set(all_transactions.category) | set(metadata["classes"] if metadata else []))


def money(value):
    return "—" if value is None else f"₹{value:,.2f}"


def heading(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="intro">{subtitle}</div>', unsafe_allow_html=True)


def show_table(frame):
    cols = [c for c in ["id", "date", "description", "amount", "category", "needs_review"] if c in frame]
    st.dataframe(frame[cols], use_container_width=True, hide_index=True,
                 column_config={"amount": st.column_config.NumberColumn("Amount (₹)", format="%.2f"),
                                "needs_review": st.column_config.CheckboxColumn("Needs review")})


def category_chart(frame):
    grouped = frame.groupby("category", as_index=False).amount.sum().sort_values("amount")
    fig = px.bar(grouped, x="amount", y="category", orientation="h", color_discrete_sequence=["#147D73"],
                 labels={"amount": "Expenditure (₹)", "category": ""})
    fig.update_layout(margin=dict(l=0, r=0, t=15, b=0), height=max(300, len(grouped) * 35), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def budget_alerts():
    if not budgets:
        st.info("Set a monthly budget in Budget Coach to see your remaining balance and spending pace.")
    for category, limit in budgets.items():
        spent = monthly.amount.sum() if not category else monthly.loc[monthly.category.eq(category), "amount"].sum()
        summary = budget_summary(spent, limit, month)
        for alert in summary["alerts"]:
            st.warning(f"{category or 'Overall'}: {alert}")


if page == "Overview":
    heading("STUDENT FINANCE, SIMPLIFIED", "Make room for what matters.", "Track the everyday. Understand your spending. Plan the rest of your month with confidence.")
    cols = st.columns(4)
    total = float(monthly.amount.sum())
    summary = budget_summary(total, budgets[""], month) if "" in budgets else None
    cols[0].metric("Recorded this month", money(total))
    cols[1].metric("Remaining budget", money(summary["remaining_budget"]) if summary else "Not set")
    cols[2].metric("Safe to spend / day", money(summary["safe_to_spend_per_day"]) if summary else "Not set")
    cols[3].metric("Expenses to review", int(monthly.needs_review.sum()))
    st.caption("Safe-to-spend uses the configured budget and recorded expenses only. Missing expenses will overstate the balance.")
    budget_alerts()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Where your money goes")
        if monthly.empty:
            st.info("Your month starts here. Add an expense or import a CSV to reveal your spending breakdown.")
        else:
            category_chart(monthly)
    with right:
        st.subheader("Recent activity")
        if monthly.empty:
            st.caption("No transactions recorded for this month.")
        else:
            show_table(monthly.head(7))
    with st.expander("Dataset & model readiness", expanded=model is None):
        if metadata:
            st.write(f"Active model: **{MODEL_NAMES[metadata['selected_model']]}** · selected by training CV macro-F1.")
            st.write(f"Verified source manifest: {metadata['dataset']['source_url']}")
            st.caption(f"Provenance: {metadata['dataset']['provenance']} · Run: {metadata['run_id']}")
        else:
            st.write("No verified dataset or completed benchmark is bundled. Manual tracking and budgets work now. Automatic categorization becomes available after training.")
            st.code("python -m src.data_loader data/raw/YOUR_FILE.csv\n# Follow data/README.md to verify the source manifest\npython -m src.train")

elif page == "Add & Import":
    heading("BUILD YOUR SPENDING PICTURE", "A little detail. A lot of clarity.", "Record a purchase, check its category, and keep your history in one place.")
    manual, upload_tab = st.tabs(["Add an expense", "Import a CSV"])
    with manual:
        left, right = st.columns([1.2, 1])
        with left:
            description = st.text_input("Transaction description", max_chars=1000, placeholder="A short merchant or purchase description")
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount (₹)", min_value=0.01, max_value=1000000000.0, value=100.0, step=10.0)
            when = c2.date_input("Expense date", value=today, max_value=today)
            if st.button("Suggest category", disabled=model is None):
                try:
                    st.session_state["prediction"] = {"description": description, "run_id": metadata["run_id"], **predict_expense(model, description)}
                except ValueError as exc:
                    st.error(str(exc))
            prediction = st.session_state.get("prediction")
            if prediction and (prediction["description"] != description or metadata is None or prediction["run_id"] != metadata["run_id"]):
                prediction = None
            options = sorted(set(categories) | ({prediction["category"]} if prediction else set()))
            if model is not None:
                category = st.selectbox("Category (editable before saving)", options,
                                        index=options.index(prediction["category"]) if prediction else 0,
                                        key="category_" + hashlib.sha256((description + str(bool(prediction))).encode()).hexdigest()[:12])
            else:
                category = st.text_input("Personal category", placeholder="Enter your own bookkeeping label")
                st.caption("Personal labels are for your ledger. No model taxonomy has been verified yet.")
            reviewed = st.checkbox("I have checked the category", value=False)
            if st.button("Save expense", type="primary"):
                try:
                    db.add_transaction({"date": when.isoformat(), "description": description, "amount": amount,
                        "category": category, "predicted_category": prediction["category"] if prediction else None,
                        "needs_review": not reviewed, "model_name": metadata["selected_model"] if prediction else None})
                    st.success("Expense saved locally. View it in Transactions & Analytics.")
                except ValueError as exc:
                    st.error(str(exc))
        with right:
            st.subheader("Category assistant")
            if prediction:
                st.write(f"Suggested category: **{prediction['category']}**")
                if prediction["probability"] is not None:
                    st.metric(prediction["score_label"], f"{prediction['probability']:.1%}")
                else:
                    st.metric(prediction["score_label"], f"{prediction['decision_margin']:.3f}")
                st.caption(prediction["explanation"])
                if prediction["needs_review"]:
                    st.warning("Needs review: " + " ".join(prediction["reasons"]))
                else:
                    st.info("Check the suggestion before saving; model estimates can be wrong.")
                st.caption("Recognized TF-IDF terms (feature coverage, not a causal explanation): " + ", ".join(prediction["recognized_terms"]))
            elif model is None:
                st.info("Prediction is pending a verified training dataset. You can still enter a category and save your expense.")
            else:
                st.caption("Enter a description and select Suggest category. Amount and date do not enter the text model.")
    with upload_tab:
        st.write("Use **date, description, amount** and optionally **category, needs_review**. Dates use YYYY-MM-DD; amounts are positive INR expenses. A category is required when no model is available.")
        st.download_button("Download CSV headers", b"date,description,amount,category\n", "expense-template.csv", "text/csv")
        uploaded = st.file_uploader("Expense CSV", type="csv")
        st.caption("Imports skip repeated date + normalized description + amount combinations, even after category correction. Identical genuine purchases should be entered manually. Preview edits are validated before any rows are saved.")
        if uploaded is not None:
            try:
                content = uploaded.getvalue()
                rows, errors = prepare_import(io.BytesIO(content), model, metadata["selected_model"] if metadata else None)
                if errors:
                    st.error("Fix these rows in the source CSV and upload it again. No rows have been saved.")
                    st.dataframe(pd.DataFrame(errors), hide_index=True)
                elif rows:
                    edited = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                                            disabled=["predicted_category", "model_name"],
                                            key="import_" + hashlib.sha256(content).hexdigest())
                    if st.button("Save validated import", type="primary"):
                        result = db.import_transactions(edited.to_dict("records"))
                        st.success(f"Saved {result['inserted']} expenses; skipped {result['duplicates']} duplicate rows.")
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                st.error(str(exc))

elif page == "Transactions & Analytics":
    heading("EVERYDAY PATTERNS", "Your spending, in perspective.", "Search your local history, review categories, and explore the trends behind your totals.")
    if all_transactions.empty:
        st.info("No expenses yet. Add your first transaction to start your history.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        search = c1.text_input("Search descriptions", placeholder="Search merchant or purchase")
        chosen_categories = c2.multiselect("Categories", categories)
        only_review = c3.checkbox("Needs review only")
        c1, c2 = st.columns(2)
        start = c1.date_input("From", value=month_date.replace(day=1))
        last_day = month_date.replace(day=calendar.monthrange(month_date.year, month_date.month)[1])
        end = c2.date_input("Through", value=min(today, last_day) if month <= today.strftime("%Y-%m") else last_day)
        filtered = all_transactions[all_transactions.date.between(start.isoformat(), end.isoformat())]
        filtered = filtered[filtered.description.str.contains(search, case=False, regex=False)]
        if chosen_categories:
            filtered = filtered[filtered.category.isin(chosen_categories)]
        if only_review:
            filtered = filtered[filtered.needs_review]
        if start > end:
            st.error("From date must be on or before Through date.")
        st.caption(f"{len(filtered)} matching transactions · {money(filtered.amount.sum())}")
        show_table(filtered)
        st.download_button("Export filtered history", db.export_csv(filtered), "spendsense-expenses.csv", "text/csv")
        if not filtered.empty:
            with st.expander("Correct a category or mark an expense reviewed"):
                record_id = st.selectbox("Transaction ID", filtered.id.tolist())
                current = filtered[filtered.id.eq(record_id)].iloc[0]
                st.caption(f"{current['date']} · {current.description} · {money(current.amount)}")
                correction = st.text_input("Correct category", value=current.category, key=f"correction_{record_id}")
                if st.button("Save category correction"):
                    try:
                        db.correct_category(record_id, correction)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            left, right = st.columns(2)
            with left:
                st.subheader("Daily spending")
                daily = filtered.groupby("date", as_index=False).amount.sum()
                st.plotly_chart(px.bar(daily, x="date", y="amount", color_discrete_sequence=["#147D73"], labels={"amount": "₹"}), use_container_width=True)
            with right:
                st.subheader("Category breakdown")
                category_chart(filtered)
        st.subheader("Monthly spending · all recorded history")
        trend = all_transactions.assign(month=all_transactions.date.str[:7]).groupby("month", as_index=False).amount.sum()
        st.plotly_chart(px.bar(trend, x="month", y="amount", color_discrete_sequence=["#687CBA"], labels={"amount": "₹"}), use_container_width=True)

elif page == "Budget Coach":
    heading("A PLAN FOR THE REST OF THE MONTH", "Spend with breathing room.", "See your remaining balance, daily spending room, and pace-based month-end estimate.")
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Set your limits")
        with st.form("overall_budget"):
            limit = st.number_input("Monthly budget (₹)", min_value=0.0, max_value=1000000000.0, value=float(budgets.get("", 0)), step=500.0)
            if st.form_submit_button("Save monthly budget"):
                db.set_budget(month, limit)
                st.rerun()
        with st.form("category_budget"):
            budget_category = st.text_input("Category to budget", help="Use the same category spelling as your transactions.")
            category_limit = st.number_input("Category limit (₹)", min_value=0.0, max_value=1000000000.0, step=100.0)
            if st.form_submit_button("Save category budget"):
                try:
                    if not budget_category.strip():
                        raise ValueError("Enter a category for this limit.")
                    db.set_budget(month, category_limit, budget_category)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if categories:
            st.caption("Recorded/model categories: " + ", ".join(categories))
    with right:
        st.subheader(month_date.strftime("%B %Y") + " · budget pulse")
        if "" not in budgets:
            st.info("Set an overall budget to calculate daily spending room.")
        else:
            summary = budget_summary(float(monthly.amount.sum()), budgets[""], month)
            c1, c2 = st.columns(2)
            c1.metric("Remaining balance", money(summary["remaining_budget"]))
            c2.metric("Overspent", money(summary["overspend"]))
            c1.metric("Safe to spend / day", money(summary["safe_to_spend_per_day"]))
            c2.metric("Pace-based month-end estimate", money(summary["projected_month_end_spend"]))
            st.caption(f"{summary['remaining_days']} calendar days remaining, including today in the current month. Past months have no daily allowance; future months have no spending projection.")
            if budgets[""] > 0:
                st.progress(min(1.0, summary["spent"] / budgets[""]), text=f"{summary['spent'] / budgets['']:.0%} of budget used")
            st.write(f"Spending drift versus an even daily budget: **{money(summary['drift'])}**.")
        budget_alerts()
    if any(budgets):
        st.subheader("Category limits")
        rows = []
        for category, limit in budgets.items():
            if category:
                spent = float(monthly.loc[monthly.category.eq(category), "amount"].sum())
                rows.append({"Category": category, "Budget (₹)": limit, "Spent (₹)": spent, "Remaining (₹)": max(0, limit - spent), "Overspent (₹)": max(0, spent - limit)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if "" in budgets and sum(v for k, v in budgets.items() if k) > budgets[""]:
            st.warning("Category limits add up to more than the overall budget. Both limits are tracked independently.")
    with st.expander("Show the arithmetic"):
        st.code("remaining = max(0, budget - spending)\noverspend = max(0, spending - budget)\nsafe_per_day = remaining / remaining_calendar_days\nmonth_end_estimate = (spending / elapsed_calendar_days) * days_in_month\ndrift = spending - budget * elapsed_calendar_days / days_in_month")
        st.write("The current day counts in both elapsed and remaining days. The estimate assumes the recorded spending pace continues; rent, fees and other irregular expenses can make it inaccurate. Missing transactions also distort it. Category budgets are independent limits.")

elif page == "ML Lab":
    heading("EVIDENCE BEFORE CONFIDENCE", "Four models. One fair comparison.", "Inspect executed experiments, category-level performance, and the trade-offs behind the selected classifier.")
    st.dataframe(pd.DataFrame({"Required model": list(MODEL_NAMES.values()), "Features": ["Training-fitted TF-IDF"] * 4,
        "Output": ["Uncalibrated probability", "Uncalibrated probability", "Decision score / margin", "Leaf probability estimate"]}), hide_index=True, use_container_width=True)
    if metadata is None:
        st.info("Benchmark pending. No verified dataset has been trained, so there are no performance scores to display.")
        st.code("python -m src.train\npython -m src.evaluate")
        st.write("See data/README.md for dataset verification. Synthetic unit-test fixtures never appear here.")
    else:
        run = metadata["run_id"]
        metrics_dir = ROOT / "reports" / "metrics" / run
        figures_dir = ROOT / "reports" / "figures" / run
        try:
            comparison = pd.DataFrame(read_metadata(metrics_dir / "comparison.json"))
            details = read_metadata(metrics_dir / "details.json")
            st.success(f"Selected: {MODEL_NAMES[metadata['selected_model']]} · training CV macro-F1")
            st.caption(metadata["selection_criterion"])
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            chart = comparison.melt(id_vars="display_name", value_vars=["macro_f1", "accuracy"], var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(chart, x="display_name", y="Score", color="Metric", barmode="group", range_y=[0, 1], color_discrete_sequence=["#147D73", "#687CBA"]), use_container_width=True)
            grouped = metadata.get("grouped_evaluation", {})
            if grouped.get("status") == "completed":
                st.subheader("Unseen-template check")
                grouped_rows = [{"Model": MODEL_NAMES[name], "Grouped test macro-F1": result["metrics"]["macro_f1"],
                                 "Grouped test accuracy": result["metrics"]["accuracy"]}
                                for name, result in grouped["models"].items()]
                st.dataframe(pd.DataFrame(grouped_rows), hide_index=True, use_container_width=True)
                st.caption("A separate split keeps descriptions with matching numeric/punctuation-normalized templates together and tunes only its own training groups. Shared generator vocabulary remains; this is not real-world or unseen-merchant validation.")
            choice = st.selectbox("Inspect model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
            left, right = st.columns(2)
            with left:
                st.image(str(figures_dir / f"{choice}_confusion.png"), use_container_width=True)
            with right:
                st.dataframe(pd.DataFrame(details[choice]["per_class"]).T, use_container_width=True)
                st.image(str(figures_dir / f"{choice}_per_class.png"), use_container_width=True)
            with st.expander("Misclassifications · descriptions withheld"):
                st.dataframe(pd.read_csv(metrics_dir / f"{choice}_errors.csv"), hide_index=True)
            with st.expander("Decision Tree · first three levels"):
                st.image(str(figures_dir / "decision_tree.png"), use_container_width=True)
                st.code((figures_dir / "decision_tree_rules.txt").read_text())
            with st.expander("Dataset profile, preprocessing & evaluation"):
                st.json({"source": metadata["dataset"], "profile": metadata["profile"], "cleaning": metadata["cleaning"], "split": metadata["split"], "grouped_evaluation": metadata["grouped_evaluation"]})
            st.caption("Macro-F1 weights each category equally. Training time includes tuning and refit; latency is batch wall time per record. Tree splits can struggle with sparse text. High scores do not establish generalization to unseen merchants or colleges.")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Benchmark artifacts are incomplete: {exc}. Run python -m src.evaluate.")

elif page == "Methodology":
    heading("BUILT TO BE UNDERSTOOD", "The method behind the numbers.", "An explainable college ML project with reproducible experiments and transparent budget arithmetic.")
    for label, filename in [("Methodology", "methodology.md"), ("Dataset card", "dataset_card.md"), ("Model card", "model_card.md"), ("Literature & references", "literature_review.md")]:
        with st.expander(label, expanded=label == "Methodology"):
            file = ROOT / "docs" / filename
            st.markdown(file.read_text() if file.exists() else "Documentation is being prepared.")

st.divider()
st.caption("SpendSense Campus · ML-powered categorization, transparent budget coaching · Your records stay on this machine.")
