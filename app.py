import io

import pandas as pd
import streamlit as st

from categories import all_categories, categorize, parse_custom_rules
from parser import parse_pdf, preview_text

st.set_page_config(page_title="Statement Analyser", page_icon="💳", layout="wide")
st.title("💳 Statement Analyser")
st.caption("Files are processed in memory only and are never saved to disk.")

with st.sidebar:
    st.header("Settings")
    password = st.text_input("PDF password (if protected)", type="password")
    custom_text = st.text_area(
        "Custom category rules",
        placeholder="NETFLIX=Subscriptions\nJUMIA=Shopping\nJOHN DOE=Rent",
        help="One rule per line: KEYWORD=Category. These override the built-in rules.",
    )
    show_raw = st.checkbox("Show raw PDF text (for debugging)")

custom_rules = parse_custom_rules(custom_text)

files = st.file_uploader("Upload PDF statements", type="pdf", accept_multiple_files=True)
if not files:
    st.info("Upload one or more text-based PDF statements to begin.")
    st.stop()

frames = []
for f in files:
    data = f.getvalue()
    try:
        df = parse_pdf(io.BytesIO(data), password or None)
    except Exception as e:
        if "password" in str(e).lower() or type(e).__name__ == "PDFPasswordIncorrect":
            st.error(f"**{f.name}**: password required or incorrect. Enter it in the sidebar.")
        else:
            st.error(f"**{f.name}**: could not read this PDF ({type(e).__name__}: {e})")
        continue

    if show_raw or df.empty:
        with st.expander(f"Raw text of first page: {f.name}", expanded=df.empty):
            st.text(preview_text(io.BytesIO(data), password or None) or "(no text found, likely a scanned PDF)")
    if df.empty:
        st.warning(f"**{f.name}**: no transactions found. It may be scanned (needs OCR) or an unusual layout.")
        continue

    df["source"] = f.name
    frames.append(df)

if not frames:
    st.stop()

df = pd.concat(frames, ignore_index=True)
df["category"] = [
    categorize(d, c, db, custom_rules) for d, c, db in zip(df["description"], df["credit"], df["debit"])
]

# ---------- editable transactions ----------
st.subheader("Transactions")
st.caption("Check these against your PDF. You can fix any category by clicking its cell; everything below updates.")
df = st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    height=350,
    column_config={
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", disabled=True),
        "description": st.column_config.TextColumn("Description", disabled=True, width="large"),
        "debit": st.column_config.NumberColumn("Debit", format="%.2f", disabled=True),
        "credit": st.column_config.NumberColumn("Credit", format="%.2f", disabled=True),
        "balance": st.column_config.NumberColumn("Balance", format="%.2f", disabled=True),
        "source": st.column_config.TextColumn("File", disabled=True),
        "category": st.column_config.SelectboxColumn("Category", options=all_categories(custom_rules)),
    },
)

# ---------- monthly transaction filter ----------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["month"] = df["date"].dt.strftime("%Y-%m")

month_options = sorted(df["month"].unique())

month_labels = {
    month: pd.Period(month, freq="M").strftime("%B %Y")
    for month in month_options
}

selected_month = st.selectbox(
    "Filter transactions by month",
    options=["All months"] + month_options,
    format_func=lambda month: (
        "All months" if month == "All months"
        else month_labels[month]
    ),
)

if selected_month == "All months":
    filtered_df = df.copy()
else:
    filtered_df = df[df["month"] == selected_month].copy()

st.download_button(
    "Download filtered transactions (CSV)",
    filtered_df.drop(columns=["month"], errors="ignore")
    .to_csv(index=False).encode("utf-8"),
    "transactions.csv",
    "text/csv",
)


# ---------- summary ----------
total_in, total_out = df["credit"].sum(), df["debit"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Money in", f"{total_in:,.2f}")
c2.metric("Money out", f"{total_out:,.2f}")
c3.metric("Net", f"{total_in - total_out:,.2f}")
c4.metric("Transactions", f"{len(df):,}")

df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")

left, right = st.columns(2)
with left:
    st.subheader("Spending by category")
    spend = df[df["debit"] > 0].groupby("category")["debit"].sum().sort_values(ascending=False)
    st.bar_chart(spend)
with right:
    st.subheader("Monthly in vs out")
    monthly = df.groupby("month")[["credit", "debit"]].sum().rename(columns={"credit": "In", "debit": "Out"})
    st.bar_chart(monthly)

# ---------- insights ----------
left, right = st.columns(2)
with left:
    st.subheader("Biggest expenses")
    top = df[df["debit"] > 0].nlargest(10, "debit")[["date", "description", "debit", "category"]]
    st.dataframe(top, hide_index=True, use_container_width=True)

with right:
    st.subheader("Likely recurring payments")
    spend_df = df[df["debit"] > 0].copy()
    spend_df["key"] = (
        spend_df["description"].str.upper().str.replace(r"[\d\W_]+", " ", regex=True).str.strip().str[:30]
    )
    rec = (
        spend_df.groupby("key")
        .agg(times=("debit", "size"), months=("month", "nunique"), avg_amount=("debit", "mean"))
        .query("months >= 3")
        .sort_values("times", ascending=False)
        .reset_index()
    )
    if rec.empty:
        st.caption("None found (needs the same merchant in 3+ different months).")
    else:
        st.dataframe(rec, hide_index=True, use_container_width=True)
