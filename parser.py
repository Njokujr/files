"""Parse PDF bank statements into one normalised DataFrame.

Output columns: date, description, debit, credit, balance
Strategy: 1) try pdfplumber table extraction, 2) fall back to line-by-line text parsing.
"""
import re
import warnings

import pandas as pd

COLUMNS = ["date", "description", "debit", "credit", "balance"]

# Supported date styles: 2024-01-31, 31/01/2024, 31-01-24, 31 Jan 2024, 31-Jan-24, 31 Jan, 2024
_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[ ,\-]{1,2}[A-Za-z]{3,9}[ ,\-]{1,2}\d{2,4})"
_LINE_START = re.compile(rf"^\s*({_DATE})(?!\d)\s*(.*)$")
_LEADING_DATE = re.compile(rf"^\s*{_DATE}(?!\d)\s*")
_LEADING_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*")
_NUM = re.compile(
    r"(?<![\w/.-])"
    r"(?:\(\s*-?\s*)?"
    r"-?\d[\d,]*(?:\.\d{1,2})?"
    r"\s*\)?"
    r"(?![\w/.-])"
)

_NOISE = re.compile(
    r"(page\s+\d+|opening balance|closing balance|total (debit|credit)|"
    r"brought forward|carried forward|statement of account|generated on)",
    re.I,
)
_CREDIT_WORDS = re.compile(r"\b(CR|CREDIT|DEPOSIT|INWARD|SALARY|REVERSAL|REFUND)\b", re.I)

# Header keywords (order matters: first match wins for each cell)
HEADER_KEYS = {
    "date": ["date"],
    "balance": ["balance"],
    "debit": ["debit", "withdrawal", "money out", "paid out", "dr"],
    "credit": ["credit", "deposit", "money in", "paid in", "cr"],
    "description": ["description", "narration", "details", "particulars", "remarks"],
    "amount": ["amount"],
}


# ---------- helpers ----------
def to_number(text):
    """'1,234.56' -> 1234.56, '(500.00)' or '-500.00' -> -500.0, '' -> None."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    negative = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = re.sub(r"[^\d.]", "", s)
    if not s or s == ".":
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _clean(x):
    return " ".join(str(x or "").split())


def _parse_date(s):
    if not s or not re.match(rf"^\s*{_DATE}", str(s)):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(str(s).strip(), dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d


def _direction(amount, balance, prev_balance, description):
    """Decide (debit, credit) for a single amount using balance movement, then keywords."""
    if balance is not None and prev_balance is not None:
        if abs(prev_balance - amount - balance) < 0.01:
            return amount, 0.0
        if abs(prev_balance + amount - balance) < 0.01:
            return 0.0, amount
    if _CREDIT_WORDS.search(description or ""):
        return 0.0, amount
    return amount, 0.0


# ---------- strategy 1: tables ----------
def _map_header(row):
    cells = [_clean(c).lower() for c in row]
    mapping = {}
    for i, cell in enumerate(cells):
        if not cell:
            continue
        for key, words in HEADER_KEYS.items():
            if key in mapping:
                continue
            if any((cell == w) if len(w) <= 2 else (w in cell) for w in words):
                mapping[key] = i
                break
    return mapping if "date" in mapping and len(mapping) >= 3 else None


def _from_tables(pdf):
    rows, mapping, prev_bal = [], None, None
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                header = _map_header(row)
                if header:
                    mapping = header
                    continue
                if not mapping:
                    continue

                def get(key):
                    i = mapping.get(key)
                    return row[i] if i is not None and i < len(row) else None

                date_str = _clean(get("date"))
                desc = _clean(get("description"))
                if _parse_date(date_str) is None:
                    # wrapped description line belonging to the previous transaction
                    if rows and not date_str and desc:
                        rows[-1]["description"] += " " + desc
                    continue

                bal = to_number(get("balance"))
                debit, credit = to_number(get("debit")), to_number(get("credit"))
                if debit is None and credit is None and "amount" in mapping:
                    amt = to_number(get("amount"))
                    if amt is None:
                        continue
                    if amt < 0:
                        debit, credit = abs(amt), 0.0
                    else:
                        debit, credit = _direction(amt, bal, prev_bal, desc)
                rows.append({
                    "date": date_str,
                    "description": desc,
                    "debit": abs(debit or 0.0),
                    "credit": abs(credit or 0.0),
                    "balance": bal,
                })
                if bal is not None:
                    prev_bal = bal
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------- strategy 2: text lines ----------
def _from_text(pdf):
    raw, cur = [], None
    for page in pdf.pages:
        for line in (page.extract_text() or "").splitlines():
            m = _LINE_START.match(line)
            if m:
                if cur:
                    raw.append(cur)
                date_str, rest = m.groups()
                rest = _LEADING_DATE.sub("", rest, count=1)  # drop value-date if present
                rest = _LEADING_TIME.sub("", rest, count=1)
                nums = list(_NUM.finditer(rest))
                desc = rest[: nums[0].start()] if nums else rest
                cur = {
                    "date": date_str,
                    "description": desc.strip(),
                    "nums": [to_number(n.group()) for n in nums],
                }
            elif cur and line.strip() and not _NOISE.search(line):
                nums = list(_NUM.finditer(line))
                if nums and not cur["nums"]:
                    # amounts were pushed onto the wrapped line
                    cur["nums"] = [to_number(n.group()) for n in nums]
                    cur["description"] += " " + line[: nums[0].start()].strip()
                else:
                    cur["description"] += " " + line.strip()
    if cur:
        raw.append(cur)

    rows, prev_bal = [], None
    for r in raw:
        nums = [n for n in r["nums"] if n is not None]
        bal = None
        if len(nums) >= 3:  # debit, credit, balance
            debit, credit, bal = abs(nums[-3]), abs(nums[-2]), nums[-1]
        elif len(nums) == 2:  # amount, balance
            bal = nums[1]
            debit, credit = _direction(abs(nums[0]), bal, prev_bal, r["description"])
        elif len(nums) == 1:  # amount only
            debit, credit = _direction(abs(nums[0]), None, prev_bal, r["description"])
        else:
            continue
        if bal is not None:
            prev_bal = bal
        rows.append({
            "date": r["date"],
            "description": r["description"],
            "debit": debit,
            "credit": credit,
            "balance": bal,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------- public API ----------

def _finalize(df):
    if df.empty:
        return df

    df = df.copy()

    # Parse dates without deleting rows that fail validation.
    df["date"] = pd.to_datetime(
        df["date"].map(_parse_date),
        errors="coerce",
    )

    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve missing values rather than automatically
    # converting unrecognized monetary figures to zero.
    df["description"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df.reset_index(drop=True)



def parse_pdf(file, password=None):
    """Parse a PDF and retain the extraction result with
    the greatest number of transaction rows."""

    import pdfplumber

    with pdfplumber.open(file, password=password) as pdf:
        table_df = _finalize(_from_tables(pdf))
        text_df = _finalize(_from_text(pdf))

    # Prefer the result containing more rows.
    # This is a basic safeguard, not proof of completeness.
    if len(text_df) > len(table_df):
        return text_df

    return table_df


def preview_text(file, password=None, pages=1):
    """Raw text of the first page(s), useful for debugging a new bank layout."""
    import pdfplumber

    with pdfplumber.open(file, password=password) as pdf:
        return "\n\n".join((p.extract_text() or "") for p in pdf.pages[:pages])
