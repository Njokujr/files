"""Parse PDF bank statements into one normalised DataFrame.

Output columns: date, description, debit, credit, balance, check

Design rules (these exist because the old parser inflated / invented figures):
  1. A number is only treated as an amount if it looks like money (1,234.56).
     Reference numbers, account numbers and session IDs are never amounts.
  2. Nothing is invented. A value that cannot be read cleanly stays NaN.
  3. Every row is verified against the running balance:
         previous balance + credit - debit == balance
  4. The statement's own opening / closing balance and totals (when printed)
     are compared with what was parsed.
  5. Of the two strategies (tables / text), the one that reconciles best wins,
     not the one that happens to produce more rows.

The `check` column per row:
  ok          amount agrees with the balance movement
  corrected   extracted debit/credit disagreed, fixed using the balance movement
  mismatch    could not be reconciled -> review this row against the PDF
  unreadable  a figure on the row could not be read cleanly (left as NaN)
  unchecked   nothing to check against (first row with no opening balance)
  no_balance  the row has no balance figure
A report dict is attached as `df.attrs["report"]`.
"""
import re
import warnings

import pandas as pd

COLUMNS = ["date", "description", "debit", "credit", "balance", "check"]
TOL = 0.011  # money tolerance (1 kobo / cent)

# Date styles: 2024-01-31, 31/01/2024, 31-01-24, 31 Jan 2024, 31-Jan-24, 31 Jan, 2024
_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[ ,\-]{1,2}[A-Za-z]{3,9}[ ,\-]{1,2}\d{2,4})"
_LINE_START = re.compile(rf"^\s*({_DATE})(?!\d)\s*(.*)$")
_LEADING_DATE = re.compile(rf"^\s*{_DATE}(?!\d)\s*")
_LEADING_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*")

# STRICT money token for free text: must have exactly 2 decimals, so long reference
# numbers (0000132401010912345678) and account numbers can never match.
_AMT = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
_MONEY = re.compile(rf"(?<![\w.,/-])(\(?-?{_AMT}\)?)(?:\s?(CR|DR)\b)?(?![\w.,/-])", re.I)

# Whole-cell number for table cells (the cell must contain ONE number and nothing else).
_CELL = re.compile(
    r"\(?\s*(-)?\s*(?:NGN|₦)?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*\)?\s*(CR|DR)?",
    re.I,
)
_BLANK_CELLS = {"", "-", "--", "—", "n/a", "nil"}

# Lines that are never transactions (page furniture, summary lines, column headers).
_SKIP = re.compile(
    r"(page\s+\d+|opening\s+balance|closing\s+balance|total\s+(debit|credit|withdrawal|deposit|money)|"
    r"brought\s+forward|carried\s+forward|balance\s+b/?f|balance\s+c/?f|statement\s+of\s+account|"
    r"generated\s+on|^\s*date\b.*\b(description|narration|details|particulars)\b.*\bbalance\b)",
    re.I,
)
_CREDIT_WORDS = re.compile(r"\b(CR|CREDIT|DEPOSIT|INWARD|SALARY|REVERSAL|REFUND)\b", re.I)

HEADER_KEYS = {
    "date": ["date"],
    "balance": ["balance"],
    "debit": ["debit", "withdrawal", "money out", "paid out", "dr"],
    "credit": ["credit", "deposit", "money in", "paid in", "cr"],
    "description": ["description", "narration", "details", "particulars", "remarks"],
    "amount": ["amount"],
}


# ---------- helpers ----------
def _clean(x):
    return " ".join(str(x or "").split())


def to_number(text, balance=False):
    """Strict single-number parser for a table cell.

    '1,234.56' -> 1234.56 | '(500.00)' / '-500.00' -> -500.0 | '' -> None.
    Anything that is not exactly one number (e.g. '1,000\\n500', '5,000.0O') -> None.
    The old version stripped every non-digit and glued numbers together
    ('1,000\\n500' became 1000500), which is how figures got inflated.
    """
    if text is None:
        return None
    s = _clean(text)
    m = _CELL.fullmatch(s) if s else None
    if not m:
        return None
    negative = s.startswith("(") or bool(m.group(1))
    if balance and (m.group(3) or "").upper() == "DR":
        negative = True  # "5,000.00 DR" balance = overdrawn
    value = float(m.group(2).replace(",", ""))
    return -value if negative else value


def _money_value(m, balance=False):
    """Value of a regex match from _MONEY (already guaranteed well-formed)."""
    s = m.group(1)
    value = float(re.sub(r"[^\d.]", "", s))
    negative = s.startswith("(") or s.startswith("-") or (balance and (m.group(2) or "").upper() == "DR")
    return -value if negative else value


def _parse_date(s):
    s = str(s or "").strip()
    m = re.match(rf"^{_DATE}", s)
    if not m:
        return None
    token = m.group(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            d = pd.to_datetime(token, format="%Y-%m-%d", errors="coerce")
        else:
            d = pd.to_datetime(token, dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d


def _keyword_direction(amount, description):
    """Last-resort direction guess when the balance cannot confirm it."""
    if _CREDIT_WORDS.search(description or ""):
        return 0.0, amount
    return amount, 0.0


def _fallback_direction(r):
    """Fill debit/credit for a row the balance could not settle, without inventing anything."""
    if r["debit"] is not None or r["credit"] is not None:
        return
    cands = [x for x in r.get("cands") or [] if x is not None]
    if len(cands) == 1:
        r["debit"], r["credit"] = _keyword_direction(cands[0], r["description"])
    # several candidates and no way to choose -> leave NaN


# ---------- statement-level facts (opening / closing / totals) ----------
def _scan_summary(pdf):
    info = {"opening": None, "closing": None, "total_debit": None, "total_credit": None}
    for page in pdf.pages:
        for line in (page.extract_text() or "").splitlines():
            ms = list(_MONEY.finditer(line))
            if not ms:
                continue
            low, val = line.lower(), _money_value(ms[-1], balance=True)
            if info["opening"] is None and re.search(r"opening\s+balance|balance\s+b/?f|brought\s+forward", low):
                info["opening"] = val
            elif re.search(r"closing\s+balance", low):
                info["closing"] = val
            elif re.search(r"total\s+(debit|withdrawal|money out)", low):
                info["total_debit"] = abs(val)
            elif re.search(r"total\s+(credit|deposit|money in)", low):
                info["total_credit"] = abs(val)
    return info


# ---------- strategy 1: tables ----------
def _map_header(row):
    cells = [_clean(c).lower() for c in row]
    mapping, value_date = {}, None
    for i, cell in enumerate(cells):
        if not cell:
            continue
        if "date" in cell and "value" in cell:  # prefer the transaction date over the value date
            value_date = i if value_date is None else value_date
            continue
        for key, words in HEADER_KEYS.items():
            if key in mapping:
                continue
            if any((cell == w) if len(w) <= 2 else (w in cell) for w in words):
                mapping[key] = i
                break
    if "date" not in mapping and value_date is not None:
        mapping["date"] = value_date
    return mapping if "date" in mapping and len(mapping) >= 3 else None


def _table_rows(pdf):
    rows, mapping = [], None
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

                date_cell = str(get("date") or "")
                date_str, desc = _clean(date_cell), _clean(get("description"))

                if _parse_date(date_str) is None:
                    # wrapped description line belonging to the previous transaction
                    if rows and not date_str and desc:
                        rows[-1]["description"] += " " + desc
                    continue

                new = {"date": date_str, "description": desc, "debit": None, "credit": None,
                       "balance": None, "cands": [], "check": None}

                # Several transactions merged into one cell -> cannot be split reliably. Do not guess.
                if len(re.findall(rf"(?<![\w/-]){_DATE}(?!\d)", date_cell)) > 1:
                    new["description"] = "[several rows merged in PDF table - unreadable] " + desc
                    new["check"] = "unreadable"
                    rows.append(new)
                    continue

                bad = False

                def read(key, balance=False):
                    nonlocal bad
                    raw = _clean(get(key))
                    if raw.lower() in _BLANK_CELLS:
                        return None
                    v = to_number(raw, balance=balance)
                    if v is None:
                        bad = True
                    return v

                new["balance"] = read("balance", balance=True)
                if "debit" in mapping or "credit" in mapping:
                    new["debit"], new["credit"] = read("debit"), read("credit")
                    new["debit"] = abs(new["debit"]) if new["debit"] is not None else None
                    new["credit"] = abs(new["credit"]) if new["credit"] is not None else None
                    new["cands"] = [x for x in (new["debit"], new["credit"]) if x]
                elif "amount" in mapping:
                    amt = read("amount")
                    if amt is not None:
                        new["cands"] = [abs(amt)]
                        if amt < 0:
                            new["debit"], new["credit"] = abs(amt), 0.0

                if bad or (new["debit"] is None and new["credit"] is None and not new["cands"]):
                    new["check"] = "unreadable"
                rows.append(new)
    return rows, 0


# ---------- strategy 2: text lines ----------
def _split_amounts(text):
    """Split a line into (description, trailing run of money tokens).

    Only the contiguous run of amounts at the END of the line counts as amounts;
    any figure inside the narration stays part of the description.
    """
    ms = list(_MONEY.finditer(text))
    if not ms:
        return _clean(text), []
    run = [ms[-1]]
    for m in reversed(ms[:-1]):
        if text[m.end():run[0].start()].strip() == "":
            run.insert(0, m)
        else:
            break
    return _clean(text[:run[0].start()] + " " + text[run[-1].end():]), run


def _text_rows(pdf):
    raw, cur = [], None
    for page in pdf.pages:
        for line in (page.extract_text() or "").splitlines():
            if _SKIP.search(line):
                if cur:
                    cur["closed"] = True  # page furniture: stop gluing text onto the last row
                continue
            m = _LINE_START.match(line)
            if m:
                if cur:
                    raw.append(cur)
                date_str, rest = m.groups()
                rest = _LEADING_TIME.sub("", _LEADING_DATE.sub("", rest, count=1), count=1)
                desc, run = _split_amounts(rest)
                cur = {"date": date_str, "desc": desc, "run": run, "closed": False}
            elif cur and not cur["closed"] and line.strip():
                desc, run = _split_amounts(line)
                if run and not cur["run"] and not line[run[-1].end():].strip():
                    cur["run"] = run  # amounts were pushed onto the wrapped line
                    cur["desc"] = _clean(cur["desc"] + " " + desc)
                else:
                    cur["desc"] = _clean(cur["desc"] + " " + line)
    if cur:
        raw.append(cur)

    rows, dropped = [], 0
    for r in raw:
        run = r["run"]
        if not run:
            dropped += 1  # a dated line with no readable amount: reported, never guessed
            continue
        row = {"date": r["date"], "description": r["desc"], "debit": None, "credit": None,
               "balance": None, "cands": [], "check": None}
        if len(run) == 1:
            row["cands"] = [abs(_money_value(run[0]))]
        else:
            row["balance"] = _money_value(run[-1], balance=True)
            vals = [abs(_money_value(m)) for m in run[:-1]]
            row["cands"] = vals
            if len(vals) >= 2:
                d, c = vals[-2], vals[-1]  # debit and credit columns both printed (one is 0.00)
                if d < TOL or c < TOL:
                    row["debit"], row["credit"] = d, c
        rows.append(row)
    return rows, dropped


# ---------- verification against the running balance ----------
def _reconcile(rows, opening):
    prev = opening
    for r in rows:
        bal = r["balance"]
        if r["check"] == "unreadable":
            if bal is not None:
                prev = bal
            continue

        cands = [x for x in r.get("cands") or [] if x is not None]
        d, c = r["debit"], r["credit"]
        have_dc = d is not None or c is not None

        if bal is not None and prev is not None:
            delta = round(bal - prev, 2)
            if have_dc and abs(delta - ((c or 0.0) - (d or 0.0))) < TOL:
                r["check"] = "ok"
            else:
                amt = abs(delta)
                fits = (amt < TOL and not any(x >= TOL for x in cands)) or any(abs(amt - x) < TOL for x in cands)
                if fits:
                    r["debit"], r["credit"] = (amt, 0.0) if delta < 0 else (0.0, amt)
                    r["check"] = "corrected" if have_dc else "ok"
                else:
                    r["check"] = "mismatch"
                    _fallback_direction(r)
        elif bal is None:
            r["check"] = "no_balance"
            _fallback_direction(r)
        else:
            r["check"] = "unchecked"
            _fallback_direction(r)

        if bal is not None:
            prev = bal


def _score(candidate):
    name, rows, _ = candidate
    verified = sum(r["check"] in ("ok", "corrected") for r in rows)
    bad = sum(r["check"] in ("mismatch", "unreadable") for r in rows)
    return (verified, -bad, name == "tables", len(rows))


# ---------- output ----------
def _to_df(rows):
    out = []
    for r in rows:
        d, c = r["debit"], r["credit"]
        if d is None and c is None:
            d = c = float("nan")
        else:
            d, c = abs(d or 0.0), abs(c or 0.0)
        out.append({
            "date": _parse_date(r["date"]),
            "description": _clean(r["description"]),
            "debit": d,
            "credit": c,
            "balance": r["balance"],
            "check": r["check"],
        })
    df = pd.DataFrame(out, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _report(df, info, strategy, dropped, errors):
    counts = df["check"].value_counts().to_dict() if not df.empty else {}
    warn = list(errors)
    n = len(df)
    bad = counts.get("mismatch", 0) + counts.get("unreadable", 0)

    if counts.get("mismatch"):
        warn.append(f"{counts['mismatch']} row(s) do not agree with the running balance. Check them against the PDF.")
    if counts.get("unreadable"):
        warn.append(f"{counts['unreadable']} row(s) contain figures that could not be read cleanly (left blank).")
    if dropped:
        warn.append(f"{dropped} dated line(s) had no readable amount and were skipped.")
    if df["date"].isna().any():
        warn.append(f"{int(df['date'].isna().sum())} row(s) have an unreadable date.")
    if counts.get("unchecked", 0) + counts.get("no_balance", 0) and not (counts.get("ok", 0) or counts.get("corrected", 0)):
        warn.append("No balance column / opening balance found, so figures could not be verified.")

    rep = {
        "strategy": strategy, "rows": n, "counts": counts, "dropped": dropped,
        "opening": info["opening"], "closing_stated": info["closing"],
        "last_balance": None if df.empty or df["balance"].dropna().empty else float(df["balance"].dropna().iloc[-1]),
        "totals": {},
    }

    stated_ok = []
    if info["closing"] is not None and rep["last_balance"] is not None:
        ok = abs(info["closing"] - rep["last_balance"]) < 0.02
        stated_ok.append(ok)
        if not ok:
            warn.append(f"Last parsed balance {rep['last_balance']:,.2f} differs from the statement's closing balance "
                        f"{info['closing']:,.2f}: rows may be missing.")
    for key, col in (("total_debit", "debit"), ("total_credit", "credit")):
        if info[key] is not None and n:
            parsed = float(df[col].sum())
            ok = abs(parsed - info[key]) < 0.02
            rep["totals"][key] = {"stated": info[key], "parsed": parsed, "ok": ok}
            stated_ok.append(ok)
            if not ok:
                warn.append(f"Parsed {col} total {parsed:,.2f} differs from the statement's {info[key]:,.2f}.")

    verified = counts.get("ok", 0) + counts.get("corrected", 0)
    rep["verified"] = verified
    rep["warnings"] = warn
    rep["trusted"] = bool(n and bad == 0 and verified > 0 and all(stated_ok) and not dropped)
    return rep


# ---------- public API ----------
def parse_pdf(file, password=None):
    """Parse a PDF. Returns a DataFrame; verification details are in df.attrs['report']."""
    import pdfplumber

    errors, candidates = [], []
    with pdfplumber.open(file, password=password) as pdf:
        info = _scan_summary(pdf)
        for name, fn in (("tables", _table_rows), ("text", _text_rows)):
            try:
                rows, dropped = fn(pdf)
            except Exception as e:  # one strategy failing must not kill the other
                errors.append(f"The {name} strategy failed ({type(e).__name__}: {e}).")
                rows, dropped = [], 0
            _reconcile(rows, info["opening"])
            candidates.append((name, rows, dropped))

    name, rows, dropped = max(candidates, key=_score)
    df = _to_df(rows)
    df.attrs["report"] = _report(df, info, name, dropped, errors)
    return df


def preview_text(file, password=None, pages=1):
    """Raw text of the first page(s), useful for debugging a new bank layout."""
    import pdfplumber

    with pdfplumber.open(file, password=password) as pdf:
        return "\n\n".join((p.extract_text() or "") for p in pdf.pages[:pages])