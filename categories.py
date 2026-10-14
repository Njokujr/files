"""Keyword rules for categorising transactions. Edit freely."""
import re

# Order matters: first matching category wins (fees before transfers, etc.)
DEFAULT_RULES = {
    "Bank Charges": ["CHARGE", "FEE", "COMMISSION", "VAT", "STAMP DUTY", "SMS ALERT", "MAINTENANCE", "LEVY"],
    "Cash Withdrawal": ["ATM", "WDL", "CASH WITHDRAWAL"],
    "Groceries": ["SHOPRITE", "SPAR", "JUSTRITE", "PRINCE EBEANO", "SUPERMARKET", "MARKET"],
    "Transport": ["UBER", "BOLT", "INDRIVE", "FUEL", "OANDO", "MOBIL", "TOTAL", "LCC", "TOLL"],
    "Food & Dining": ["KFC", "DOMINO", "CHICKEN REPUBLIC", "MR BIGGS", "SWEET SENSATION",
                      "RESTAURANT", "CAFE", "EATERY", "KILIMANJARO"],
    "Airtime & Data": ["AIRTIME", "MTN", "GLO", "AIRTEL", "9MOBILE", "DATA"],
    "Bills & Utilities": ["DSTV", "GOTV", "STARTIMES", "IKEDC", "EKEDC", "PHCN", "ELECTRICITY",
                          "NEPA", "PREPAID", "SPECTRANET", "WATER"],
    "Subscriptions": ["NETFLIX", "SPOTIFY", "APPLE", "GOOGLE", "YOUTUBE", "SHOWMAX", "PRIME VIDEO"],
    "Shopping": ["JUMIA", "KONGA", "AMAZON", "ALIEXPRESS", "POS"],
    "Health": ["PHARMACY", "HOSPITAL", "CLINIC", "MEDPLUS", "HEALTH"],
    "Betting": ["SPORTYBET", "BET9JA", "NAIRABET", "1XBET", "BETKING"],
    "Savings & Investments": ["PIGGYVEST", "COWRYWISE", "RISE", "INVEST", "SAVINGS"],
    "Transfers": ["TRANSFER", "TRF", "NIP"],
}

INCOME_RULES = {
    "Salary": ["SALARY", "PAYROLL"],
    "Refund / Reversal": ["REFUND", "REVERSAL"],
    "Interest": ["INTEREST"],
    "Incoming Transfer": ["TRANSFER", "TRF", "NIP"],
}

OTHER = "Other"
OTHER_INCOME = "Other Income"


def _matches(keyword, text):
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def parse_custom_rules(text):
    """'NETFLIX=Subscriptions' per line -> {'NETFLIX': 'Subscriptions'}"""
    rules = {}
    for line in (text or "").splitlines():
        if "=" in line:
            key, cat = line.split("=", 1)
            if key.strip() and cat.strip():
                rules[key.strip().upper()] = cat.strip()
    return rules


def categorize(description, credit, debit, custom=None):
    text = (description or "").upper()

    for keyword, cat in (custom or {}).items():  # your own rules win
        if _matches(keyword, text):
            return cat

    if credit and credit > 0 and not (debit and debit > 0):
        for cat, keywords in INCOME_RULES.items():
            if any(_matches(k, text) for k in keywords):
                return cat
        return OTHER_INCOME

    for cat, keywords in DEFAULT_RULES.items():
        if any(_matches(k, text) for k in keywords):
            return cat
    return OTHER


def all_categories(custom=None):
    cats = list(DEFAULT_RULES) + list(INCOME_RULES) + [OTHER, OTHER_INCOME]
    cats += [c for c in (custom or {}).values() if c not in cats]
    return cats
