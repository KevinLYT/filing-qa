"""Recognize numeric literals as printed in financial tables."""
import re
from decimal import Decimal, InvalidOperation

# Match an entire accounting number so (1,577) cannot support a positive claim.
DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
SIGNED = rf"(?:[-−]\s*)?(?:\$\s*)?{DIGITS}"
NUMBER = re.compile(rf"(?<![\w.,])(?:\(\s*{SIGNED}%?\s*\)|{SIGNED}%?)(?!\w|[.,]\d)")


def raw_key(text):
    """Return the signed decimal value and whether it had a percent sign."""
    if not isinstance(text, str) or NUMBER.fullmatch(text.strip()) is None:
        return None
    number = re.sub(r"[\s,$]", "", text.strip()).replace("−", "-")
    if number.startswith("("):
        number = "-" + number[1:-1]
    try:
        return Decimal(number.rstrip("%")), number.endswith("%")
    except InvalidOperation:
        return None
