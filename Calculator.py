
"""
Calculator module for AI Document Assistant.

All actual mathematical operations
are performed using Python.
"""


def calculate(operation, values, percentage=None):

    numbers = []

    for value in values:

        if isinstance(value, bool):
            continue

        try:
            number = float(
                str(value).replace(",", "").strip()
            )

            numbers.append(number)

        except (TypeError, ValueError):
            continue

    if operation == "count":
        return len(numbers)

    if not numbers:
        return None

    if operation == "sum":
        return sum(numbers)

    if operation == "average":
        return sum(numbers) / len(numbers)

    if operation == "maximum":
        return max(numbers)

    if operation == "minimum":
        return min(numbers)

    if operation == "percentage":

        if percentage is None:
            return None

        return sum(numbers) * float(percentage) / 100

    if operation == "add_percentage":

        if percentage is None:
            return None

        base = sum(numbers)

        extra = base * float(percentage) / 100

        return {
            "base": base,
            "percentage_amount": extra,
            "total": base + extra
        }

    return None


def format_number(number):

    if isinstance(number, float) and number.is_integer():
        return f"{int(number):,}"

    return f"{number:,.2f}"
