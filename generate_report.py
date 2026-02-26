"""
Генерация месячного отчёта.
Запуск: python generate_report.py [YYYY-MM]
Если месяц не указан — генерирует за текущий (или предыдущий, если < 5-го числа).
"""
import sys
from analytics import generate_monthly_report, generate_text_summary

if __name__ == "__main__":
    year, month = None, None
    if len(sys.argv) > 1:
        parts = sys.argv[1].split("-")
        year, month = int(parts[0]), int(parts[1])

    print(generate_text_summary(year, month))
    path = generate_monthly_report(year, month)
    print(f"\nОтчёт сохранён: {path}")
