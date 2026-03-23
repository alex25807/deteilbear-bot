import os
import csv
import json
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openpyxl.drawing.image import Image as XlImage

LOGS_DIR = "logs"
REPORTS_DIR = "reports"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# gpt-4o-mini pricing (per 1 token)
PRICE_INPUT = 0.15 / 1_000_000
PRICE_OUTPUT = 0.60 / 1_000_000

USD_RUB_RATE = float(os.getenv("USD_RUB_RATE", "90"))

# Прогнозные диапазоны цен (мин=S-класс, средн=M-класс, макс=L-класс / максимальный пакет)
# Источник: прайс-лист студии + типичные коэффициенты S/M/L
SERVICE_PRICE_RANGES: list[tuple[tuple[str, ...], int, int, int]] = [
    (("быстр", "двухфаз"),               3_000,   4_200,   5_500),
    (("комплексн", "мойк"),              8_000,  11_500,  15_000),
    (("деконтам",),                      6_000,   8_500,  12_000),
    (("подкапот",),                      9_000,  12_000,  16_000),
    (("полиров",),                      25_000,  36_000,  55_000),
    (("керамич",),                       5_000,  14_000,  28_000),
    (("интерьер", "салон", "химчист"),  28_000,  40_000,  58_000),
    (("пленк", "ppf", "брон"),          30_000,  58_000,  90_000),
    (("скол", "трещин"),                 3_500,   5_500,   8_000),
    (("самообслуж",),                      700,   1_600,   2_800),
]
_DEFAULT_PRICE_RANGE = (3_000, 7_000, 15_000)


def estimate_price_range(service_topic: str) -> tuple[int, int, int]:
    """Возвращает (мин, средний, макс) прогноз цены по теме услуги."""
    topic = (service_topic or "").lower()
    for keywords, p_min, p_avg, p_max in SERVICE_PRICE_RANGES:
        if any(kw in topic for kw in keywords):
            return p_min, p_avg, p_max
    return _DEFAULT_PRICE_RANGE


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ================ ЛОГИРОВАНИЕ СОБЫТИЙ ================

def _append_csv(filename: str, header: list[str], row: list):
    path = os.path.join(LOGS_DIR, filename)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def log_new_client(user_id: int, source: str, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv("clients.csv", ["user_id", "source", "timestamp"], [user_id, source, ts])


def log_question(user_id: int, question: str, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv("questions.csv", ["user_id", "question", "timestamp"], [user_id, question, ts])


def log_rating(user_id: int, rating: int, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv("ratings.csv", ["user_id", "rating", "timestamp"], [user_id, rating, ts])


def log_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv(
        "tokens.csv",
        ["timestamp", "prompt_tokens", "completion_tokens", "total_tokens"],
        [ts, prompt_tokens, completion_tokens, total_tokens],
    )


def log_button_click(user_id: int, button_data: str, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv("buttons.csv", ["user_id", "button", "timestamp"], [user_id, button_data, ts])


def log_booking(
    user_id: int,
    service: str,
    amount_from: int,
    amount_avg: int = 0,
    amount_max: int = 0,
    timestamp: str = None,
):
    ts = timestamp or datetime.now().isoformat()
    _append_csv(
        "bookings.csv",
        ["user_id", "service", "amount_from", "amount_avg", "amount_max", "timestamp"],
        [user_id, service, amount_from, amount_avg, amount_max, ts],
    )


# ================ ЗАГРУЗКА ДАННЫХ ================

def _load_csv(filename: str, date_col: str = "timestamp") -> pd.DataFrame:
    path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[date_col])
    return df


def _filter_period(df: pd.DataFrame, start: datetime, end: datetime, col: str = "timestamp") -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df[col] >= start) & (df[col] <= end)]


# ================ РАСЧЁТ СТОИМОСТИ ================

def calc_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * PRICE_INPUT + completion_tokens * PRICE_OUTPUT


def calc_cost_rub(prompt_tokens: int, completion_tokens: int) -> float:
    return calc_cost_usd(prompt_tokens, completion_tokens) * USD_RUB_RATE


# ================ ГРАФИКИ ================

def _fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _chart_daily_users(questions_df: pd.DataFrame, clients_df: pd.DataFrame, start: datetime, end: datetime) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))

    if not questions_df.empty:
        daily_active = questions_df.groupby(questions_df["timestamp"].dt.date)["user_id"].nunique()
        ax.plot(daily_active.index, daily_active.values, marker="o", linewidth=2, label="Активные пользователи", color="#2196F3")

    if not clients_df.empty:
        daily_new = clients_df.groupby(clients_df["timestamp"].dt.date)["user_id"].nunique()
        ax.bar(daily_new.index, daily_new.values, alpha=0.4, label="Новые пользователи", color="#4CAF50")

    ax.set_title("Пользователи по дням")
    ax.set_ylabel("Количество")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.autofmt_xdate()
    return _fig_to_bytes(fig)


def _chart_daily_cost(tokens_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))

    if not tokens_df.empty:
        tokens_df = tokens_df.copy()
        tokens_df["cost_rub"] = tokens_df.apply(
            lambda r: calc_cost_rub(r["prompt_tokens"], r["completion_tokens"]), axis=1
        )
        daily_cost = tokens_df.groupby(tokens_df["timestamp"].dt.date)["cost_rub"].sum()
        ax.bar(daily_cost.index, daily_cost.values, color="#FF9800", alpha=0.8)
        for i, (d, v) in enumerate(zip(daily_cost.index, daily_cost.values)):
            if v > 0:
                ax.text(d, v + 0.01, f"{v:.2f}₽", ha="center", fontsize=9)

    ax.set_title("Расходы на OpenAI по дням (₽)")
    ax.set_ylabel("Рубли")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.autofmt_xdate()
    return _fig_to_bytes(fig)


def _chart_daily_tokens(tokens_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))

    if not tokens_df.empty:
        daily = tokens_df.groupby(tokens_df["timestamp"].dt.date).agg(
            prompt=("prompt_tokens", "sum"),
            completion=("completion_tokens", "sum"),
        )
        ax.bar(daily.index, daily["prompt"], label="Prompt", color="#2196F3", alpha=0.7)
        ax.bar(daily.index, daily["completion"], bottom=daily["prompt"], label="Completion", color="#E91E63", alpha=0.7)

    ax.set_title("Токены по дням")
    ax.set_ylabel("Токены")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.autofmt_xdate()
    return _fig_to_bytes(fig)


def _chart_top_buttons(buttons_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))

    if not buttons_df.empty:
        top = buttons_df["button"].value_counts().head(10)
        colors = plt.cm.Set3(range(len(top)))
        ax.barh(top.index[::-1], top.values[::-1], color=colors)
        for i, v in enumerate(top.values[::-1]):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=10)

    ax.set_title("Топ-10 популярных кнопок")
    ax.set_xlabel("Нажатий")
    return _fig_to_bytes(fig)


def _chart_bookings_by_service(bookings_df: pd.DataFrame) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not bookings_df.empty:
        svc = (
            bookings_df.groupby("service")
            .agg(count=("service", "count"), avg=("amount_from", "mean"))
            .sort_values("count", ascending=False)
            .head(8)
        )
        labels = [s[:30] + "…" if len(s) > 30 else s for s in svc.index]

        # Левый: количество записей
        colors_l = plt.cm.Set2(range(len(svc)))
        axes[0].barh(labels[::-1], svc["count"].values[::-1], color=colors_l)
        for i, v in enumerate(svc["count"].values[::-1]):
            axes[0].text(v + 0.1, i, str(v), va="center", fontsize=10)
        axes[0].set_title("Записей по услугам")
        axes[0].set_xlabel("Количество")

        # Правый: средний чек
        colors_r = plt.cm.Set3(range(len(svc)))
        axes[1].barh(labels[::-1], svc["avg"].values[::-1].astype(int), color=colors_r)
        for i, v in enumerate(svc["avg"].values[::-1].astype(int)):
            axes[1].text(v + 100, i, f"{v:,} ₽", va="center", fontsize=10)
        axes[1].set_title("Средний чек по услугам (₽)")
        axes[1].set_xlabel("Рублей")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    return _fig_to_bytes(fig)


def _chart_top_questions(questions_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5))

    if not questions_df.empty:
        top = questions_df["question"].value_counts().head(10)
        labels = [q[:50] + "..." if len(q) > 50 else q for q in top.index]
        colors = plt.cm.Pastel1(range(len(top)))
        ax.barh(labels[::-1], top.values[::-1], color=colors)
        for i, v in enumerate(top.values[::-1]):
            ax.text(v + 0.2, i, str(v), va="center", fontsize=10)

    ax.set_title("Топ-10 вопросов клиентов")
    ax.set_xlabel("Количество")
    return _fig_to_bytes(fig)


# ================ ГЕНЕРАЦИЯ МЕСЯЧНОГО ОТЧЁТА ================

def generate_monthly_report(year: int = None, month: int = None) -> str:
    """Генерирует детальный отчёт за месяц. Возвращает путь к Excel-файлу."""
    now = datetime.now()
    if year is None or month is None:
        if now.day < 5:
            ref = now.replace(day=1) - timedelta(days=1)
        else:
            ref = now
        year, month = ref.year, ref.month

    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    period_str = start.strftime("%Y-%m")

    clients_df = _filter_period(_load_csv("clients.csv"), start, end)
    questions_df = _filter_period(_load_csv("questions.csv"), start, end)
    tokens_df = _filter_period(_load_csv("tokens.csv"), start, end)
    buttons_df = _filter_period(_load_csv("buttons.csv"), start, end)
    ratings_df = _filter_period(_load_csv("ratings.csv"), start, end)
    bookings_df = _filter_period(_load_csv("bookings.csv"), start, end)

    total_unique_users = questions_df["user_id"].nunique() if not questions_df.empty else 0
    total_new_clients = clients_df["user_id"].nunique() if not clients_df.empty else 0
    total_messages = len(questions_df)
    total_button_clicks = len(buttons_df)
    total_bookings = len(bookings_df)
    booked_users = bookings_df["user_id"].nunique() if not bookings_df.empty else 0
    conversion_to_booking = round((booked_users / max(total_unique_users, 1)) * 100, 1)
    revenue_from_bookings = int(bookings_df["amount_from"].sum()) if not bookings_df.empty else 0
    avg_msgs_per_user = round(total_messages / max(total_unique_users, 1), 1)

    total_prompt = int(tokens_df["prompt_tokens"].sum()) if not tokens_df.empty else 0
    total_completion = int(tokens_df["completion_tokens"].sum()) if not tokens_df.empty else 0
    total_tokens = total_prompt + total_completion
    cost_usd = round(calc_cost_usd(total_prompt, total_completion), 4)
    cost_rub = round(calc_cost_rub(total_prompt, total_completion), 2)
    romi_like = round(revenue_from_bookings / cost_rub, 1) if cost_rub > 0 else None
    avg_rating = round(ratings_df["rating"].mean(), 1) if not ratings_df.empty else "—"

    days_in_period = (end - start).days + 1
    active_days = questions_df["timestamp"].dt.date.nunique() if not questions_df.empty else 0
    avg_msgs_per_day = round(total_messages / max(active_days, 1), 1)

    # Предварительный прогноз выручки по диапазонам цен (до сборки сводки)
    # Каждая запись → (мин, средн, макс) из таблицы, суммируем построчно
    if not bookings_df.empty:
        _ranges = bookings_df["service"].apply(estimate_price_range)
        revenue_avg_forecast = int(_ranges.apply(lambda t: t[1]).sum())
        revenue_max_forecast = int(_ranges.apply(lambda t: t[2]).sum())
    else:
        revenue_avg_forecast = 0
        revenue_max_forecast = 0

    romi_avg = round(revenue_avg_forecast / cost_rub, 1) if cost_rub > 0 else None

    # --- Сводка ---
    summary = pd.DataFrame([{
        "Период": f"{start.strftime('%d.%m.%Y')} — {end.strftime('%d.%m.%Y')}",
        "Дней в периоде": days_in_period,
        "Дней с активностью": active_days,
        "Уникальных пользователей": total_unique_users,
        "Новых клиентов (/start)": total_new_clients,
        "Всего сообщений": total_messages,
        "Нажатий на кнопки": total_button_clicks,
        "Подтвержденных записей через бота": total_bookings,
        "Клиентов с записью": booked_users,
        "Конверсия в запись": f"{conversion_to_booking}%",
        "Выручка: факт min (₽)": f"{revenue_from_bookings:,} ₽",
        "Выручка: прогноз средн (₽)": f"{revenue_avg_forecast:,} ₽",
        "Выручка: прогноз макс (₽)": f"{revenue_max_forecast:,} ₽",
        "ROMI-like (факт min / OpenAI)": f"x{romi_like}" if romi_like is not None else "—",
        "ROMI-like (прогноз средн / OpenAI)": f"x{romi_avg}" if romi_avg is not None else "—",
        "Сообщений / пользователь": avg_msgs_per_user,
        "Сообщений / день": avg_msgs_per_day,
        "Средняя оценка": avg_rating,
        "Prompt-токенов": total_prompt,
        "Completion-токенов": total_completion,
        "Всего токенов": total_tokens,
        "Стоимость (USD)": f"${cost_usd}",
        "Курс USD/RUB": USD_RUB_RATE,
        "Стоимость (RUB)": f"{cost_rub} ₽",
    }])

    # --- Активность по дням ---
    daily_rows = []
    if not questions_df.empty:
        dates = pd.date_range(start, end, freq="D")
        for d in dates:
            day = d.date()
            day_q = questions_df[questions_df["timestamp"].dt.date == day]
            day_c = clients_df[clients_df["timestamp"].dt.date == day] if not clients_df.empty else pd.DataFrame()
            day_t = tokens_df[tokens_df["timestamp"].dt.date == day] if not tokens_df.empty else pd.DataFrame()
            day_b = buttons_df[buttons_df["timestamp"].dt.date == day] if not buttons_df.empty else pd.DataFrame()
            day_bookings = bookings_df[bookings_df["timestamp"].dt.date == day] if not bookings_df.empty else pd.DataFrame()

            p_tok = int(day_t["prompt_tokens"].sum()) if not day_t.empty else 0
            c_tok = int(day_t["completion_tokens"].sum()) if not day_t.empty else 0

            daily_rows.append({
                "Дата": day.strftime("%d.%m.%Y"),
                "Новые клиенты": day_c["user_id"].nunique() if not day_c.empty else 0,
                "Активные пользователи": day_q["user_id"].nunique() if not day_q.empty else 0,
                "Сообщений": len(day_q),
                "Кнопок нажато": len(day_b),
                "Записей через бота": len(day_bookings),
                "Выручка min (₽)": int(day_bookings["amount_from"].sum()) if not day_bookings.empty else 0,
                "Токенов": p_tok + c_tok,
                "Стоимость (₽)": round(calc_cost_rub(p_tok, c_tok), 2),
            })
    daily_df = pd.DataFrame(daily_rows)

    # --- Топ вопросов ---
    if not questions_df.empty:
        top_q = questions_df["question"].value_counts().head(20).reset_index()
        top_q.columns = ["Вопрос", "Количество"]
    else:
        top_q = pd.DataFrame(columns=["Вопрос", "Количество"])

    # --- Топ кнопок ---
    if not buttons_df.empty:
        top_b = buttons_df["button"].value_counts().reset_index()
        top_b.columns = ["Кнопка", "Нажатий"]
    else:
        top_b = pd.DataFrame(columns=["Кнопка", "Нажатий"])

    # --- Топ услуг по бронированиям с прогнозом чека ---
    if not bookings_df.empty:
        svc_grp = (
            bookings_df.groupby("service")
            .agg(cnt=("service", "count"), revenue_min=("amount_from", "sum"))
            .reset_index()
            .sort_values("cnt", ascending=False)
        )
        # Прогноз диапазона чеков из таблицы SERVICE_PRICE_RANGES
        prognosis = svc_grp["service"].apply(estimate_price_range)
        svc_grp["Прогноз: мин (₽)"]    = prognosis.apply(lambda t: t[0])
        svc_grp["Прогноз: средн (₽)"]  = prognosis.apply(lambda t: t[1])
        svc_grp["Прогноз: макс (₽)"]   = prognosis.apply(lambda t: t[2])
        svc_grp["Прогноз: выручка средн (₽)"] = svc_grp["cnt"] * svc_grp["Прогноз: средн (₽)"]
        svc_grp["Прогноз: выручка макс (₽)"]  = svc_grp["cnt"] * svc_grp["Прогноз: макс (₽)"]
        top_services = svc_grp.rename(columns={
            "service": "Услуга (тема)",
            "cnt": "Записей",
            "revenue_min": "Факт. выручка min (₽)",
        })
    else:
        top_services = pd.DataFrame(columns=[
            "Услуга (тема)", "Записей", "Факт. выручка min (₽)",
            "Прогноз: мин (₽)", "Прогноз: средн (₽)", "Прогноз: макс (₽)",
            "Прогноз: выручка средн (₽)", "Прогноз: выручка макс (₽)",
        ])

    # Прогноз совокупной выручки за период
    revenue_avg_forecast = int(top_services["Прогноз: выручка средн (₽)"].sum()) if not top_services.empty else 0
    revenue_max_forecast = int(top_services["Прогноз: выручка макс (₽)"].sum()) if not top_services.empty else 0

    # --- Пользователи ---
    user_rows = []
    if not questions_df.empty:
        for uid in questions_df["user_id"].unique():
            uq = questions_df[questions_df["user_id"] == uid]
            ub = buttons_df[buttons_df["user_id"] == uid] if not buttons_df.empty else pd.DataFrame()
            uc = clients_df[clients_df["user_id"] == uid] if not clients_df.empty else pd.DataFrame()
            first_seen = uc["timestamp"].min().strftime("%d.%m.%Y %H:%M") if not uc.empty else "—"
            user_rows.append({
                "User ID": uid,
                "Первый визит": first_seen,
                "Сообщений": len(uq),
                "Кнопок нажато": len(ub),
                "Последнее сообщение": uq["timestamp"].max().strftime("%d.%m.%Y %H:%M"),
            })
    users_df = pd.DataFrame(user_rows)
    if not users_df.empty:
        users_df = users_df.sort_values("Сообщений", ascending=False)

    # --- Графики ---
    charts = {}
    charts["users"] = _chart_daily_users(questions_df, clients_df, start, end)
    charts["bookings"] = _chart_bookings_by_service(bookings_df)
    charts["cost"] = _chart_daily_cost(tokens_df)
    charts["tokens"] = _chart_daily_tokens(tokens_df)
    charts["buttons"] = _chart_top_buttons(buttons_df)
    charts["questions"] = _chart_top_questions(questions_df)

    # --- Сохраняем Excel ---
    report_path = os.path.join(REPORTS_DIR, f"report_{period_str}.xlsx")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary.T.to_excel(writer, sheet_name="Сводка", header=False)
        if not daily_df.empty:
            daily_df.to_excel(writer, sheet_name="По дням", index=False)
        top_q.to_excel(writer, sheet_name="Топ вопросов", index=False)
        top_b.to_excel(writer, sheet_name="Популярные кнопки", index=False)
        top_services.to_excel(writer, sheet_name="Записи по услугам", index=False)
        if not bookings_df.empty:
            bookings_df.to_excel(writer, sheet_name="Бронирования", index=False)
        if not users_df.empty:
            users_df.to_excel(writer, sheet_name="Пользователи", index=False)

        ws = writer.book.create_sheet("Графики")
        row = 1
        for name, img_bytes in charts.items():
            img_path = os.path.join(REPORTS_DIR, f"_tmp_{name}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            img = XlImage(img_path)
            ws.add_image(img, f"A{row}")
            row += 30

    for name in charts:
        tmp = os.path.join(REPORTS_DIR, f"_tmp_{name}.png")
        if os.path.exists(tmp):
            os.remove(tmp)

    return report_path


# ================ ТЕКСТОВАЯ СВОДКА ================

def generate_text_summary(year: int = None, month: int = None) -> str:
    """Краткая текстовая сводка для отправки в Telegram."""
    now = datetime.now()
    if year is None or month is None:
        if now.day < 5:
            ref = now.replace(day=1) - timedelta(days=1)
        else:
            ref = now
        year, month = ref.year, ref.month

    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    clients_df = _filter_period(_load_csv("clients.csv"), start, end)
    questions_df = _filter_period(_load_csv("questions.csv"), start, end)
    tokens_df = _filter_period(_load_csv("tokens.csv"), start, end)
    buttons_df = _filter_period(_load_csv("buttons.csv"), start, end)
    bookings_df = _filter_period(_load_csv("bookings.csv"), start, end)

    total_users = questions_df["user_id"].nunique() if not questions_df.empty else 0
    total_new = clients_df["user_id"].nunique() if not clients_df.empty else 0
    total_msgs = len(questions_df)
    total_clicks = len(buttons_df)
    total_bookings = len(bookings_df)
    booked_users = bookings_df["user_id"].nunique() if not bookings_df.empty else 0
    conversion_to_booking = round((booked_users / max(total_users, 1)) * 100, 1)
    revenue_from_bookings = int(bookings_df["amount_from"].sum()) if not bookings_df.empty else 0

    p_tok = int(tokens_df["prompt_tokens"].sum()) if not tokens_df.empty else 0
    c_tok = int(tokens_df["completion_tokens"].sum()) if not tokens_df.empty else 0
    cost_rub = round(calc_cost_rub(p_tok, c_tok), 2)
    romi_like = round(revenue_from_bookings / cost_rub, 1) if cost_rub > 0 else None

    # Топ услуг по записям
    services_block = ""
    if not bookings_df.empty:
        svc_stats = (
            bookings_df.groupby("service")
            .agg(cnt=("service", "count"), avg=("amount_from", "mean"))
            .sort_values("cnt", ascending=False)
            .head(5)
        )
        # Прогноз выручки по диапазонам
        _txt_ranges = bookings_df["service"].apply(estimate_price_range)
        _txt_avg_fc = int(_txt_ranges.apply(lambda t: t[1]).sum())
        _txt_max_fc = int(_txt_ranges.apply(lambda t: t[2]).sum())

        lines = []
        for svc, row in svc_stats.iterrows():
            label = (svc[:26] + "…") if len(svc) > 26 else svc
            p_min, p_avg, p_max = estimate_price_range(svc)
            lines.append(
                f"   • {label}: {int(row['cnt'])} зап.\n"
                f"     чек {p_min:,}–{p_avg:,}–{p_max:,} ₽ (мин/средн/макс)"
            )
        services_block = (
            "🏷 Топ услуг по записям:\n" + "\n".join(lines) + "\n"
            f"💹 Прогноз выручки: {_txt_avg_fc:,}–{_txt_max_fc:,} ₽\n"
        )
    else:
        _txt_avg_fc = 0
        _txt_max_fc = 0

    romi_avg = round(_txt_avg_fc / cost_rub, 1) if cost_rub > 0 else None

    return (
        f"📊 Отчёт за {start.strftime('%B %Y')}\n"
        f"{'─' * 28}\n"
        f"👥 Уникальных пользователей: {total_users}\n"
        f"🆕 Новых клиентов: {total_new}\n"
        f"💬 Сообщений: {total_msgs}\n"
        f"🔘 Нажатий на кнопки: {total_clicks}\n"
        f"{'─' * 28}\n"
        f"📅 Записей через бота: {total_bookings}\n"
        f"🙋 Клиентов с записью: {booked_users}\n"
        f"📈 Конверсия в запись: {conversion_to_booking}%\n"
        f"💵 Выручка min: {revenue_from_bookings:,} ₽  "
        f"| прогноз: {_txt_avg_fc:,}–{_txt_max_fc:,} ₽\n"
        f"📊 ROMI-like: факт {'x' + str(romi_like) if romi_like is not None else '—'}"
        f"  | прогноз {'x' + str(romi_avg) if romi_avg is not None else '—'}\n"
        f"{services_block}"
        f"{'─' * 28}\n"
        f"🔤 Токенов: {p_tok + c_tok:,}\n"
        f"   ├ prompt: {p_tok:,}\n"
        f"   └ completion: {c_tok:,}\n"
        f"💰 Расходы OpenAI: {cost_rub} ₽\n"
        f"   (курс: {USD_RUB_RATE} ₽/$)\n"
    )


# ================ СОГЛАСИЕ НА ОБРАБОТКУ ПД ================

def log_consent(user_id: int, timestamp: str = None):
    ts = timestamp or datetime.now().isoformat()
    _append_csv("consents.csv", ["user_id", "timestamp"], [user_id, ts])


def has_consent(user_id: int) -> bool:
    path = os.path.join(LOGS_DIR, "consents.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        df = pd.read_csv(path)
        return int(user_id) in df["user_id"].values
    except Exception:
        return False


def delete_user_data(user_id: int) -> dict:
    """Удаляет все данные пользователя из CSV. Возвращает {файл: удалённых строк}."""
    result = {}
    for filename in ("clients.csv", "questions.csv", "buttons.csv", "ratings.csv", "consents.csv", "bookings.csv"):
        path = os.path.join(LOGS_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
            before = len(df)
            df = df[df["user_id"].astype(str) != str(user_id)]
            removed = before - len(df)
            if removed > 0:
                df.to_csv(path, index=False)
                result[filename] = removed
        except Exception:
            pass
    return result


# ================ LEGACY: НЕДЕЛЬНЫЙ ОТЧЁТ ================

def generate_weekly_report() -> str:
    now = datetime.now()
    start = now - timedelta(days=7)
    return generate_monthly_report(now.year, now.month)
