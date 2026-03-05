import logging
import os
import asyncio
import time
import datetime as dt
import re
import difflib
import csv
import unicodedata
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import pathlib

load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.error import TimedOut
from telegram.request import HTTPXRequest
import openai

from analytics import (
    log_new_client, log_question, log_token_usage, log_button_click,
    log_rating, generate_monthly_report, generate_text_summary,
    log_consent, has_consent, delete_user_data, log_booking,
)

# ================== НАСТРОЙКИ ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_FILE_PATH = os.getenv("KNOWLEDGE_FILE_PATH", "knowledges.txt")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))
BOOKING_MARKER = "[ЗАПИСЬ]"
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))
RATING_DELAY = int(os.getenv("RATING_DELAY", "300"))
FOLLOWUP_DELAY = int(os.getenv("FOLLOWUP_DELAY", "86400"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "21600"))

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не установлен в .env")

# ================== ЛОГИРОВАНИЕ ==================
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

file_handler = RotatingFileHandler(
    "logs/bot.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.propagate = False

# ================== КЛАВИАТУРЫ ==================
BOOKING_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/"))],
    [InlineKeyboardButton("✅ Я записался", callback_data="booking_done")],
])

PHONE_LINE = "\n\n📞 Или позвоните администратору: +7 (926) 021-60-00"

GREETING_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🔧 Услуги мастеров", callback_data="menu_services"),
        InlineKeyboardButton("🚿 Самообслуживание", callback_data="menu_self"),
    ],
    [
        InlineKeyboardButton("🧴 Магазин автохимии", callback_data="menu_shop"),
        InlineKeyboardButton("☕ Комфорт в студии", callback_data="menu_comfort"),
    ],
    [
        InlineKeyboardButton("💰 Цены", callback_data="menu_prices"),
        InlineKeyboardButton("📍 Как нас найти", callback_data="menu_address"),
    ],
    [
        InlineKeyboardButton("🏆 Наши преимущества", callback_data="menu_advantages"),
        InlineKeyboardButton("📸 Примеры работ", callback_data="menu_portfolio"),
    ],
    [
        InlineKeyboardButton("🔥 Новинки и акции", url="https://t.me/bearlake_detailing"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

WASH_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🚿 Быстрая мойка (от 3 000 ₽)", callback_data="sub_wash_twophase"),
    ],
    [
        InlineKeyboardButton("🧽 Комплексная мойка (от 8 000 ₽)", callback_data="sub_wash_complex"),
    ],
    [
        InlineKeyboardButton("🧪 Деконтаминация (от 6 000 ₽)", callback_data="sub_decon"),
    ],
    [
        InlineKeyboardButton("🛠 Подкапотное пространство (от 9 000 ₽)", callback_data="sub_engine_wash"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

SERVICES_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🚿 Мойка", callback_data="sub_wash"),
        InlineKeyboardButton("✨ Полировка", callback_data="sub_polish"),
    ],
    [
        InlineKeyboardButton("🛡 Защита ЛКП", callback_data="sub_protection"),
        InlineKeyboardButton("🧹 Детейлинг салона", callback_data="sub_interior"),
    ],
    [
        InlineKeyboardButton("🪟 Бронирование стекла", callback_data="sub_glass"),
        InlineKeyboardButton("🔧 Ремонт сколов", callback_data="sub_chips"),
    ],
    [
        InlineKeyboardButton("🧪 Деконтаминация", callback_data="sub_decon"),
    ],
    [
        InlineKeyboardButton("💰 Все цены", callback_data="sub_all_prices"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

SELF_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🔧 Оборудование", callback_data="sub_self_equip"),
        InlineKeyboardButton("💰 Стоимость", callback_data="sub_self_price"),
    ],
    [
        InlineKeyboardButton("📋 Что включено", callback_data="sub_self_included"),
        InlineKeyboardButton("⚠️ Правила", callback_data="sub_self_rules"),
    ],
    [
        InlineKeyboardButton("🧴 Купить автохимию", callback_data="menu_shop"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

SHOP_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🧴 Что рекомендуете", callback_data="sub_shop_recommend"),
        InlineKeyboardButton("🏷 Скидки и промокоды", callback_data="sub_shop_discounts"),
    ],
    [
        InlineKeyboardButton("🛒 Магазин на Ozon", callback_data="sub_shop_ozon"),
        InlineKeyboardButton("🏪 Сайт студии", url="https://bearlake.clients.site/"),
    ],
    [
        InlineKeyboardButton("🛒 Открыть Ozon", url="https://ozon.ru/t/bal1Akq"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

COMFORT_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("☕ Кофе и перекусы", callback_data="sub_comfort_food"),
        InlineKeyboardButton("📶 Wi-Fi и зона отдыха", callback_data="sub_comfort_wifi"),
    ],
    [
        InlineKeyboardButton("❄️ Климат в помещении", callback_data="sub_comfort_climate"),
        InlineKeyboardButton("♿ Доступная среда", callback_data="sub_comfort_access"),
    ],
    [
        InlineKeyboardButton("📅 Записаться онлайн", web_app=WebAppInfo(url="https://n1024167.yclients.com/")),
    ],
])

SUB_MENUS = {
    "menu_services": SERVICES_KEYBOARD,
    "menu_self": SELF_KEYBOARD,
    "menu_shop": SHOP_KEYBOARD,
    "menu_comfort": COMFORT_KEYBOARD,
    "sub_wash": WASH_KEYBOARD,
}

CONSENT_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("✅ Принимаю", callback_data="consent_accept")],
])

RATING_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("1 ⭐", callback_data="rate_1"),
        InlineKeyboardButton("2 ⭐", callback_data="rate_2"),
        InlineKeyboardButton("3 ⭐", callback_data="rate_3"),
        InlineKeyboardButton("4 ⭐", callback_data="rate_4"),
        InlineKeyboardButton("5 ⭐", callback_data="rate_5"),
    ],
])

NO_CONSENT_TEXT = (
    "Для использования бота необходимо дать согласие "
    "на обработку данных.\nНажмите /start, чтобы начать."
)

PRIVACY_TEXT = (
    "🔒 Политика конфиденциальности\n\n"
    "Оператор персональных данных:\n"
    "Студия BEARLAKE DETAILING & SHOP\n\n"
    "📋 Обрабатываемые данные:\n"
    "• Telegram ID (технический идентификатор)\n"
    "• Текст сообщений в чате с ботом\n"
    "• Дата и время обращений\n"
    "• Нажатия на кнопки меню\n\n"
    "🎯 Цели обработки:\n"
    "• Консультирование по услугам студии\n"
    "• Улучшение качества обслуживания\n"
    "• Формирование обезличенной статистики\n\n"
    "🛡 Защита данных:\n"
    "• Данные хранятся на защищённом сервере\n"
    "• Доступ имеют только уполномоченные сотрудники\n"
    "• Данные не передаются третьим лицам\n\n"
    "⏱ Срок хранения:\n"
    "До момента удаления вами через /deletedata\n\n"
    "Ваши права (ст. 14–17 ФЗ-152):\n"
    "• Получить информацию об обработке данных\n"
    "• Потребовать удаления — /deletedata\n"
    "• Отозвать согласие — /deletedata\n\n"
    "📩 Контакт по вопросам ПД:\n"
    "+7 (926) 021-60-00"
)

TOPIC_LABELS = {
    "menu_services": "услугами мастеров",
    "menu_self": "постом самообслуживания",
    "menu_prices": "ценами на услуги",
    "menu_address": "адресом студии",
    "menu_shop": "магазином автохимии",
    "menu_comfort": "условиями в студии",
    "sub_wash": "мойкой",
    "sub_wash_twophase": "быстрой детейлинг-мойкой",
    "sub_wash_complex": "комплексной мойкой Bearlake",
    "sub_polish": "полировкой",
    "sub_protection": "защитой ЛКП",
    "sub_interior": "детейлингом салона",
    "sub_glass": "бронированием стекла",
    "sub_chips": "ремонтом сколов",
    "sub_all_prices": "ценами на услуги",
    "sub_self_equip": "оборудованием для самообслуживания",
    "sub_self_price": "стоимостью самообслуживания",
    "sub_self_included": "условиями аренды поста",
    "sub_self_rules": "правилами самообслуживания",
    "sub_shop_recommend": "автохимией",
    "sub_shop_ozon": "магазином на Ozon",
    "sub_shop_discounts": "скидками в магазине",
    "sub_comfort_food": "перекусами в студии",
    "sub_comfort_wifi": "Wi-Fi и зоной отдыха",
    "sub_comfort_climate": "климатом в студии",
    "sub_comfort_access": "доступностью студии",
    "menu_advantages": "нашими преимуществами",
    "menu_portfolio": "примерами работ",
}

MENU_PROMPTS = {
    # Главное меню
    "menu_services": "Какие услуги вы предоставляете?",
    "menu_self": "Расскажите про пост самообслуживания — условия, оборудование, цены",
    "menu_prices": "Какие у вас цены на услуги?",
    "menu_address": "Какой у вас адрес и график работы?",
    "menu_shop": "Расскажите про ваш магазин автохимии и автокосметики. Где можно купить? Какие бренды?",
    "menu_comfort": "Какие у вас условия в студии? Расскажите про комфорт для клиентов",
    "menu_portfolio": "Где посмотреть примеры ваших работ? Покажите примеры",
    # Подменю: услуги мастеров
    "sub_wash": "Какие форматы мойки и очистки вы предлагаете? Расскажите про быструю, комплексную мойку, деконтаминацию и мойку подкапотного пространства.",
    "sub_wash_twophase": "Расскажите подробно про быструю детейлинг-мойку — что входит, сколько стоит, сколько по времени?",
    "sub_wash_complex": "Расскажите подробно про комплексную Детейлинг мойку Bearlake — что входит, сколько стоит, сколько по времени?",
    "sub_polish": "Сколько стоит полировка и что включено?",
    "sub_protection": "Какие защитные покрытия вы предлагаете? Сколько стоит защита ЛКП?",
    "sub_interior": "Расскажите про детейлинг салона",
    "sub_glass": "Сколько стоит бронирование лобового стекла?",
    "sub_chips": "Сколько стоит ремонт сколов и трещин на стекле?",
    "sub_all_prices": "Расскажите все цены на услуги студии",
    # Подменю: самообслуживание
    "sub_self_equip": "Какое оборудование есть на посту самообслуживания?",
    "sub_self_price": "Сколько стоит аренда поста самообслуживания? Какие тарифы?",
    "sub_self_included": "Что включено в аренду поста? Нужно ли брать что-то с собой?",
    "sub_self_rules": "Какие есть ограничения и правила на посту? Можно ли прийти без записи? Можно ли работать вдвоём?",
    # Подменю: магазин
    "sub_shop_recommend": "Какую автохимию и автокосметику рекомендуете? Какие бренды используете?",
    "sub_shop_ozon": "Расскажите подробнее про ваш магазин на Ozon и бренды в ассортименте.",
    "sub_shop_discounts": "Какие скидки и промокоды действуют в магазине?",
    # Подменю: комфорт
    "sub_comfort_food": "Есть ли у вас кофе, чай, перекусы? Можно ли поесть в студии?",
    "sub_comfort_wifi": "Есть ли Wi-Fi? Есть ли зона отдыха? Можно ли оставить вещи?",
    "sub_comfort_climate": "Есть ли отопление и кондиционирование? Есть ли туалет?",
    "sub_comfort_access": "Есть ли условия для маломобильных клиентов?",
}

MASTER_SERVICE_PRICE_NOTE = (
    "\n\n*Стоимость услуг мастеров зависит от класса авто (S / M / L) "
    "и уточняется после осмотра.*"
)

STATIC_MENU_ANSWERS = {
    "menu_services": (
        "В студии BEARLAKE доступны все ключевые форматы ухода — от поддерживающей мойки до глубокой защиты кузова и салона.\n\n"
        "Услуги:\n"
        "• Самообслуживание 24/7 (почасовая аренда)\n"
        "• Быстрая детейлинг-мойка — от 3 000 ₽, от 1 часа\n"
        "• Комплексная детейлинг-мойка Bearlake — от 8 000 ₽, от 4 часов\n"
        "• Полировка ЛКП (Light Polish) — от 25 000 ₽, от 12 часов\n"
        "• Нанесение керамических покрытий — от 5 000 ₽, от 4 часов\n"
        "• Детейлинг чистка интерьера — от 28 000 ₽, 1 день\n"
        "• Защитные полиуретановые пленки (зоны риска/кузов) — от 85 000 ₽, от 2 дней\n"
        "• Бронирование лобового стекла — от 30 000 ₽, от 1 дня\n"
        "• Ремонт трещин и сколов стекла — от 3 500 ₽, от 1 часа\n"
        "• Деконтаминация ЛКП (металлические вкрапления и битум) — от 6 000 ₽, от 2 часов\n"
        "• Мойка подкапотного пространства — от 9 000 ₽, от 4 часов\n\n"
        "*Стоимость услуг мастеров (кроме самообслуживания) зависит от класса авто (S / M / L) "
        "и уточняется после осмотра.*\n\n"
        "Выберите раздел ниже — расскажу состав услуги, сроки и ориентир по стоимости ⬇️"
    ),
    "menu_self": (
        "Самообслуживание 24/7 в BEARLAKE — это профессиональная студия для самостоятельного ухода за авто в комфортных условиях.\n\n"
        "Что получаете:\n"
        "• онлайн-запись через YClients\n"
        "• климат-контроль и приточно-вытяжную вентиляцию\n"
        "• АВД, горячую воду, мощный пылесос, турбосушку\n"
        "• ведра, пенокомплекты и широкий набор аксессуаров\n"
        "• комнату для хранения вещей и отдыха\n"
        "• возможность работать вдвоем\n\n"
        "Тарифы:\n"
        "• 700 ₽/час — бокс и оборудование\n"
        "• 900 ₽/час — бокс, оборудование + шампунь для 1/2 фазы, губка и микрофибра\n"
        "Минимум — 2 часа. Максимум не ограничен (по согласованию можно на несколько дней).\n\n"
        "При наличии квалификации на посту самообслуживания можно самостоятельно выполнять "
        "те же этапы ухода, что и мастера: мойка, подготовка, полировка, нанесение защитных "
        "составов и детальная уборка салона.\n\n"
        "Выберите пункт ниже — расскажу детали и условия ⬇️"
    ),
    "menu_shop": (
        "В магазине собрали только проверенную автохимию, которой реально работаем в студии.\n"
        "Выберите раздел ниже — подскажу, что лучше под Вашу задачу ⬇️"
    ),
    "menu_comfort": (
        "Пока автомобиль в работе, Вы отдыхаете в комфортной зоне:\n"
        "• кофе/чай и лёгкие перекусы\n"
        "• Wi-Fi и зона отдыха\n"
        "• санузел и место для хранения вещей\n"
        "• отопление и кондиционирование\n"
        "• доступная среда\n\n"
        "Выберите интересующий пункт ниже ⬇️"
    ),
    "menu_address": (
        "Мы находимся по адресу: Московская область, г.о. Пушкинский, пос. Нагорное, 8А.\n"
        "Студия работает ежедневно по предварительной записи.\n"
        "Рядом есть бесплатная парковка.\n\n"
        "Яндекс-карты: https://yandex.ru/maps/org/bearlake/46971604224/?ll=38.023081%2C56.067485&z=17"
    ),
    "menu_portfolio": (
        "Хотите увидеть результат «до/после»? Примеры работ доступны:\n"
        "• на сайте студии\n"
        "• в Telegram-канале"
        "\n\nСайт: https://bearlake.clients.site/\n"
        "Telegram: https://t.me/bearlake_detailing"
    ),
    "menu_discounts": (
        "Актуальные выгоды для клиентов:\n"
        "• В офлайн-магазине — 5% для постоянных клиентов\n"
        "• На Ozon при первом заказе (после записи на услуги) — 5% по промокоду BEARLAKE\n\n"
        "Акции публикуем в Telegram-канале: https://t.me/bearlake_detailing"
    ),
    "sub_wash": (
        "По мойке и очистке есть 4 формата:\n"
        "• Быстрая детейлинг-мойка — поддерживающий уход (от 3 000 ₽, от 1 часа)\n"
        "• Комплексная детейлинг-мойка Bearlake — глубокий уход внутри и снаружи (от 8 000 ₽, от 4 часов)\n"
        "• Деконтаминация ЛКП — удаление битума и металлических вкраплений (от 6 000 ₽, от 2 часов)\n\n"
        "• Мойка подкапотного пространства — деликатная очистка с изоляцией чувствительных компонентов (от 9 000 ₽, от 4 часов)\n\n"
        "Деконтаминация — это этап глубокой химической очистки ЛКП от стойких загрязнений,\n"
        "которые обычная мойка не убирает и которые со временем могут вредить покрытию.\n\n"
        "Выберите вариант ниже — и сразу покажу, что входит и когда это действительно нужно ⬇️"
    ),
    "sub_wash_twophase": (
        "Быстрая детейлинг-мойка — от 3 000 ₽, от 1 часа.\n"
        "Идеальный формат, когда важно быстро привести автомобиль в порядок без компромиссов по качеству.\n\n"
        "Что входит:\n"
        "• бережная двухфазная мойка кузова\n"
        "• мойка внутренних порогов\n"
        "• мойка/пылесос ковриков\n"
        "• продувка замков\n\n"
        "Результат: чистый и свежий автомобиль в короткий срок и поддержание аккуратного состояния ЛКП."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_wash_complex": (
        "Комплексная детейлинг-мойка Bearlake — от 8 000 ₽, от 4 часов.\n"
        "Включает:\n"
        "• бережную 2-фазную мойку кузова и внутренних порогов\n"
        "• чистку труднодоступных мест кистями\n"
        "• тщательную чистку дисков и шин с пропиткой резины\n"
        "• полную продувку кузова горячим воздухом\n"
        "• очистку стекол снаружи и внутри\n"
        "• уборку салона и багажника пылесосом\n"
        "• глубокую чистку салонных ковров\n"
        "• деликатную очистку интерьера с консервацией\n"
        "• обработку ЛКП и дисков керамическим квик-детейлером\n\n"
        "Это формат «всё и сразу», когда нужен максимальный визуальный и тактильный результат внутри и снаружи.\n"
        "При сложных загрязнениях может потребоваться деконтаминация."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_polish": (
        "Полировка ЛКП (Light Polish) — от 25 000 ₽, от 12 часов.\n"
        "Перед услугой обязательна быстрая или комплексная мойка.\n"
        "Что входит:\n"
        "• подготовительная детейлинг-мойка\n"
        "• бережная коррекция ЛКП\n"
        "• удаление до 50% царапин\n"
        "• восстановление блеска и гладкости\n"
        "• защита ЛКП на 3–4 месяца (финишный силант)\n\n"
        "Работаем по двухступенчатой технологии с химией Optimum Polymer Technologies и NXTZEN, "
        "используем круги Lake Country и оборудование Zentool / LC Power Tools / FLEX.\n"
        "Результат: глубокий зеркальный блеск и заметно более свежий внешний вид автомобиля."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_protection": (
        "По защите ЛКП доступны:\n"
        "• керамические покрытия — от 5 000 ₽, от 4 часов\n"
        "• полиуретановые пленки (зоны риска/кузов) — от 85 000 ₽, от 2 дней\n\n"
        "Керамика подбирается по задачам и сроку службы: легкие составы (3–12 месяцев) "
        "и профессиональные покрытия (2–3 года).\n"
        "Доступны, в том числе, NXTZEN Graphene Serum, NXTZEN Elite GSO2 и Opti-Coat Pro Plus.\n"
        "По пленкам доступны варианты NXTZEN / SKINCARS / Crystal Ultra Gloss — защита от сколов, "
        "царапин, пескоструя и реагентов."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_decon": (
        "Деконтаминация ЛКП — от 6 000 ₽, от 2 часов.\n"
        "Перед услугой обязательна быстрая или комплексная мойка.\n"
        "Удаляем металлические вкрапления, битум, смолы и стойкие органические загрязнения.\n"
        "Работаем премиальной химией Optimum Polymer Technologies, NXTZEN и TAC System.\n"
        "Результат: гладкое и чистое ЛКП, профилактика коррозии и подготовка к следующим этапам ухода."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_engine_wash": (
        "Мойка подкапотного пространства — от 9 000 ₽, от 4 часов.\n"
        "Что делаем: предварительная изоляция чувствительных компонентов, "
        "деликатная мойка, полная продувка и обработка защитным консервантом.\n"
        "Результат: чистое подкапотное пространство, более простая диагностика "
        "и поддержание ресурса узлов."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_interior": (
        "Детейлинг чистка интерьера (химчистка) — от 28 000 ₽, 1 день.\n"
        "Перед химчисткой требуется мойка.\n"
        "Что входит:\n"
        "• удаление загрязнений и пыли со всех поверхностей\n"
        "• глубокая чистка карпета салона и багажника\n"
        "• очистка сидений, дверных карт, панели и дефлекторов\n"
        "• чистка стекол изнутри и проработка труднодоступных мест\n\n"
        "Дополнительно можно нанести защиту интерьера:\n"
        "• NXTZEN L-Coat (1–2 года)\n"
        "• Opti-Guard Leather (1–2 года)\n"
        "• NXTZEN Fiber Coat (6–9 месяцев)\n\n"
        "Результат: свежий и аккуратный салон, комфортный микроклимат и защита материалов от износа."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_glass": (
        "Бронирование лобового стекла — от 30 000 ₽, от 1 дня.\n"
        "Виды пленок: Rayno Crystal Shield, Never Scratch.\n"
        "Что входит: тщательная очистка/обезжиривание и профессиональная установка.\n"
        "Пленка защищает от сколов, трещин, пескоструя и царапин, "
        "остается прозрачной и не искажает обзор."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_chips": (
        "Ремонт трещин и сколов стекла — от 3 500 ₽, от 1 часа.\n"
        "Задача услуги — остановить дальнейшее разрушение стекла и "
        "сделать дефект менее заметным."
        + MASTER_SERVICE_PRICE_NOTE
    ),
    "sub_self_equip": (
        "На посту самообслуживания есть всё для полноценного ухода своими руками:\n"
        "• АВД и шланг для промывки\n"
        "• горячая вода\n"
        "• мощный пылесос\n"
        "• турбосушка\n"
        "• ведра и пенокомплекты\n"
        "• широкий набор аксессуаров (кисти, варежки, инструменты для шин/дисков/арок)\n"
        "• комната для хранения вещей и отдыха."
    ),
    "sub_self_price": (
        "Тарифы:\n"
        "• 700 ₽/час — бокс + оборудование\n"
        "• 900 ₽/час — бокс + оборудование + расходники\n"
        "Минимальное время — 2 часа.\n"
        "Максимальное время не ограничено: можно арендовать на несколько дней по согласованию.\n"
        "Оплата: наличными, картой или онлайн."
    ),
    "sub_self_included": (
        "В тариф 900 ₽/час входят: шампунь для 1 и 2 фазы, губка и микрофибровое полотенце. "
        "В тариф 700 ₽/час — бокс и оборудование.\n"
        "Можно использовать свою химию."
    ),
    "sub_self_rules": (
        "Рекомендуем запись заранее через YClients.\n"
        "Арендовать можно на любое количество часов, в том числе на длительный период (посуточно/на несколько дней) по согласованию.\n"
        "Можно работать вдвоем.\n"
        "При наличии квалификации можно самостоятельно выполнять те же этапы ухода, "
        "что и мастера, соблюдая правила студии и технику безопасности.\n"
        "Первое посещение — по предварительному согласованию с 9:00 до 22:00.\n"
        "Запрещены грязные ремонтные работы и агрессивные жидкости."
    ),
    "sub_comfort_food": "В студии есть кофе, чай, лёгкие перекусы и фрукты для клиентов.",
    "sub_comfort_wifi": "Да, есть бесплатный Wi-Fi и зона отдыха.",
    "sub_comfort_climate": "Да, есть отопление, кондиционирование и санузел.",
    "sub_comfort_access": "Да, предусмотрены условия для маломобильных клиентов.",
    "direct_booking": "Записаться можно по кнопке ниже ⬇️ [ЗАПИСЬ]",
}

TEXT_INTENT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("цены", "прайс", "сколько стоит", "стоимость"), "menu_prices"),
    (("адрес", "как найти", "как добраться", "где вы", "где находитесь"), "menu_address"),
    (("график", "режим работы", "часы работы"), "menu_address"),
    (("самообслуж", "пост", "аренда бокса"), "menu_self"),
    (("оборудован", "авд", "пылесос", "турбосуш"), "sub_self_equip"),
    (("700", "900", "тариф", "расходник", "минимальное время"), "sub_self_price"),
    (("мойка", "двухфаз", "комплексн"), "sub_wash"),
    (("полиров", "light polish"), "sub_polish"),
    (("керамик", "пленк", "ppf", "защит"), "sub_protection"),
    (("деконтам", "битум", "вкраплен"), "sub_decon"),
    (("подкапот", "двигател"), "sub_engine_wash"),
    (("салон", "химчист", "интерьер"), "sub_interior"),
    (("лобов", "стекл", "бронир"), "sub_glass"),
    (("несколько дней", "на день", "на сутки", "длительн", "аренд"), "sub_self_rules"),
    (("скол", "трещин"), "sub_chips"),
    (("магазин", "автохим", "ozon"), "menu_shop"),
    (("скидк", "промокод", "акци"), "sub_shop_discounts"),
    (("кофе", "чай", "wifi", "вайфай", "комфорт", "отдых"), "menu_comfort"),
    (("примеры", "портфолио", "работ"), "menu_portfolio"),
    (("преимущ", "почему вы", "чем лучше"), "menu_advantages"),
]


PRIORITY_TEXT_INTENT_RULES: list[tuple[tuple[str, ...], str]] = [
    # Жесткий приоритет: запись
    (("как записаться", "хочу записаться", "записаться", "запись", "yclients", "юклиентс"), "direct_booking"),
    # Жесткий приоритет: цены
    (("сколько стоит", "какая цена", "цены", "прайс", "стоимость"), "menu_prices"),
    # Жесткий приоритет: адрес/график
    (("где вы находитесь", "как доехать", "как добраться", "адрес", "график", "режим работы"), "menu_address"),
    # Жесткий приоритет: скидки
    (("промокод", "скидка", "скидки", "акция", "акции"), "sub_shop_discounts"),
    # Жесткий приоритет: длительная аренда поста
    (("на несколько дней", "несколько дней", "на сутки", "посуточ", "длительная аренда"), "sub_self_rules"),
]

# Для текстовых сообщений шаблонный ответ оставляем только для
# наиболее критичных бизнес-сценариев.
STRICT_TEXT_INTENTS = {
    "direct_booking",
    "menu_prices",
    "sub_all_prices",
    "menu_address",
    "menu_shop",
    "sub_shop_recommend",
    "sub_shop_ozon",
    "sub_shop_discounts",
    "menu_advantages",
}

# Для карточек услуг по кнопкам приоритет отдаем KB/GPT, а не шаблонам.
AI_FIRST_BUTTON_CALLBACKS = {
    "menu_services",
    "menu_self",
    "sub_wash",
    "sub_wash_twophase",
    "sub_wash_complex",
    "sub_decon",
    "sub_engine_wash",
    "sub_polish",
    "sub_protection",
    "sub_interior",
    "sub_glass",
    "sub_chips",
    "sub_self_equip",
    "sub_self_price",
    "sub_self_included",
    "sub_self_rules",
}

NO_CTA_CALLBACKS = {
    "menu_address",
    "menu_portfolio",
    "menu_comfort",
    "sub_comfort_food",
    "sub_comfort_wifi",
    "sub_comfort_climate",
    "sub_comfort_access",
    "menu_discounts",
}

DETAILABLE_CALLBACKS = {
    "sub_wash",
    "sub_wash_twophase",
    "sub_wash_complex",
    "sub_decon",
    "sub_engine_wash",
    "sub_polish",
    "sub_protection",
    "sub_interior",
    "sub_glass",
    "sub_chips",
    "sub_self_equip",
    "sub_self_price",
    "sub_self_included",
    "sub_self_rules",
}


def _matches_keyword(text_lower: str, keyword: str) -> bool:
    """Проверяет keyword без ложных совпадений внутри других слов."""
    if " " in keyword:
        return keyword in text_lower
    pattern = rf"(?<!\w){re.escape(keyword)}\w*"
    return re.search(pattern, text_lower) is not None


def _detect_priority_intent(text_lower: str) -> str | None:
    """Приоритетные интенты, которые должны перебивать обычный роутинг."""
    # Формулировки вида "на 2 дня/3 суток" должны жестко идти в длительную аренду.
    if re.search(r"(?:на\s*)?\d+\s*(?:дн|дня|дней|сут|сутк)", text_lower):
        return "sub_self_rules"
    for keywords, intent in PRIORITY_TEXT_INTENT_RULES:
        if any(_matches_keyword(text_lower, keyword) for keyword in keywords):
            return intent
    return None


def _is_short_followup_request(user_text: str) -> bool:
    text = user_text.strip().lower()
    if len(text) > 24:
        return False
    triggers = {
        "подскажи", "подскажите", "давай", "ок", "хорошо", "понятно",
        "что лучше", "что посоветуешь", "и что", "и как",
    }
    return text in triggers


def _context_followup_answer(last_topic: str) -> str | None:
    low = last_topic.lower()
    if "самообслуж" in low or "аренд" in low:
        return (
            "Подскажу по самообслуживанию коротко:\n"
            "• 700 ₽/час — бокс и оборудование\n"
            "• 900 ₽/час — бокс, оборудование и расходники\n"
            "• минимум — 2 часа, максимум не ограничен (на несколько дней по согласованию)\n\n"
            "Если планируете аренду на 2+ дня, сразу сориентирую по правилам и оптимальному формату."
        )
    if "мойк" in low:
        return "Подскажу по мойке: для быстрого результата — быстрая детейлинг-мойка, для полного ухода — комплексная. Если хотите, сразу помогу выбрать под вашу задачу."
    if "защит" in low or "керамик" in low or "пленк" in low:
        return "Подскажу по защите: керамика — про блеск и удобный уход, пленка — про максимальную физическую защиту от сколов. Могу подобрать вариант под ваш бюджет и срок."
    return None


INTENT_CLARIFY_LABELS = {
    "menu_prices": "цены",
    "menu_address": "адрес и график",
    "menu_self": "самообслуживание",
    "sub_self_equip": "оборудование поста",
    "sub_self_price": "стоимость самообслуживания",
    "sub_self_included": "что включено в аренду",
    "sub_self_rules": "правила и длительная аренда",
    "sub_wash": "мойка",
    "sub_polish": "полировка",
    "sub_protection": "защита ЛКП",
    "sub_decon": "деконтаминация",
    "sub_engine_wash": "мойка подкапотного пространства",
    "sub_interior": "уход за салоном",
    "sub_glass": "бронирование стекла",
    "sub_chips": "ремонт сколов/трещин",
    "menu_shop": "магазин автохимии",
    "sub_shop_discounts": "скидки и промокоды",
    "menu_comfort": "условия в студии",
    "menu_portfolio": "примеры работ",
    "menu_advantages": "преимущества студии",
    "direct_booking": "запись",
}


def _detect_text_intent(text_lower: str) -> tuple[str | None, list[str]]:
    """Возвращает (intent, candidates). Если intent=None и есть candidates — нужна уточнялка."""
    scores: dict[str, int] = {}
    for keywords, intent in TEXT_INTENT_RULES:
        score = sum(1 for keyword in keywords if _matches_keyword(text_lower, keyword))
        if score > 0:
            scores[intent] = score

    if not scores:
        return None, []

    top_score = max(scores.values())
    top_intents = [intent for intent, score in scores.items() if score == top_score]

    if len(top_intents) == 1:
        return top_intents[0], top_intents

    return None, top_intents

# ================== ЗАГРУЗКА БАЗЫ ЗНАНИЙ ==================
try:
    with open(KNOWLEDGE_FILE_PATH, "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
    logger.info("База знаний загружена (%d символов).", len(KNOWLEDGE_BASE))
except FileNotFoundError:
    KNOWLEDGE_BASE = ""
    logger.error("Файл базы знаний %s не найден.", KNOWLEDGE_FILE_PATH)


def _build_kb_faq(knowledge_text: str) -> list[dict]:
    """Парсит блоки вида '### вопрос' + ответ из knowledges.txt."""
    entries: list[dict] = []
    current_q: str | None = None
    current_answer_lines: list[str] = []

    def flush_entry() -> None:
        nonlocal current_q, current_answer_lines
        if not current_q:
            return
        answer = "\n".join(current_answer_lines).strip()
        if answer:
            variants = [
                v.strip().lower()
                for v in re.split(r"\s*/\s*", current_q)
                if v.strip()
            ]
            entries.append(
                {
                    "question": current_q.strip(),
                    "variants": variants,
                    "answer": answer,
                    # Индекс для поиска: варианты вопроса + начало ответа.
                    "search_tokens": _tokenize_for_match(
                        f"{current_q} {answer[:500]}"
                    ),
                }
            )
        current_q = None
        current_answer_lines = []

    for raw_line in knowledge_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### "):
            flush_entry()
            current_q = line[4:].strip()
            current_answer_lines = []
            continue
        if line.startswith("## "):
            flush_entry()
            continue
        if current_q is not None:
            current_answer_lines.append(line)

    flush_entry()
    return entries


def _tokenize_for_match(text: str) -> set[str]:
    stop_words = {
        "как", "где", "что", "это", "или", "для", "при", "есть", "нет", "можно",
        "нужно", "вас", "вы", "мы", "они", "она", "оно", "его", "ее", "их",
        "the", "and", "for", "with", "from",
    }

    def _normalize_token(token: str) -> str:
        token = token.lower().replace("ё", "е")
        # Легкая нормализация окончаний (без внешних библиотек),
        # чтобы "стоимость/стоимости", "полировка/полировки" матчились стабильнее.
        for suffix in ("иями", "ями", "ами", "ией", "ий", "ой", "ей", "ом", "ам", "ям", "ах", "ях", "ия", "ие", "ые", "ого", "ему", "ыми", "ыми", "ть", "ться", "лся", "лась", "лись", "ов", "ев", "иям", "иях", "ый", "ий", "ая", "ое", "ые", "у", "ю", "а", "я", "ы", "и", "е", "о"):
            if len(token) > 4 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    raw_tokens = re.findall(r"[a-zA-Zа-яА-Я0-9]+", text.lower())
    norm_tokens = {_normalize_token(t) for t in raw_tokens}
    return {t for t in norm_tokens if len(t) >= 3 and t not in stop_words}


def _detect_user_language(text: str) -> str:
    """Определяет язык собеседника (ru/en) по тексту пользователя."""
    cyr = len(re.findall(r"[а-яА-ЯёЁ]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    if cyr >= lat:
        return "ru"
    return "en"


def _contains_cyrillic(text: str) -> bool:
    return re.search(r"[а-яА-ЯёЁ]", text) is not None


def _is_emoji_char(ch: str) -> bool:
    if not ch:
        return False
    category = unicodedata.category(ch)
    if category == "So":
        return True
    code = ord(ch)
    return (
        0x2600 <= code <= 0x27BF
        or 0x1F300 <= code <= 0x1FAFF
    )


def _emoji_count(text: str) -> int:
    return sum(1 for ch in text if _is_emoji_char(ch))


def _trim_to_max_chars(text: str, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip()
    return f"{clipped}…"


def _enforce_emoji_range(text: str, min_count: int = 2, max_count: int = 4) -> str:
    current = _emoji_count(text)
    if current > max_count:
        kept = 0
        out_chars: list[str] = []
        for ch in text:
            if _is_emoji_char(ch):
                if kept >= max_count:
                    continue
                kept += 1
            out_chars.append(ch)
        text = "".join(out_chars).rstrip()
        current = _emoji_count(text)
    if current < min_count:
        need = min_count - current
        pool = ["✅", "🚗", "📌", "🛠️"]
        addon = " ".join(pool[:need])
        text = f"{text.rstrip()} {addon}".strip()
    return text


async def _translate_text_if_needed(text: str, user_lang: str) -> str:
    """Переводит ответ под язык собеседника, если язык не совпадает."""
    if user_lang == "ru":
        return text
    if not _contains_cyrillic(text):
        return text
    try:
        completion = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the text to English. Keep meaning, bullet structure, links "
                        "and marker [ЗАПИСЬ] unchanged if present."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        translated = completion.choices[0].message.content.strip()
        return translated or text
    except Exception as e:
        logger.warning("Не удалось перевести ответ на en: %s", e)
        return text


async def _finalize_response_text(text: str, user_lang: str) -> str:
    """Жестко применяет ограничения: язык, длина, эмодзи."""
    out = await _translate_text_if_needed(text, user_lang)
    out = _trim_to_max_chars(out, 1000)
    out = _enforce_emoji_range(out, min_count=2, max_count=4)
    out = _trim_to_max_chars(out, 1000)
    return out


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if not cleaned:
        return set()
    if len(cleaned) <= n:
        return {cleaned}
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def _find_kb_answer(user_text: str) -> tuple[str | None, float, str | None]:
    """Ищет наиболее релевантный ответ в FAQ-блоках базы знаний."""
    if not KB_FAQ:
        return None, 0.0, None

    query = user_text.strip().lower()
    query_tokens = _tokenize_for_match(query)
    best_answer = None
    best_question = None
    best_score = 0.0
    second_best_score = 0.0
    query_ngrams = _char_ngrams(query)

    for entry in KB_FAQ:
        answer = entry["answer"]
        original_question = entry["question"]
        entry_tokens = entry.get("search_tokens", set())
        for variant in entry["variants"]:
            ratio = difflib.SequenceMatcher(None, query, variant).ratio()
            variant_tokens = _tokenize_for_match(variant)
            overlap_variant = (
                len(query_tokens & variant_tokens) / max(len(variant_tokens), 1)
                if query_tokens and variant_tokens
                else 0.0
            )
            overlap_entry = (
                len(query_tokens & entry_tokens) / max(len(query_tokens), 1)
                if query_tokens and entry_tokens
                else 0.0
            )
            variant_ngrams = _char_ngrams(variant)
            ngram_similarity = (
                len(query_ngrams & variant_ngrams) / max(len(query_ngrams | variant_ngrams), 1)
                if query_ngrams and variant_ngrams
                else 0.0
            )
            contains_bonus = 0.10 if (variant in query or query in variant) else 0.0
            score = (
                0.38 * ratio
                + 0.22 * overlap_variant
                + 0.25 * overlap_entry
                + 0.15 * ngram_similarity
                + contains_bonus
            )
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_answer = answer
                best_question = original_question
            elif score > second_best_score:
                second_best_score = score

    # Если два топ-кандидата почти равны, лучше вернуть "нет матча"
    # и перейти к уточнению, чем дать нерелевантный ответ.
    if best_score >= 0.48 and (best_score - second_best_score) >= 0.05:
        return best_answer, best_score, best_question
    return None, best_score, best_question


def _log_kb_match(user_id: int, user_text: str, matched_question: str | None, score: float, status: str) -> None:
    """Логирует матчи FAQ в CSV для последующего улучшения базы знаний."""
    path = pathlib.Path("logs") / "kb_matches.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "user_id", "user_text", "matched_question", "score", "status"])
        writer.writerow([
            dt.datetime.now().isoformat(),
            user_id,
            user_text[:500],
            (matched_question or "")[:300],
            round(score, 4),
            status,
        ])


KB_FAQ = _build_kb_faq(KNOWLEDGE_BASE)
KB_TOKEN_INDEX: set[str] = set()
for _entry in KB_FAQ:
    KB_TOKEN_INDEX.update(_entry.get("search_tokens", set()))

ON_TOPIC_HINTS = (
    "детейлинг", "мойк", "полиров", "керамик", "ppf", "пленк",
    "стекл", "скол", "трещин", "защит", "салон", "химчист", "деконтам",
    "подкапот", "самообслуж", "пост", "аренда", "цена", "стоим", "прайс",
    "адрес", "график", "запис", "yclients", "ozon", "магазин", "скидк", "акци",
    "чат", "помог", "консультац",
    "bearlake",
)


def _is_on_topic(user_text: str) -> bool:
    text_lower = user_text.lower()
    if any(greet in text_lower for greet in ("привет", "здравст", "добрый", "спасибо", "ок", "понял")):
        return True
    if any(_matches_keyword(text_lower, hint) for hint in ON_TOPIC_HINTS):
        return True
    tokens = _tokenize_for_match(user_text)
    if not tokens:
        return True
    return bool(tokens & KB_TOKEN_INDEX)


COMPLEX_CONSULTATION_HINTS = (
    "что лучше", "чем отличается", "сравни", "сравнение", "подберите", "подбери",
    "какой вариант", "какую услугу", "что выбрать", "под мой бюджет", "бюджет",
    "срок", "по времени", "как часто", "ежедневно", "трасса", "город",
    "хранение", "новая машина", "б/у", "комплексно", "под ключ",
)


def _is_complex_consultation(user_text: str) -> bool:
    text_lower = user_text.lower()
    if any(phrase in text_lower for phrase in COMPLEX_CONSULTATION_HINTS):
        return True
    # Признак комплексного запроса: длинный вопрос с несколькими условиями.
    return len(user_text) >= 120 and ("," in user_text or " и " in text_lower)


def _build_consultation_query(user_text: str) -> str:
    """Формирует уточненный запрос к модели для консультаций."""
    return (
        "Клиент просит помочь с подбором решения. Дай консультацию как опытный детейлер:\n"
        "1) Короткий вывод по ситуации клиента.\n"
        "2) Рекомендованный вариант (базовый/оптимальный/максимальный).\n"
        "3) Почему именно так (2-4 аргумента без воды).\n"
        "4) Что входит в предложенные услуги и ориентир по срокам.\n"
        "5) В конце мягкий CTA к записи через кнопку [ЗАПИСЬ].\n"
        "Обязательно учитывай, что стоимость услуг мастеров зависит от класса авто "
        "(S / M / L) и уточняется после осмотра.\n\n"
        f"Запрос клиента: {user_text}"
    )


def _detect_consultation_priority(user_text: str) -> str | None:
    """Распознает приоритет клиента после уточняющего вопроса."""
    text = user_text.lower()
    if any(k in text for k in ("скорост", "быстр", "срочно", "оператив", "по времени")):
        return "скорость"
    if any(k in text for k in ("результат", "максим", "качество", "идеал", "без компромисс")):
        return "максимальный результат"
    if any(k in text for k in ("защит", "керамик", "пленк", "ppf", "долго", "долговеч")):
        return "долговременная защита"
    if any(k in text for k in ("бюджет", "дешев", "эконом", "стоим", "цена", "недорог")):
        return "бюджет"
    return None


def _build_refined_consultation_query(base_request: str, priority: str, user_reply: str) -> str:
    """Формирует follow-up запрос с учетом приоритета клиента."""
    return (
        "Клиент уточнил приоритет после консультации. Обнови рекомендацию:\n"
        "1) Коротко подтверди, что понял приоритет.\n"
        "2) Дай оптимальный пакет услуг под этот приоритет.\n"
        "3) Укажи ориентир по срокам и стартовой стоимости.\n"
        "4) Добавь 1 альтернативу (короче/дешевле или сильнее/дольше).\n"
        "5) В конце мягкий CTA через [ЗАПИСЬ].\n"
        "Обязательно учти: стоимость услуг мастеров зависит от класса авто "
        "(S / M / L) и уточняется после осмотра.\n\n"
        f"Изначальный запрос клиента: {base_request}\n"
        f"Уточнение клиента: {user_reply}\n"
        f"Приоритет клиента: {priority}"
    )


def _consultation_followup_question(user_text: str, answer_text: str) -> str:
    """Добавляет короткий уточняющий вопрос для дожима к записи."""
    low = f"{user_text} {answer_text}".lower()
    if "под ключ" in low or "комплекс" in low:
        return "Чтобы точнее подобрать вариант, что приоритетнее: скорость, максимальный результат, защита или бюджет?"
    if "бюджет" in low or "цена" in low or "стоим" in low:
        return "Чтобы предложить оптимальный пакет, уточните приоритет одним словом: скорость, результат, защита или бюджет."
    if "защит" in low or "керамик" in low or "пленк" in low:
        return "Уточню для точного подбора: что приоритетнее — скорость, максимальный результат, защита или бюджет?"
    return "Чтобы точнее сориентировать, напишите одним словом приоритет: скорость, результат, защита или бюджет."

# ================== СИСТЕМНЫЙ ПРОМТ ==================
SYSTEM_PROMPT = r"""
#РОЛЬ
Ты — специалист-консультант студии детейлинга BEARLAKE DETAILING & SHOP с опытом более 10 лет.
Выступаешь в онлайн-чате на сайте, помогаешь клиентам выбрать подходящие услуги и формат обслуживания.

#ЦЕЛИ И ЗАДАЧИ
• Помогать клиентам выбирать между услугами мастеров или постом самообслуживания  
• Консультировать по вопросам детейлинга  
• Направлять клиентов на запись через доступные каналы  
• Соблюдать фазы продажи
• Создавать положительное впечатление о салоне  
• Укреплять доверие и лояльность клиентов  
• Обеспечивать индивидуальный подход

# ПРИОРИТЕТЫ

• Если нужно дать ссылку, не вставляй URL в текст. Пиши короткое название ресурса (например: «Яндекс-карты», «Telegram-канал», «Ozon», «сайт студии») — бот добавит активные кнопки автоматически.

#ПРИВЕТСТВИЕ
Само приветственное сообщение клиенту отправляет бот.
Ты в своих ответах НЕ пишешь отдельное приветствие и не начинаешь фразы с
«Здравствуйте», «Добрый день» и т.п., если только клиент сам прямо не просит
сформулировать приветствие.
Считай, что с клиентом уже поздоровались — отвечай сразу по делу.


#ЯЗЫК ОТВЕТОВ
Отвечать на языке собеседника.

#СТИЛЬ ОТВЕТА
• Общаться от мужского лица
• Обращаться к клиентам на «Вы»  
• Формулировать понятные ответы (до 1000 символов)  
• Общаться в дружелюбном, вежливом и профессиональном тоне  
• Демонстрировать экспертность и спокойствие, не перегружая техническими терминами  
• Говорить по существу, тактично и вежливо  
• Не повторять информацию, предоставленную клиентом, другими словами  
• Представлять перечисления в виде списков
• Использовать эмодзи для визуального оформления (2–4 на сообщение, не перебарщивай)

#ЛОГИКА ПОСТРОЕНИЯ ДИАЛОГА

## ФАЗА 1 — ВЫЯВЛЕНИЕ ПОТРЕБНОСТЕЙ:
1. Поздороваться с клиентом
2. Кратко рассказать об услугах студии и наличии поста самообслуживания  
3. Если клиент выбирает услуги студии, задать уточняющие вопросы:  
   • Что требуется привести в порядок (кузов, салон, защита, комплекс)  
   • Как используется авто (ежедневно/по выходным, трасса/город, хранение)  
   • Интересует ли защита кузова, салон или комплекс  
   • Важность сроков и бюджета  
4. Если клиент выбирает самообслуживание, не запрашивать уточняющие данные по использованию авто  
5. Если клиент запрашивает адрес, ответить:
Мы находимся по адресу : Московская обл., г.о. Пушкинский, пос. Нагорное, 8А.
Адрес на яндекс-картах : https://yandex.ru/maps/org/bearlake/46971604224/?ll=38.023081%2C56.067485&z=17  
5. Если клиент запрашивает график работы, ответить:
Студия работает ежедневно с 9.00 до 22.00, без выходных
Автомойка (пост самообслуживания) работает круглосуточно
Адрес на яндекс-картах : https://yandex.ru/maps/org/bearlake/46971604224/?ll=38.023081%2C56.067485&z=17  

## ФАЗА 2 — ПРЕДЛОЖЕНИЕ ЗАПИСИ
• Бот не запрашивает дату, время, марку авто или другие детали записи, так как не имеет доступа к расписанию и не может подтвердить доступность.  
• После выявления потребностей или ответа на информационный вопрос (цены, адрес, условия и т.д.) — сразу предложить записаться по кнопке.
• НЕ вставляй ссылку YClients и номер телефона текстом — поставь маркер [ЗАПИСЬ] в самом конце ответа, кнопки появятся автоматически.
• Пример: «Отлично! Записаться можно по кнопке ниже ⬇️ Администратор подберёт удобное время и уточнит стоимость! [ЗАПИСЬ]»

## ФАЗА 3 — ПОДТВЕРЖДЕНИЕ:
1. После предложения записи — поблагодарить за интерес  
2. Напомнить, что администратор свяжется для подтверждения  

#РАБОТА С ВАРИАНТАМИ ФОРМУЛИРОВОК ВОПРОСОВ
• Если клиент спрашивает «как вас найти», «где вы находитесь», «как до вас доехать» — считайте это одним и тем же вопросом и отвечайте по разделу «Адрес и как добраться».
• Если клиент спрашивает «какие у вас преимущества», «почему выбрать вас», «чем вы лучше других» — отвечайте по разделу «Наши преимущества».  
• Если клиент упоминает «Yclients», «Юклиентс», «yc», «запись через сайт» — считайте это ссылкой на онлайн-запись и отвечайте по разделу «Как записаться?».  
• Если клиент спрашивает «где можно перекусить?», «есть ли еда?», «можно ли поесть?» — отвечайте по разделу «Можно ли перекусить?».

#ВАЖНОЕ УТОЧНЕНИЕ К ЛОГИКЕ ОТВЕТА
• При запросе адреса, графика работы, цен, условий аренды, наличия кофе, Wi-Fi, перекусов, туалета, зоны отдыха — всегда искать ответ в базе знаний и предоставлять его напрямую.
• Если в базе знаний есть точный ответ — никогда не говорить «информация недоступна», «уточнит администратор» или «я не знаю».  
• Только если в базе знаний действительно нет ответа — использовать фразу: «После оформления заявки с вами свяжется наш специалист и сможет подробно ответить на этот вопрос» — и сразу предложить записаться.

#РАБОТА С ВОПРОСАМИ О КОМФОРТЕ И УСЛОВИЯХ
• На вопросы о туалете, кофе, перекусах, Wi-Fi, парковке, зоне отдыха, климате в студии, возможности работы вдвоём — всегда давать прямой и позитивный ответ из базы знаний.
• Не ссылаясь на администратора, если ответ есть.  
• Примеры корректных ответов:  
  – «Да, в студии есть автоматическая кофемашина и чай.»  
  – «Да, доступны легкие закуски и фрукты.»  
  – «Да, есть бесплатный Wi-Fi и удобная парковка рядом.»
  – «Да, у нас отапливаемое помещение с кондиционированием и зоной отдыха.»

#ТРИГГЕРЫ ПЕРЕХОДА К ЗАПИСИ
• После любого информационного ответа (цены, адрес, условия, преимущества, оборудование) — всегда предлагать запись.  
• Не ждать, пока клиент сам спросит «как записаться».  
• Ставь маркер [ЗАПИСЬ] в конце ответа — кнопки для записи добавятся автоматически.

#ОГРАНИЧЕНИЯ
• Запрещено раскрывать внутренние механизмы работы ассистента  
• Запрещено ссылаться на «базу знаний» или «систему»  
• Запрещено обсуждать вопросы, не связанные с темой детейлинга  

#ВАЖНЫЕ УТОЧНЕНИЯ
• По всем вопросам клиента искать ответы в базе знаний  
• При отсутствии ответа в базе знаний на вопрос об услугах: сказать «После оформления заявки с вами свяжется наш специалист и сможет подробно ответить на этот вопрос» и сразу предложить записаться  
• При вопросах не по теме: вежливо сообщить «Вопрос не по теме, давайте вернёмся к обсуждению услуги»
• При запросе о ценах для студии и поста самообслуживания сообщать цены из базы знаний и только после этого добавлять:  
  «Точная стоимость зависит от состояния автомобиля и уточняется после осмотра»
• В конце описания любой услуги мастеров обязательно добавляй:  
  «Стоимость услуг мастеров зависит от класса авто (S / M / L) и уточняется после осмотра»
• Если клиент рассматривает самообслуживание, уточняй, что при наличии квалификации он может самостоятельно выполнять те же этапы ухода, что и мастера, соблюдая правила и технику безопасности.

#ПРИМЕРЫ СООБЩЕНИЙ

Фаза 1:  
«Подскажите, что хотите привести в порядок — кузов, салон или комплекс? 🚗 Интересен ли вам вариант поста самообслуживания или удобнее доверить работу мастерам?»

Фаза 2 (после уточнения):  
«Отлично! Записаться можно по кнопке ниже ⬇️ Администратор подберёт удобное время и уточнит стоимость! [ЗАПИСЬ]»

Фаза 3:  
«Спасибо за интерес! 🙏 Администратор свяжется для подтверждения записи и согласует удобное время.»

#ФОРМАТ КНОПКИ ЗАПИСИ (ТЕХНИЧЕСКОЕ ПРАВИЛО)
• Когда предлагаешь клиенту записаться — НЕ вставляй ссылку на YClients и номер телефона в текст ответа.
• Вместо этого напиши «Записаться можно по кнопке ниже ⬇️» и поставь маркер [ЗАПИСЬ] в самом конце ответа.
• Маркер [ЗАПИСЬ] будет автоматически заменён на интерактивные кнопки для клиента.
• Другие ссылки (Яндекс-карты, Ozon, Telegram-канал, сайт студии) — вставляй в текст как обычно.
"""

# ================== ГОТОВЫЙ ОТВЕТ ПО ПРЕИМУЩЕСТВАМ ==================
ADVANTAGES_ANSWER = (
    "Преимущества BEARLAKE, за которые нас выбирают:\n\n"
    "1️⃣ Полный цикл услуг в одном месте — от быстрой мойки до полировки, "
    "керамики, PPF, химчистки и ухода за стеклами\n"
    "2️⃣ Реальная экспертиза — работаем премиальной химией OPT, NXTZEN, TAC System "
    "и подбираем решения под задачу, а не «по шаблону»\n"
    "3️⃣ Два формата обслуживания — услуги мастеров и самообслуживание 24/7 "
    "с профессиональным оборудованием\n"
    "4️⃣ Комфорт клиента — зона отдыха, Wi‑Fi, напитки/перекусы, вентиляция и климат-контроль\n"
    "5️⃣ Прозрачный подход — честно объясняем, что входит в услугу, какие есть опции "
    "и что даст максимальный результат именно для вашего авто ✅\n\n"
    "Записаться можно по кнопке ниже ⬇️"
)

PRICES_ANSWER = (
    "Актуальные стартовые цены по услугам:\n\n"
    "• Быстрая детейлинг-мойка — от 3 000 ₽\n"
    "• Комплексная детейлинг-мойка Bearlake — от 8 000 ₽\n"
    "• Деконтаминация ЛКП — от 6 000 ₽\n"
    "• Мойка подкапотного пространства — от 9 000 ₽\n"
    "• Полировка ЛКП (Light Polish) — от 25 000 ₽\n"
    "• Керамические покрытия — от 5 000 ₽\n"
    "• Детейлинг чистка интерьера — от 28 000 ₽\n"
    "• Защитные полиуретановые пленки — от 85 000 ₽\n"
    "• Бронирование лобового стекла — от 30 000 ₽\n"
    "• Ремонт трещин и сколов стекла — от 3 500 ₽\n"
    "• Самообслуживание 24/7 — 700 ₽/час или 900 ₽/час\n\n"
    "Сроки по ключевым услугам:\n"
    "• Быстрая мойка — от 1 часа\n"
    "• Комплексная мойка — от 4 часов\n"
    "• Деконтаминация — от 2 часов\n"
    "• Мойка подкапотного пространства — от 4 часов\n"
    "• Полировка — от 12 часов\n"
    "• Керамические покрытия — от 4 часов\n"
    "• Оклейка PPF — от 2 дней\n"
    "• Химчистка — 1 день\n"
    "• Бронирование стекла — от 1 дня\n\n"
    "*Стоимость услуг мастеров (кроме самообслуживания) зависит от класса авто "
    "(S / M / L) и уточняется после осмотра.*\n\n"
    "Если хотите, сориентирую по оптимальному набору услуг под ваш бюджет и сценарий эксплуатации.\n"
    "Записаться можно по кнопке ниже ⬇️"
)

SHOP_RECOMMEND_ANSWER = (
    "Рекомендуем проверенную профессиональную автохимию и аксессуары, "
    "которыми мы работаем в студии каждый день.\n\n"
    "Основные бренды:\n"
    "• OPT / Optimum Polymer Technologies\n"
    "• NXTZEN\n"
    "• TAC System\n"
    "• iK\n"
    "• Detailers of Russia (DOFR)\n\n"
    "Подберем решение под вашу задачу: поддерживающая мойка, "
    "глубокая очистка, защита ЛКП или уход за салоном."
)

SHOP_OZON_ANSWER = (
    "Bearlake на Ozon — выбор профессионалов для вашего автомобиля.\n\n"
    "В магазине собрали только те товары, которыми работаем сами в студии: "
    "каждый состав и аксессуар проходит практические тесты на реальных задачах.\n\n"
    "Топ-бренды в ассортименте:\n"
    "• OPT (Optimum Polymer Technologies) и линейка Opti-Coat\n"
    "• NXTZEN (Австралия)\n"
    "• TAC System (Корея)\n"
    "• iK — профессиональные распылители\n"
    "• Detailers of Russia (DOFR) — микрофибры, полотенца, аппликаторы\n"
    "• и другие проверенные позиции\n\n"
    "Подходит и профессионалам, и автолюбителям, которые ценят качественный уход.\n"
    "На Ozon удобно и выгодно: быстрая доставка в удобный пункт выдачи.\n\n"
    "Если сомневаетесь в выборе, помогу подобрать химию и аксессуары под ваши задачи."
)

SHOP_DISCOUNTS_ANSWER = (
    "Актуальные скидки:\n\n"
    "• В офлайн-магазине — 5% для постоянных клиентов\n"
    "• На Ozon при первом заказе — 5% по промокоду BEARLAKE\n\n"
    "Актуальные акции также публикуем в Telegram-канале."
)

# ================== ИНИЦИАЛИЗАЦИЯ OPENAI ==================
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

FULL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\n# БАЗА ЗНАНИЙ СТУДИИ BEARLAKE DETAILING & SHOP\n"
    + KNOWLEDGE_BASE
)

# ================== ФУНКЦИЯ ДЛЯ GPT ==================
FALLBACK_RESPONSE = (
    "Извините, сейчас не удалось обработать запрос. "
    "Попробуйте, пожалуйста, ещё раз чуть позже."
)


async def _call_openai(messages: list[dict]) -> str:
    """Непосредственный вызов OpenAI API."""
    completion = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens
    log_token_usage(prompt_tokens, completion_tokens, total_tokens)
    return completion.choices[0].message.content.strip()


async def get_gpt_response(user_text: str, history: list[dict]) -> str:
    """Отправляет запрос в GPT с таймаутом OPENAI_TIMEOUT секунд."""
    messages = [{"role": "system", "content": FULL_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        return await asyncio.wait_for(
            _call_openai(messages),
            timeout=OPENAI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Таймаут OpenAI (%d сек) для запроса: %r", OPENAI_TIMEOUT, user_text[:80])
        return FALLBACK_RESPONSE
    except Exception as e:
        logger.exception("Ошибка при запросе к OpenAI: %s", e)
        return FALLBACK_RESPONSE

# ================== КЭШ ОТВЕТОВ НА КНОПКИ ==================
_button_cache: dict[str, tuple[str, float]] = {}


def _get_from_cache(button_data: str) -> str | None:
    """Возвращает кэшированный ответ или None, если кэш пуст/истёк."""
    if button_data not in _button_cache:
        return None
    response, cached_at = _button_cache[button_data]
    if time.time() - cached_at >= CACHE_TTL:
        del _button_cache[button_data]
        return None
    return response


def _put_to_cache(button_data: str, response: str) -> None:
    _button_cache[button_data] = (response, time.time())


# ================== РАБОТА С ИСТОРИЕЙ ДИАЛОГА ==================
def get_history(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    if "history" not in context.chat_data:
        context.chat_data["history"] = []
    return context.chat_data["history"]


def append_to_history(context: ContextTypes.DEFAULT_TYPE, role: str, content: str) -> None:
    history = get_history(context)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        context.chat_data["history"] = history[-MAX_HISTORY:]


def _check_consent(context: ContextTypes.DEFAULT_TYPE, user_id) -> bool:
    """Проверяет согласие: сначала кэш в chat_data, потом CSV."""
    if context.chat_data.get("consent"):
        return True
    if has_consent(user_id):
        context.chat_data["consent"] = True
        return True
    return False


def _ensure_fresh_session(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Сбрасывает сессию при наступлении нового дня."""
    today = dt.date.today().isoformat()
    if context.chat_data.get("last_active_date") == today:
        return
    context.chat_data["history"] = []
    context.chat_data["rated"] = False
    context.chat_data["booking_confirmed"] = False
    context.chat_data["booking_logged"] = False
    context.chat_data["greeted"] = False
    context.chat_data["awaiting_consultation_priority"] = False
    context.chat_data["consultation_base_request"] = ""
    context.chat_data["consultation_priority"] = ""
    context.chat_data["last_active_date"] = today
    _cancel_user_timers(context.job_queue, chat_id)
    logger.info("Новый день — сессия сброшена для чата %s", chat_id)


# ================== ТАЙМЕРЫ: ОЦЕНКА И ПОВТОРНОЕ ВОВЛЕЧЕНИЕ ==================
def _cancel_jobs(job_queue, name: str) -> None:
    """Отменяет все запланированные задачи с данным именем."""
    if not job_queue:
        return
    for job in job_queue.get_jobs_by_name(name):
        job.schedule_removal()


def _cancel_user_timers(job_queue, chat_id: int) -> None:
    """Отменяет все таймеры (оценка + follow-up) для данного чата."""
    _cancel_jobs(job_queue, f"rating_{chat_id}")
    _cancel_jobs(job_queue, f"followup_{chat_id}")


def _schedule_rating(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Планирует запрос оценки через RATING_DELAY секунд после последнего сообщения."""
    if not context.job_queue:
        return
    if not context.chat_data.get("booking_confirmed"):
        return
    if context.chat_data.get("rated"):
        return
    name = f"rating_{chat_id}"
    _cancel_jobs(context.job_queue, name)
    context.job_queue.run_once(
        _send_rating_request,
        when=RATING_DELAY,
        chat_id=chat_id,
        name=name,
    )


def _schedule_followup(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                        user_id: int, topic: str) -> None:
    """Планирует напоминание о записи через FOLLOWUP_DELAY секунд."""
    if not context.job_queue:
        return
    name = f"followup_{chat_id}"
    _cancel_jobs(context.job_queue, name)
    context.job_queue.run_once(
        _send_followup,
        when=FOLLOWUP_DELAY,
        chat_id=chat_id,
        name=name,
        data={"user_id": user_id, "topic": topic},
    )


async def _send_rating_request(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job-callback: отправляет запрос оценки."""
    chat_id = context.job.chat_id
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "Как вы оцениваете консультацию? 🙏\n"
                "Ваша оценка поможет нам стать лучше!"
            ),
            reply_markup=RATING_KEYBOARD,
        )
    except Exception as e:
        logger.warning("Не удалось отправить запрос оценки в %s: %s", chat_id, e)


async def _send_followup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job-callback: напоминание о записи через 24 часа."""
    chat_id = context.job.chat_id
    data = context.job.data or {}
    user_id = data.get("user_id")

    if user_id and not has_consent(user_id):
        return

    topic = data.get("topic", "нашими услугами")
    text = (
        f"Добрый день! 👋\n\n"
        f"Вы недавно интересовались {topic}. "
        f"Хотите записаться на удобное время?\n\n"
        f"Записаться можно по кнопке ниже ⬇️"
        f"{PHONE_LINE}"
    )
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=BOOKING_KEYBOARD,
        )
        logger.info("Follow-up отправлен в чат %s (тема: %s)", chat_id, topic)
    except Exception as e:
        logger.warning("Не удалось отправить follow-up в %s: %s", chat_id, e)


def _estimate_booking_amount(service_topic: str) -> int:
    """Оценка минимальной выручки по теме интереса."""
    topic = (service_topic or "").lower()
    pricing_rules = [
        (("быстр", "двухфаз"), 3000),
        (("комплексн", "мойк"), 8000),
        (("деконтам"), 6000),
        (("подкапот"), 9000),
        (("полиров",), 25000),
        (("керамич",), 5000),
        (("интерьер", "салон", "химчист"), 28000),
        (("пленк", "ppf", "брон"), 30000),
        (("скол", "трещин"), 3500),
        (("самообслуж",), 700),
    ]
    for keywords, amount in pricing_rules:
        if any(keyword in topic for keyword in keywords):
            return amount
    return 3000


# ================== ОБРАБОТЧИКИ КОМАНД ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    args = context.args
    source = args[0] if args else "unknown"

    user_id = update.effective_user.id if update.effective_user else "unknown"
    logger.info("Пользователь %s начал чат из источника: %s", user_id, source)

    log_new_client(user_id, source)

    context.chat_data["history"] = []
    context.chat_data["rated"] = False
    context.chat_data["booking_confirmed"] = False
    context.chat_data["booking_logged"] = False
    context.chat_data["awaiting_consultation_priority"] = False
    context.chat_data["consultation_base_request"] = ""
    context.chat_data["consultation_priority"] = ""
    context.chat_data["last_active_date"] = dt.date.today().isoformat()
    _cancel_user_timers(context.job_queue, update.effective_chat.id)

    if has_consent(user_id):
        context.chat_data["consent"] = True
        context.chat_data["greeted"] = True
        greeting = (
            "С возвращением! 👋\n\n"
            "Я — виртуальный помощник студии "
            "BEARLAKE DETAILING & SHOP 🐻\n\n"
            "Помогу подобрать услугу, расскажу о ценах "
            "и условиях, а также запишу на удобное время.\n\n"
            "Выберите интересующий раздел ⬇️ "
            "или просто напишите свой вопрос!"
        )
        append_to_history(context, "assistant", greeting)
        await update.message.reply_text(greeting, reply_markup=GREETING_KEYBOARD)
    else:
        consent_text = (
            "Здравствуйте! 👋\n\n"
            "Я — виртуальный помощник студии "
            "BEARLAKE DETAILING & SHOP 🐻\n\n"
            "Для работы бота мы обрабатываем:\n"
            "• Ваш Telegram ID\n"
            "• Текст сообщений в чате\n"
            "• Дату и время обращений\n\n"
            "Данные используются исключительно для "
            "консультирования по услугам студии.\n\n"
            "📄 Подробнее — /privacy\n\n"
            "Нажимая «Принимаю», вы даёте согласие "
            "на обработку персональных данных "
            "в соответствии с ФЗ-152."
        )
        await update.message.reply_text(consent_text, reply_markup=CONSENT_KEYBOARD)


def extract_booking_marker(text: str) -> tuple[str, bool]:
    """Убирает маркер [ЗАПИСЬ] из текста. Возвращает (чистый текст, был ли маркер)."""
    if BOOKING_MARKER in text:
        return text.replace(BOOKING_MARKER, "").strip(), True
    return text, False


URL_RE = re.compile(r"https?://[^\s)>\]]+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _extract_links(text: str) -> tuple[str, list[tuple[str | None, str]]]:
    """Извлекает URL (в т.ч. markdown-ссылки), удаляя их из текста."""
    links: list[tuple[str | None, str]] = []

    def md_repl(match: re.Match) -> str:
        label = match.group(1).strip()
        url = match.group(2).strip()
        links.append((label, url))
        return label

    text = MD_LINK_RE.sub(md_repl, text)

    def url_repl(match: re.Match) -> str:
        raw = match.group(0)
        url = raw.rstrip(".,);")
        if url:
            links.append((None, url))
        trailing = raw[len(url):]
        return trailing

    text = URL_RE.sub(url_repl, text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, links


def _button_from_link(label: str | None, url: str) -> InlineKeyboardButton:
    """Подбирает кнопку-иконку под известные ссылки."""
    low = url.lower()
    if "n1024167.yclients.com" in low:
        return InlineKeyboardButton(
            "📅 Записаться онлайн",
            web_app=WebAppInfo(url="https://n1024167.yclients.com/"),
        )
    if "yandex.ru/maps" in low:
        return InlineKeyboardButton("📍 Яндекс Карты", url=url)
    if "t.me/bearlake_detailing" in low:
        return InlineKeyboardButton("🔥 Telegram-канал", url=url)
    if "ozon.ru" in low:
        return InlineKeyboardButton("🛒 Ozon", url=url)
    if "bearlake.clients.site" in low:
        return InlineKeyboardButton("🏪 Сайт студии", url=url)
    if "avito.ru" in low:
        return InlineKeyboardButton("🧾 Avito", url=url)
    if label:
        short = label.strip()
        if len(short) > 22:
            short = short[:19] + "..."
        return InlineKeyboardButton(f"🔗 {short}", url=url)
    return InlineKeyboardButton("🔗 Открыть ссылку", url=url)


def _button_key(button: InlineKeyboardButton) -> tuple[str, str, str]:
    web_app = getattr(button, "web_app", None)
    if web_app and getattr(web_app, "url", None):
        return ("web_app", button.text, web_app.url)
    return ("url", button.text, button.url or "")


def _build_reply_markup(
    show_booking: bool,
    links: list[tuple[str | None, str]],
    base_markup: InlineKeyboardMarkup | None = None,
) -> InlineKeyboardMarkup | None:
    """Собирает клавиатуру: базовые кнопки + запись + ссылки-иконки."""
    rows: list[list[InlineKeyboardButton]] = []
    if base_markup:
        rows.extend([list(row) for row in base_markup.inline_keyboard])
    if show_booking:
        rows.extend([list(row) for row in BOOKING_KEYBOARD.inline_keyboard])

    # Убираем дубли по всем кнопкам (база + запись + ссылки),
    # чтобы не было двух одинаковых "Записаться онлайн".
    dedup_rows: list[list[InlineKeyboardButton]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        dedup_row: list[InlineKeyboardButton] = []
        for btn in row:
            key = _button_key(btn)
            if key in seen:
                continue
            seen.add(key)
            dedup_row.append(btn)
        if dedup_row:
            dedup_rows.append(dedup_row)
    rows = dedup_rows

    for label, url in links:
        button = _button_from_link(label, url)
        key = _button_key(button)
        if key in seen:
            continue
        rows.append([button])
        seen.add(key)

    return InlineKeyboardMarkup(rows) if rows else None


def _with_detail_button(
    markup: InlineKeyboardMarkup | None,
    callback_data: str,
) -> InlineKeyboardMarkup | None:
    """Добавляет кнопки 'Подробнее/Пока достаточно' для краткой карточки услуги."""
    if callback_data not in DETAILABLE_CALLBACKS:
        return markup
    detail_btn = InlineKeyboardButton("✅ Да, хочу подробнее", callback_data=f"detail::{callback_data}")
    skip_btn = InlineKeyboardButton("👌 Пока достаточно", callback_data=f"detail_skip::{callback_data}")
    if markup is None:
        return InlineKeyboardMarkup([[detail_btn], [skip_btn]])

    rows = [list(row) for row in markup.inline_keyboard]
    seen = {_button_key(btn) for row in rows for btn in row}
    detail_key = _button_key(detail_btn)
    skip_key = _button_key(skip_btn)
    to_add: list[InlineKeyboardButton] = []
    if detail_key not in seen:
        rows.append([detail_btn])
    if skip_key not in seen:
        rows.append([skip_btn])
    return InlineKeyboardMarkup(rows)


def _with_detail_offer(text: str, callback_data: str) -> str:
    """Добавляет callout к кнопке подробностей после краткого ответа."""
    if callback_data not in DETAILABLE_CALLBACKS:
        return text
    low = text.lower()
    if "хотите узнать об услуге больше" in low or "хочу подробнее" in low:
        return text
    return f"{text}\n\nХотите узнать об услуге больше? Нажмите «✅ Да, хочу подробнее» 👇"


def _cleanup_outgoing_text(text: str) -> str:
    """Нормализует markdown-артефакты для plain-text Telegram сообщений."""
    cleaned = text.replace("**", "").replace("__", "")
    cleaned = re.sub(r"(?m)^\s*-\s+", "• ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\.\s+", "• ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _apply_cta_by_context(text: str, callback_data: str | None = None) -> str:
    """Добавляет CTA только для разделов, где он уместен."""
    if callback_data and callback_data in NO_CTA_CALLBACKS:
        return text
    return _with_sales_cta(text)


async def send_answer(
    message,
    text: str,
    force_booking: bool = False,
    user_lang: str = "ru",
) -> None:
    """Отправляет ответ, автоматически добавляя кнопки записи при маркере."""
    clean_text, has_marker = extract_booking_marker(text)
    clean_text = _cleanup_outgoing_text(clean_text)
    clean_text = _with_sales_cta(clean_text)
    clean_text, links = _extract_links(clean_text)
    show_booking = has_marker or force_booking
    if show_booking:
        clean_text += PHONE_LINE
    clean_text = await _finalize_response_text(clean_text, user_lang)
    reply_markup = _build_reply_markup(show_booking, links)
    await message.reply_text(
        clean_text,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


def _with_sales_cta(text: str) -> str:
    """Добавляет мягкий продающий CTA в конце ответа."""
    default_cta = (
        "Готов помочь выбрать лучший вариант под ваш авто "
        "и сразу сориентировать по цене и срокам ✅"
    )
    text_lower = text.lower()

    if "мойк" in text_lower:
        cta = "Если хотите, подскажу какой формат мойки лучше именно под вашу задачу и время ⏱️"
    elif "полиров" in text_lower:
        cta = "Могу сориентировать, какой эффект полировки получите на вашем авто и сколько это займет ✨"
    elif "керамич" in text_lower or "пленк" in text_lower or "защит" in text_lower:
        cta = "Если хотите, подберу оптимальную защиту по сроку службы и бюджету 🛡️"
    elif "салон" in text_lower or "химчист" in text_lower or "интерьер" in text_lower:
        cta = "Подскажу оптимальный формат ухода за салоном под текущее состояние и бюджет 🧼"
    elif "самообслуж" in text_lower or "700 ₽/час" in text_lower or "900 ₽/час" in text_lower:
        cta = "Если хотите, подскажу оптимальный тариф самообслуживания под ваш сценарий 🚿"
    elif "цен" in text_lower or "прайс" in text_lower or "стоим" in text_lower:
        cta = "Могу сразу сориентировать по цене и срокам именно для вашего класса авто (S / M / L) 💰"
    else:
        cta = default_cta

    # В чисто информационных блоках не добавляем продающий хвост.
    no_cta_markers = (
        "мы находимся по адресу",
        "яндекс-карты",
        "хотите увидеть результат «до/после»",
        "в студии есть кофе",
        "да, есть бесплатный wi-fi",
        "да, предусмотрены условия для маломобильных",
        "актуальные скидки",
    )
    if any(marker in text_lower for marker in no_cta_markers):
        return text

    if cta in text or default_cta in text:
        return text
    if text == FALLBACK_RESPONSE:
        return text
    if "выберите нужный раздел в меню ниже" in text_lower:
        return text
    if "выберите раздел магазина по кнопкам ниже" in text_lower:
        return text
    if "записаться можно по кнопке ниже" in text_lower:
        return text
    return f"{text}\n\n{cta}"


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на кнопки меню и подменю."""
    query = update.callback_query
    try:
        await query.answer()
    except Exception:
        pass

    user_id = update.effective_user.id if update.effective_user else "unknown"

    if query.data and query.data.startswith("detail::"):
        source_cb = query.data.split("::", 1)[1]
        user_text = MENU_PROMPTS.get(source_cb) or f"Расскажите подробно про {TOPIC_LABELS.get(source_cb, 'услугу')}"
        kb_answer, kb_score, kb_question = _find_kb_answer(user_text)
        if kb_answer:
            _log_kb_match(user_id, user_text, kb_question, kb_score, "matched")
        elif kb_question:
            _log_kb_match(user_id, user_text, kb_question, kb_score, "no_match")

        history = get_history(context)
        append_to_history(context, "user", user_text)
        gpt_detail_query = (
            f"{user_text}\n\n"
            "Дайте подробный ответ по услуге на основе базы знаний без сокращений: "
            "что это, что входит, сроки, стоимость от, когда рекомендуется. "
            "Если есть список этапов — перечислите его полностью.\n"
            "В конце обязательно добавьте: 'Записаться можно по кнопке ниже ⬇️ [ЗАПИСЬ]'."
        )
        if kb_answer:
            gpt_detail_query += f"\n\nОпорный фрагмент из базы знаний:\n{kb_answer}"

        detail_text = await get_gpt_response(gpt_detail_query, history)
        if detail_text == FALLBACK_RESPONSE and kb_answer:
            detail_text = f"{kb_answer}\n\nЗаписаться можно по кнопке ниже ⬇️ [ЗАПИСЬ]"
        append_to_history(context, "assistant", detail_text)
        await send_answer(query.message, detail_text)

        topic = TOPIC_LABELS.get(source_cb, "услугой")
        context.chat_data["last_topic"] = topic
        _schedule_followup(context, query.message.chat_id, user_id, topic)
        return

    if query.data and query.data.startswith("detail_skip::"):
        source_cb = query.data.split("::", 1)[1]
        short_ack = (
            "Отлично, оставим краткий формат 👍\n"
            "Если захотите, в любой момент дам подробный разбор именно по вашему авто.\n\n"
            "Записаться можно по кнопке ниже ⬇️ [ЗАПИСЬ]"
        )
        append_to_history(context, "assistant", short_ack)
        await send_answer(query.message, short_ack)
        topic = TOPIC_LABELS.get(source_cb, "услугой")
        context.chat_data["last_topic"] = topic
        _schedule_followup(context, query.message.chat_id, user_id, topic)
        return

    if query.data == "consent_accept":
        log_consent(user_id)
        context.chat_data["consent"] = True
        context.chat_data["greeted"] = True
        context.chat_data["history"] = []
        context.chat_data["booking_confirmed"] = False
        context.chat_data["booking_logged"] = False
        context.chat_data["awaiting_consultation_priority"] = False
        context.chat_data["consultation_base_request"] = ""
        context.chat_data["consultation_priority"] = ""
        greeting = (
            "Спасибо! Согласие принято ✅\n\n"
            "Помогу подобрать услугу, расскажу о ценах "
            "и условиях, а также запишу на удобное время.\n\n"
            "Выберите интересующий раздел ⬇️ "
            "или просто напишите свой вопрос!"
        )
        append_to_history(context, "assistant", greeting)
        await query.message.reply_text(greeting, reply_markup=GREETING_KEYBOARD)
        return

    if query.data == "booking_done":
        if not context.chat_data.get("booking_logged"):
            service_topic = context.chat_data.get("last_topic", "не указано")
            amount_from = _estimate_booking_amount(service_topic)
            log_booking(user_id, service_topic, amount_from)
            context.chat_data["booking_logged"] = True
        context.chat_data["booking_confirmed"] = True
        context.chat_data["rated"] = False
        _cancel_jobs(context.job_queue, f"followup_{query.message.chat_id}")
        _schedule_rating(context, query.message.chat_id)
        await query.message.reply_text(
            "Отлично, спасибо! ✅\n"
            "Через пару минут попрошу коротко оценить консультацию."
        )
        return

    if query.data and query.data.startswith("rate_"):
        rating = int(query.data.split("_")[1])
        log_rating(user_id, rating)
        logger.info("Оценка от %s: %d ⭐", user_id, rating)
        context.chat_data["rated"] = True
        _cancel_jobs(context.job_queue, f"rating_{query.message.chat_id}")
        if rating >= 4:
            text = "Спасибо за высокую оценку! 🌟 Рады, что смогли помочь!"
        else:
            text = (
                "Спасибо за отзыв! 🙏 Мы постараемся стать лучше.\n"
                "Если остались вопросы — напишите, или свяжитесь "
                "с администратором: +7 (926) 021-60-00"
            )
        await query.message.reply_text(text)
        return

    if not _check_consent(context, user_id):
        await query.message.reply_text(NO_CONSENT_TEXT)
        return

    _ensure_fresh_session(context, query.message.chat_id)

    if query.data == "menu_advantages":
        log_button_click(user_id, query.data)
        append_to_history(context, "assistant", ADVANTAGES_ANSWER)
        await send_answer(query.message, ADVANTAGES_ANSWER, force_booking=True)
        chat_id = query.message.chat_id
        _schedule_followup(context, chat_id, user_id, "нашими преимуществами")
        return

    if query.data in {"menu_prices", "sub_all_prices"}:
        log_button_click(user_id, query.data)
        user_text = MENU_PROMPTS.get(query.data, "Какие у вас цены на услуги?")
        append_to_history(context, "user", user_text)
        append_to_history(context, "assistant", PRICES_ANSWER)
        await send_answer(query.message, PRICES_ANSWER, force_booking=True)
        chat_id = query.message.chat_id
        _schedule_followup(context, chat_id, user_id, "ценами на услуги")
        return

    if query.data == "menu_shop":
        log_button_click(user_id, query.data)
        answer = _with_sales_cta("Выберите раздел магазина по кнопкам ниже ⬇️")
        append_to_history(context, "assistant", answer)
        await query.message.reply_text(
            answer,
            reply_markup=SHOP_KEYBOARD,
            disable_web_page_preview=True,
        )
        chat_id = query.message.chat_id
        _schedule_followup(context, chat_id, user_id, "магазином автохимии")
        return

    if query.data in {"sub_shop_recommend", "sub_shop_ozon", "sub_shop_discounts"}:
        log_button_click(user_id, query.data)
        user_text = MENU_PROMPTS.get(query.data, "")
        if user_text:
            append_to_history(context, "user", user_text)
        if query.data == "sub_shop_recommend":
            answer = _apply_cta_by_context(SHOP_RECOMMEND_ANSWER, query.data)
        elif query.data == "sub_shop_ozon":
            answer = _apply_cta_by_context(SHOP_OZON_ANSWER, query.data)
        else:
            answer = _apply_cta_by_context(SHOP_DISCOUNTS_ANSWER, query.data)
        append_to_history(context, "assistant", answer)
        await query.message.reply_text(
            answer,
            reply_markup=SHOP_KEYBOARD,
            disable_web_page_preview=True,
        )
        chat_id = query.message.chat_id
        topic = TOPIC_LABELS.get(query.data, "магазином автохимии")
        _schedule_followup(context, chat_id, user_id, topic)
        return

    user_text = MENU_PROMPTS.get(query.data)
    if not user_text:
        return

    chat_id = query.message.chat_id

    logger.info("Кнопка от %s: %s → %r", user_id, query.data, user_text)
    log_button_click(user_id, query.data)
    log_question(user_id, user_text)

    ai_first = query.data in AI_FIRST_BUTTON_CALLBACKS
    static_answer = STATIC_MENU_ANSWERS.get(query.data)
    if static_answer and not ai_first:
        answer = static_answer
    else:
        answer = None
        if ai_first:
            kb_answer, kb_score, kb_question = _find_kb_answer(user_text)
            if kb_answer:
                answer = kb_answer
                _log_kb_match(user_id, user_text, kb_question, kb_score, "matched")
            elif kb_question:
                _log_kb_match(user_id, user_text, kb_question, kb_score, "no_match")

        if answer is None:
            cached = _get_from_cache(query.data)
            if cached:
                logger.info("Кэш-попадание: %s", query.data)
                answer = cached
            else:
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                except Exception as e:
                    logger.warning("Не удалось отправить 'typing': %s", e)

                history = get_history(context)
                answer = await get_gpt_response(user_text, history)

                if answer != FALLBACK_RESPONSE:
                    _put_to_cache(query.data, answer)

    append_to_history(context, "user", user_text)
    append_to_history(context, "assistant", answer)

    sub_menu = SUB_MENUS.get(query.data)
    if sub_menu:
        clean_text, has_marker = extract_booking_marker(answer)
        clean_text = _cleanup_outgoing_text(clean_text)
        clean_text = _apply_cta_by_context(clean_text, query.data)
        clean_text = _with_detail_offer(clean_text, query.data)
        clean_text, links = _extract_links(clean_text)
        reply_markup = _build_reply_markup(has_marker, links, base_markup=sub_menu)
        reply_markup = _with_detail_button(reply_markup, query.data)
        await query.message.reply_text(
            clean_text,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )
    else:
        if query.data in DETAILABLE_CALLBACKS:
            clean_text, has_marker = extract_booking_marker(answer)
            clean_text = _cleanup_outgoing_text(clean_text)
            clean_text = _apply_cta_by_context(clean_text, query.data)
            clean_text = _with_detail_offer(clean_text, query.data)
            clean_text, links = _extract_links(clean_text)
            reply_markup = _build_reply_markup(has_marker, links)
            reply_markup = _with_detail_button(reply_markup, query.data)
            await query.message.reply_text(
                clean_text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
        else:
            await send_answer(query.message, answer)

    topic = TOPIC_LABELS.get(query.data, "нашими услугами")
    context.chat_data["last_topic"] = topic
    _schedule_followup(context, chat_id, user_id, topic)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return

    user_text = update.message.text.strip()
    user_lang = _detect_user_language(user_text)
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else "unknown"

    if not _check_consent(context, user_id):
        consent_text = await _finalize_response_text(NO_CONSENT_TEXT, user_lang)
        await update.message.reply_text(consent_text)
        return

    _ensure_fresh_session(context, chat_id)

    if not context.chat_data.get("greeted"):
        context.chat_data["greeted"] = True
        greeting = (
            "С возвращением! 👋\n\n"
            "Выберите раздел или задайте вопрос ⬇️"
        )
        greeting = await _finalize_response_text(greeting, user_lang)
        await update.message.reply_text(greeting, reply_markup=GREETING_KEYBOARD)

    logger.info("Сообщение от %s: %r", user_id, user_text)

    log_question(user_id, user_text)

    if _is_short_followup_request(user_text):
        last_topic = context.chat_data.get("last_topic", "")
        contextual = _context_followup_answer(last_topic)
        if contextual:
            contextual = await _finalize_response_text(contextual, user_lang)
            append_to_history(context, "user", user_text)
            append_to_history(context, "assistant", contextual)
            await send_answer(update.message, contextual, user_lang=user_lang)
            _schedule_followup(context, chat_id, user_id, last_topic or "нашими услугами")
            return

    if context.chat_data.get("awaiting_consultation_priority"):
        priority = _detect_consultation_priority(user_text)
        if priority:
            base_request = context.chat_data.get("consultation_base_request") or "подбор услуг по авто"
            history = get_history(context)
            append_to_history(context, "user", user_text)
            refined_query = _build_refined_consultation_query(base_request, priority, user_text)
            answer = await get_gpt_response(refined_query, history)
            append_to_history(context, "assistant", answer)
            await send_answer(update.message, answer, user_lang=user_lang)
            context.chat_data["awaiting_consultation_priority"] = False
            context.chat_data["consultation_priority"] = priority
            context.chat_data["last_topic"] = "подбором услуг под приоритет клиента"
            _schedule_followup(context, chat_id, user_id, "подбором услуг под приоритет клиента")
            return
        if len(user_text) <= 80:
            answer = (
                "Чтобы продолжить подбор, напишите одним словом приоритет: "
                "скорость, результат, защита или бюджет."
            )
            answer = await _finalize_response_text(answer, user_lang)
            append_to_history(context, "user", user_text)
            append_to_history(context, "assistant", answer)
            await update.message.reply_text(answer)
            return
        context.chat_data["awaiting_consultation_priority"] = False

    if not _is_on_topic(user_text):
        answer = (
            "Вопрос не по теме детейлинга. "
            "Давайте вернёмся к услугам студии: мойка, полировка, защита, "
            "салон, самообслуживание, цены, адрес или запись."
        )
        answer = await _finalize_response_text(answer, user_lang)
        append_to_history(context, "user", user_text)
        append_to_history(context, "assistant", answer)
        await update.message.reply_text(answer, reply_markup=GREETING_KEYBOARD)
        _schedule_followup(context, chat_id, user_id, "нашими услугами")
        return

    text_lower = user_text.lower()
    priority_intent = _detect_priority_intent(text_lower)
    if priority_intent:
        intent, candidates = priority_intent, [priority_intent]
    else:
        intent, candidates = _detect_text_intent(text_lower)
    kb_answer, kb_score, kb_question = _find_kb_answer(user_text)
    is_complex = _is_complex_consultation(user_text)

    prefer_static = bool(priority_intent) or (intent in STRICT_TEXT_INTENTS if intent else False)

    # Для свободных формулировок: сначала KB, затем GPT.
    if kb_answer and not prefer_static and not is_complex:
        _log_kb_match(user_id, user_text, kb_question, kb_score, "matched")
        append_to_history(context, "user", user_text)
        append_to_history(context, "assistant", kb_answer)
        await send_answer(update.message, kb_answer, user_lang=user_lang)
        context.chat_data["last_topic"] = "вопросом по услугам"
        _schedule_followup(context, chat_id, user_id, "вопросом по услугам")
        logger.info("Ответ из базы знаний (score=%.2f) для %s: %r", kb_score, user_id, user_text[:80])
        return
    if kb_question:
        _log_kb_match(user_id, user_text, kb_question, kb_score, "no_match")
    elif intent:
        _log_kb_match(user_id, user_text, None, 0.0, "fallback_intent")

    if not prefer_static:
        history = get_history(context)
        append_to_history(context, "user", user_text)
        gpt_query = _build_consultation_query(user_text) if is_complex else user_text
        answer = await get_gpt_response(gpt_query, history)
        if is_complex:
            followup = _consultation_followup_question(user_text, answer)
            if followup not in answer:
                answer = f"{answer}\n\n{followup}"
            context.chat_data["awaiting_consultation_priority"] = True
            context.chat_data["consultation_base_request"] = user_text
        append_to_history(context, "assistant", answer)
        await send_answer(update.message, answer, user_lang=user_lang)
        context.chat_data["last_topic"] = "вопросом по услугам"
        _schedule_followup(context, chat_id, user_id, "вопросом по услугам")
        return

    if not intent:
        if candidates:
            candidate_labels = [
                f"• {INTENT_CLARIFY_LABELS.get(candidate, candidate)}"
                for candidate in candidates[:4]
            ]
            answer = (
                "Хочу ответить точно, но запрос сейчас двусмысленный.\n"
                "Уточните, пожалуйста, что именно интересует:\n"
                f"{chr(10).join(candidate_labels)}"
            )
        else:
            answer = (
                "Чтобы ответить максимально точно, выберите нужный раздел в меню ниже ⬇️\n\n"
                "Или напишите короче, например: «цены», «мойка», «полировка», "
                "«самообслуживание», «адрес», «скидки»."
            )
        answer = _with_sales_cta(answer)
        answer = await _finalize_response_text(answer, user_lang)
        append_to_history(context, "user", user_text)
        append_to_history(context, "assistant", answer)
        await update.message.reply_text(answer, reply_markup=GREETING_KEYBOARD)
        _schedule_followup(context, chat_id, user_id, "нашими услугами")
        return

    append_to_history(context, "user", user_text)

    if intent == "menu_advantages":
        answer = ADVANTAGES_ANSWER
        append_to_history(context, "assistant", answer)
        await send_answer(update.message, answer, force_booking=True, user_lang=user_lang)
    elif intent in {"menu_prices", "sub_all_prices"}:
        answer = PRICES_ANSWER
        append_to_history(context, "assistant", answer)
        await send_answer(update.message, answer, force_booking=True, user_lang=user_lang)
    elif intent == "menu_shop":
        answer = _with_sales_cta("Выберите раздел магазина по кнопкам ниже ⬇️")
        answer = await _finalize_response_text(answer, user_lang)
        append_to_history(context, "assistant", answer)
        await update.message.reply_text(answer, reply_markup=SHOP_KEYBOARD)
    elif intent in {"sub_shop_recommend", "sub_shop_ozon", "sub_shop_discounts"}:
        if intent == "sub_shop_recommend":
            answer = _with_sales_cta(SHOP_RECOMMEND_ANSWER)
        elif intent == "sub_shop_ozon":
            answer = _with_sales_cta(SHOP_OZON_ANSWER)
        else:
            answer = _with_sales_cta(SHOP_DISCOUNTS_ANSWER)
        answer = await _finalize_response_text(answer, user_lang)
        append_to_history(context, "assistant", answer)
        await update.message.reply_text(answer, reply_markup=SHOP_KEYBOARD, disable_web_page_preview=True)
    else:
        answer = STATIC_MENU_ANSWERS.get(intent)
        if not answer:
            answer = _with_sales_cta("Выберите, пожалуйста, нужный раздел в меню ниже ⬇️")
            answer = await _finalize_response_text(answer, user_lang)
            append_to_history(context, "assistant", answer)
            await update.message.reply_text(answer, reply_markup=GREETING_KEYBOARD)
        else:
            append_to_history(context, "assistant", answer)
            sub_menu = SUB_MENUS.get(intent)
            if sub_menu:
                clean_text, has_marker = extract_booking_marker(answer)
                clean_text = _cleanup_outgoing_text(clean_text)
                clean_text = _with_sales_cta(clean_text)
                clean_text, links = _extract_links(clean_text)
                clean_text = await _finalize_response_text(clean_text, user_lang)
                reply_markup = _build_reply_markup(has_marker, links, base_markup=sub_menu)
                await update.message.reply_text(
                    clean_text,
                    reply_markup=reply_markup,
                    disable_web_page_preview=True,
                )
            else:
                await send_answer(update.message, answer, user_lang=user_lang)

    topic = TOPIC_LABELS.get(intent, "нашими услугами")
    context.chat_data["last_topic"] = topic
    _schedule_followup(context, chat_id, user_id, topic)

# ================== КОМАНДЫ ПД ==================
async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает политику конфиденциальности."""
    if update.message is None:
        return
    await update.message.reply_text(PRIVACY_TEXT)


async def deletedata_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Удаляет все данные пользователя и отзывает согласие."""
    if update.message is None:
        return

    user_id = update.effective_user.id if update.effective_user else 0
    _cancel_user_timers(context.job_queue, update.effective_chat.id)
    result = delete_user_data(user_id)
    context.chat_data.clear()

    if result:
        details = "\n".join(f"  • {f}: {n} записей" for f, n in result.items())
        text = (
            "✅ Все ваши данные удалены:\n"
            f"{details}\n\n"
            "Согласие на обработку отозвано.\n"
            "Для повторного использования бота нажмите /start"
        )
    else:
        text = (
            "ℹ️ Данных для удаления не найдено.\n\n"
            "Если хотите начать сначала — /start"
        )

    await update.message.reply_text(text)


# ================== АДМИН-КОМАНДЫ ==================
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Генерирует и отправляет отчёт администратору."""
    user_id = update.effective_user.id if update.effective_user else 0
    if ADMIN_USER_ID and user_id != ADMIN_USER_ID:
        await update.message.reply_text("⛔ Эта команда доступна только администратору.")
        return

    await update.message.reply_text("📊 Генерирую отчёт, подождите...")

    try:
        text_summary = generate_text_summary()
        report_path = generate_monthly_report()

        await update.message.reply_text(text_summary)
        sent = False
        for attempt in range(2):
            try:
                with open(report_path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=os.path.basename(report_path),
                        caption="📎 Детальный отчёт с графиками",
                        read_timeout=60,
                        write_timeout=60,
                    )
                sent = True
                break
            except TimedOut:
                logger.warning("TimedOut при отправке отчёта (попытка %d/2)", attempt + 1)
                if attempt == 0:
                    await asyncio.sleep(2)
        if not sent:
            await update.message.reply_text(
                "⚠️ Отчёт сгенерирован, но отправка файла заняла слишком много времени.\n"
                "Повторите /report через минуту, пожалуйста."
            )
    except Exception as e:
        logger.exception("Ошибка генерации отчёта: %s", e)
        await update.message.reply_text(f"❌ Ошибка при генерации отчёта: {e}")


async def clearcache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Очищает кэш ответов на кнопки."""
    user_id = update.effective_user.id if update.effective_user else 0
    if ADMIN_USER_ID and user_id != ADMIN_USER_ID:
        await update.message.reply_text("⛔ Эта команда доступна только администратору.")
        return

    count = len(_button_cache)
    _button_cache.clear()
    await update.message.reply_text(f"🗑 Кэш очищен ({count} записей)")


# ================== АВТООТЧЁТ ==================
async def send_monthly_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Автоматически отправляет отчёт за прошлый месяц 1-го числа."""
    if not ADMIN_USER_ID:
        logger.warning("ADMIN_USER_ID не задан, автоотчёт пропущен.")
        return

    logger.info("Автоматическая отправка месячного отчёта...")

    try:
        now = dt.datetime.now()
        prev = now.replace(day=1) - dt.timedelta(days=1)

        text_summary = generate_text_summary(prev.year, prev.month)
        report_path = generate_monthly_report(prev.year, prev.month)

        await context.bot.send_message(
            chat_id=ADMIN_USER_ID,
            text=f"📊 Автоматический отчёт\n\n{text_summary}",
        )
        with open(report_path, "rb") as f:
            await context.bot.send_document(
                chat_id=ADMIN_USER_ID,
                document=f,
                filename=os.path.basename(report_path),
                caption="📎 Детальный отчёт с графиками",
            )
        logger.info("Автоотчёт за %s-%02d отправлен.", prev.year, prev.month)
    except Exception as e:
        logger.exception("Ошибка автоотчёта: %s", e)
        try:
            await context.bot.send_message(
                chat_id=ADMIN_USER_ID,
                text=f"❌ Ошибка при генерации автоотчёта: {e}",
            )
        except Exception:
            pass


# ================== ЗАПУСК ==================
async def post_init(application: Application) -> None:
    """Регистрирует команды в меню Telegram при старте бота."""
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 Главное меню"),
        BotCommand("privacy", "🔒 Политика конфиденциальности"),
        BotCommand("deletedata", "🗑 Удалить мои данные"),
    ])
    logger.info("Команды бота зарегистрированы в меню Telegram.")


def main() -> None:
    request = HTTPXRequest(
        connect_timeout=10.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=10.0,
    )
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("privacy", privacy_command))
    application.add_handler(CommandHandler("deletedata", deletedata_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("clearcache", clearcache_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if ADMIN_USER_ID and application.job_queue:
        msk = dt.timezone(dt.timedelta(hours=3))
        application.job_queue.run_monthly(
            send_monthly_report,
            when=dt.time(hour=9, minute=0, tzinfo=msk),
            day=1,
        )
        logger.info("Автоотчёт запланирован: 1-е число каждого месяца, 09:00 МСК")

    logger.info("Бот запущен. Ожидаю сообщения...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()