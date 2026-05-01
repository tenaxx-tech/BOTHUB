import sqlite3
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_PATH = "bot_data.db"

def init_db():
    """Создаёт все необходимые таблицы"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Таблица пользователей: баланс и текущая модель
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            balance INTEGER DEFAULT 0,
            current_model TEXT,
            last_activity TIMESTAMP
        )
    ''')

    # Таблица истории диалогов
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Таблица учёта бесплатных изображений
    c.execute('''
        CREATE TABLE IF NOT EXISTS weekly_image_usage (
            user_id INTEGER,
            week_start TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, week_start)
        )
    ''')

    # Таблица для OAuth-токенов DonationAlerts
    c.execute('''
        CREATE TABLE IF NOT EXISTS donation_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            access_token TEXT,
            refresh_token TEXT,
            expires_at INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ---------- Новая таблица для заказов Robokassa ----------
    c.execute('''
        CREATE TABLE IF NOT EXISTS robokassa_orders (
            inv_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount INTEGER,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("База данных инициализирована")

# ----- Работа с балансом -----
def get_user_balance(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def add_balance(user_id, amount):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
    if c.rowcount == 0:
        c.execute("INSERT INTO users (user_id, balance) VALUES (?, ?)", (user_id, amount))
    conn.commit()
    conn.close()

def deduct_balance(user_id, amount):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if row and row[0] >= amount:
        c.execute("UPDATE users SET balance = balance - ? WHERE user_id = ?", (amount, user_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def update_user_activity(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO users (user_id, last_activity)
        VALUES (?, CURRENT_TIMESTAMP)
    ''', (user_id,))
    conn.commit()
    conn.close()

# ----- История диалогов -----
def save_message(user_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (user_id, role, content)
        VALUES (?, ?, ?)
    ''', (user_id, role, content))
    conn.commit()
    conn.close()

def get_history(user_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT role, content FROM chat_history
        WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def clear_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# ----- Лимит изображений -----
def get_week_start():
    today = datetime.now().date()
    start = today - timedelta(days=today.weekday())
    return start.isoformat()

def get_weekly_image_count(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    week_start = get_week_start()
    c.execute("SELECT count FROM weekly_image_usage WHERE user_id = ? AND week_start = ?",
              (user_id, week_start))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_weekly_image_count(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    week_start = get_week_start()
    c.execute('''
        INSERT INTO weekly_image_usage (user_id, week_start, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, week_start) DO UPDATE SET count = count + 1
    ''', (user_id, week_start))
    conn.commit()
    conn.close()

# ----- DonationAlerts токены -----
def save_donation_token(access_token, refresh_token, expires_in):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM donation_tokens")
    expires_at = int(time.time()) + expires_in
    c.execute("INSERT INTO donation_tokens (access_token, refresh_token, expires_at) VALUES (?, ?, ?)",
              (access_token, refresh_token, expires_at))
    conn.commit()
    conn.close()

def get_donation_token():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT access_token, refresh_token, expires_at FROM donation_tokens ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row and row[2] > int(time.time()):
        return row[0]
    return None

def update_donation_token(access_token, refresh_token, expires_in):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    expires_at = int(time.time()) + expires_in
    c.execute("UPDATE donation_tokens SET access_token=?, refresh_token=?, expires_at=? WHERE id=1",
              (access_token, refresh_token, expires_at))
    conn.commit()
    conn.close()

# ---------- Новые функции для Robokassa ----------
def create_robokassa_order(inv_id: int, user_id: int, amount: int):
    """Создаёт запись о заказе в Robokassa"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO robokassa_orders (inv_id, user_id, amount, status)
        VALUES (?, ?, ?, 'pending')
    ''', (inv_id, user_id, amount))
    conn.commit()
    conn.close()

def update_robokassa_order_status(inv_id: int, status: str):
    """Обновляет статус заказа (success, fail)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE robokassa_orders SET status = ? WHERE inv_id = ?', (status, inv_id))
    conn.commit()
    conn.close()

def get_robokassa_order(inv_id: int):
    """Возвращает информацию о заказе"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, amount, status FROM robokassa_orders WHERE inv_id = ?', (inv_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"user_id": row[0], "amount": row[1], "status": row[2]}
    return None
