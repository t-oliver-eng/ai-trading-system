# auth.py - User Authentication and Management

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime

DATABASE_PATH = "users.db"


class User(UserMixin):
    def __init__(self, id, username, is_admin=False, last_login=None):
        self.id = id
        self.username = username
        self.is_admin = is_admin
        self.last_login = last_login


def init_db():
    """Initialize the database with users table"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()


def create_user(username, password, is_admin=False):
    """Create a new user"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, password_hash, is_admin),
        )

        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists


def get_user_by_username(username):
    """Retrieve a user by username"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, is_admin, last_login FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return User(
            id=row[0], username=row[1], is_admin=bool(row[2]), last_login=row[3]
        )
    return None


def get_user_by_id(user_id):
    """Retrieve a user by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, is_admin, last_login FROM users WHERE id = ?", (user_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return User(
            id=row[0], username=row[1], is_admin=bool(row[2]), last_login=row[3]
        )
    return None


def verify_password(username, password):
    """Verify a user's password"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return check_password_hash(row[0], password)
    return False


def update_last_login(username):
    """Update the last login timestamp for a user"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE users SET last_login = ? WHERE username = ?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username),
    )

    conn.commit()
    conn.close()


def get_all_users():
    """Get all users (for admin panel)"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, is_admin, created_at, last_login FROM users ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    users = []
    for row in rows:
        users.append(
            {
                "id": row[0],
                "username": row[1],
                "is_admin": bool(row[2]),
                "created_at": row[3],
                "last_login": row[4],
            }
        )
    return users


def delete_user(user_id):
    """Delete a user by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Prevent deleting admin users
    cursor.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()

    if row and row[0]:
        conn.close()
        return False  # Cannot delete admin

    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True


def change_password(username, new_password):
    """Change a user's password"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    password_hash = generate_password_hash(new_password)
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (password_hash, username),
    )

    conn.commit()
    conn.close()


def ensure_admin_exists(admin_username, admin_password):
    """Ensure the admin user exists in the database"""
    init_db()

    # Check if admin exists
    user = get_user_by_username(admin_username)
    if not user:
        # Create admin user
        create_user(admin_username, admin_password, is_admin=True)
