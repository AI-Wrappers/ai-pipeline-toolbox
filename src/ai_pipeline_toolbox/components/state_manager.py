import sqlite3
from ai_pipeline_toolbox.core.interfaces import BaseStateManager

class SQLiteStateManager(BaseStateManager):
    """
    Persists execution state using SQLite.
    Tracks statuses: pending, completed, failed.
    """
    def __init__(self, db_path: str = "state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT
                )
            ''')
            conn.commit()

    def is_completed(self, task_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT status FROM tasks WHERE task_id = ?', (task_id,)
            )
            row = cursor.fetchone()
            if row and row[0] == 'completed':
                return True
        return False

    def mark_completed(self, task_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO tasks (task_id, status) VALUES (?, 'completed')
                   ON CONFLICT(task_id) DO UPDATE SET status='completed', error=NULL''',
                (task_id,)
            )
            conn.commit()

    def mark_failed(self, task_id: str, error: Exception) -> None:
        error_msg = str(error)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO tasks (task_id, status, error) VALUES (?, 'failed', ?)
                   ON CONFLICT(task_id) DO UPDATE SET status='failed', error=?''',
                (task_id, error_msg, error_msg)
            )
            conn.commit()
