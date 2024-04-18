import sqlite3
from typing import Any

import numpy as np

from predwordimp.eval.util import get_rank_limit


def get_user_rankings(user_id, db) -> dict[Any, Any]:
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT r.word_order, a.task_id
        FROM rankings_assignment AS a
        INNER JOIN rankings_ranking AS r ON a.id = r.assignment_id
        WHERE a.user_id = ?
    """,
        (user_id,),
    )

    # Fetch all the rows
    rankings = cursor.fetchall()

    user_rankings = {task_id: word_order for word_order, task_id in rankings}

    cursor.close()
    conn.close()

    return user_rankings


def get_len(task_id, db) -> int:
    # Connect to the SQLite database
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT content
        FROM rankings_task
        WHERE id = ?
    """,
        (task_id,),
    )

    # Fetch the task
    task = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return len(task[0].split())


def annot2rank(ranks, txt_len) -> Any:
    label = np.ones(txt_len) * (get_rank_limit(0.1, txt_len) + 1)

    for i, e in enumerate(ranks):
        pos = e["position"]
        label[pos] = i + 1

    return label
