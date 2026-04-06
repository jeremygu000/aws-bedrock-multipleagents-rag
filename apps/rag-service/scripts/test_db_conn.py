#!/usr/bin/env python3
import os
import sys

import psycopg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Settings  # noqa: E402
from app.secrets import resolve_db_password  # noqa: E402

s = Settings()
password = resolve_db_password(s)
print(f"Password resolved: {'yes' if password else 'no'} (len={len(password) if password else 0})")

conn = psycopg.connect(
    host=s.db_host, port=s.db_port, dbname=s.db_name,
    user=s.db_user, password=password, connect_timeout=10
)
cur = conn.cursor()
cur.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name
""")
for schema, table in cur.fetchall():
    cur2 = conn.cursor()
    try:
        cur2.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
        cnt = cur2.fetchone()[0]
        print(f"{schema}.{table}: {cnt} rows")
    except Exception as e:
        conn.rollback()
        print(f"{schema}.{table}: ERROR {e}")
conn.close()
