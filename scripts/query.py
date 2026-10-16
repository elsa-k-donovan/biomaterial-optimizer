"""Run a SQL query against the database and print the result.

Usage:
    uv run scripts/query.py "SELECT * FROM recipes LIMIT 5"
    uv run scripts/query.py < queries/seaweed_biomaterials.sql
"""

import sys

import duckdb

con = duckdb.connect("data/biomaterials.duckdb", read_only=True)
sql = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
print(con.sql(sql))
