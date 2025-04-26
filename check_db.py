import sqlite3
import os

print("Current directory:", os.getcwd())
print("Checking if database exists:", os.path.exists('journal_recommender.db'))

conn = sqlite3.connect('journal_recommender.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nDatabase tables:")
for table in tables:
    print(f"- {table[0]}")

# Check users table
print("\nUsers:")
cursor.execute("SELECT id, username, email FROM users")
users = cursor.fetchall()
for user in users:
    print(f"ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")

# Check ratings table
print("\nRatings:")
cursor.execute("""
    SELECT r.id, u.username, r.journal_name, r.rating, r.created_at 
    FROM ratings r
    JOIN users u ON r.user_id = u.id
""")
ratings = cursor.fetchall()
for rating in ratings:
    print(f"ID: {rating[0]}, User: {rating[1]}, Journal: {rating[2]}, Rating: {rating[3]}, Date: {rating[4]}")

conn.close()