from connector.connector import Connector


# =========================
# 1. Kết nối salesdatabase
# =========================
conn = Connector()

conn.connect()

sql = "SELECT * FROM customer"
df = conn.queryDataset(sql)

print("===== DATABASE: salesdatabase =====")
print(df)


# =========================
# 2. Chuyển sang database sakila
# =========================
conn.database = "sakila"

conn.connect()

sql = "SELECT * FROM actor"
df = conn.queryDataset(sql)

print("\n===== DATABASE: sakila =====")
print(df)