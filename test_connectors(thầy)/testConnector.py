from connectors.connector import Connector

conn=Connector()
conn.connect()
sql="select * from customer"
df=conn.queryDataset(sql)
print(df)
print("*"*30)
sql="select * from customer_spend_score"
df=conn.queryDataset(sql)
print(df)
print("*"*30)
sql="select * from customer where Age>=19 and Age<=23"
df=conn.queryDataset(sql)
print(df)

conn.database="sakila"
conn.connect()
sql="select * from actor"
df=conn.queryDataset(sql)
print(df)