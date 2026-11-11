import mysql.connector
import pandas as pd


class Connector:
    def __init__(self,
                 server="localhost",
                 port=3306,
                 database="salesdatabase",
                 username="root",
                 password="MySQL@123456"):
        self.server=server
        self.port=port
        self.database=database
        self.username=username
        self.password=password
    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.server,
            port=self.port,
            database=self.database,
            user=self.username,
            password=self.password,
            use_pure=True)
    def queryDataset(self,sql):
        try:
            cursor=self.conn.cursor()
            cursor.execute(sql)
            df=pd.DataFrame(cursor.fetchall())
            df.columns=cursor.column_names
            return df
        except Exception as e:
            print(str(e))
            return None