from connector.connector import Connector
from model.user import User

class UserDAO:
    @staticmethod
    def login(uid,pwd):
        sql="select * from users where username=%s and password=%s"
        val=(uid, pwd)
        conn = Connector()
        conn.connect()
        df=conn.queryDataset(sql,val)
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        us = User()
        us.id = row["id"]
        us.username = row["username"]
        us.password = row["password"]
        us.is_banned = row["is_banned"]
        us.role = row["role"]
        return us