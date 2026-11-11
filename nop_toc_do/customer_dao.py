from connector.connector import Connector
from cosine_similarity import CosineSimilarity


class CustomerDao:
    @staticmethod
    def getAllCustomers():
        conn = Connector()
        conn.connect()
        sql="select * from customer"
        df = conn.queryDataset(sql)
        return df
    @staticmethod
    def getAllCustomersByAge(min,max):
        conn = Connector()
        conn.connect()
        sql="select * from customer where age>=%s and age<=%s"
        val=(min,max)
        df = conn.queryDataset(sql, val)
        return df

    @staticmethod
    def searchCustomersByCosine(keywords, n):
        """
        Tìm N customer có Name giống keywords nhất (Cosine Similarity).
        Trả về DataFrame: Id, CustomerID, Name, Cosine Similarity (giảm dần).
        """
        conn = Connector()
        conn.connect()
        sql = "select Id, CustomerID, Name from customer"
        df = conn.queryDataset(sql)
        if df is None or df.empty:
            return df

        queryVec = CosineSimilarity.vectorize(keywords)
        df = df.copy()
        df["Cosine Similarity"] = df["Name"].apply(
            lambda name: round(
                CosineSimilarity.cosine(queryVec, CosineSimilarity.vectorize(name)), 4))
        return (df.sort_values(by="Cosine Similarity", ascending=False)
                  .head(int(n))
                  .reset_index(drop=True))
