from connector.connector import Connector
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
    def getCustomersForClustering():
        sql="""
        SELECT DISTINCT
        customer.CustomerId,
        Age,
        Annual_Income,
        Spending_Score
        FROM customer, customer_spend_score
        WHERE customer.CustomerId = customer_spend_score.CustomerID;
        """
        conn = Connector()
        conn.connect()
        df = conn.queryDataset(sql)
        return df