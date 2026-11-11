import pandas as pd
from customer_dao import CustomerDao

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", None)

keywords = "Nguyen Van An"
n = 5
df = CustomerDao.searchCustomersByCosine(keywords, n)
print("Keywords:", keywords, "| N =", n)
print(df)