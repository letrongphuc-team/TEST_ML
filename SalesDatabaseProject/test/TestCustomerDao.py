from SalesDatabaseProject.dal.customer_dao import CustomerDao:

df=CustomerDao().getCustomer()
print(df)

df2=CustomerDao().getCustomerById(min: 35, max: 40)
print("#"*20)
print(df2)