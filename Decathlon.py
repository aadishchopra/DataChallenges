import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from collections import Counter


#convert to csv's for faster processing
customer_l=pd.read_csv('customer_transactions_sample.csv')
customer_p=pd.read_csv('customer_p.csv')

#merge the two datasets

df=pd.concat([customer_l, customer_p], axis=0)

#for performance reasons
del(customer_p)
del(customer_l)

'''
Exploring the data
 
df.describe()
df.info()
df['Invoice'].isnull().sum()
1.Change date format 
2.To calculate metrics relating to purchase we need a new column called amount
3. Add columns for time period analysis
1.Few invoices were cancelled and hence they have been excluded from the analysis
3.We observed that few items were thrown away and hence the quantity is subtracted from the inventory
'''


#Data Quality Checks


dtype=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
dtype=pd.concat([dtype,pd.DataFrame(df.isnull().sum()).T])

dtype=pd.concat([dtype,pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T])
dtype.rename(index={1:'nulls',2:'%nulls'},inplace=True)
print(dtype)



#Data Pre-Processing
#1
df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, format='%m/%d/%Y %H:%M')

#2
df['Amount']=df['Quantity'] * df['Price']
#3

df['YearMonth'] = df['InvoiceDate'].map(lambda x: str(x.year) +'-'+ str(x.month))
#Monday=1 and Sunday=7
df['DayofWeek']=df.InvoiceDate.dt.dayofweek+1
df['Month']=df.InvoiceDate.dt.month
df['Hour']=df.InvoiceDate.dt.hour


#2
Cancelled_Invoice=[]
Unknown=[]
Invoice=[]
for x in df.Invoice:
    if isinstance(x, str):
        if x.startswith('C'):
            Cancelled_Invoice.append(x)
        else:
            Unknown.append(x)
    else:
        Invoice.append(x)
#2.
Regular_df=df.loc[~df['Invoice'].isin(Cancelled_Invoice)]

#3
Discarded_Items_df=Regular_df[Regular_df['Quantity']<0]

df_l=Regular_df[~Regular_df['Quantity']<0]




'''
1. How many are cancelled invoices ? What is the effect of cancellation on the system ?
2. Plot Cancellation by Region (Country). Plot relative percentages.
3. How many are my unique customers ( excluding cases where Customer ID's are missing )
4. Where does my Customer reside (Geographical Location) (( including cases where Customer ID's are missing )?
5. Top 10 Customers by Revenue ( Excluding Cancelled Invoices)
5.1 5. Top 10 Customers by Revenue ( Including Cancelled Invoices)
6.  Avg Amount Per Invoice (Basket)/Transaction (Includes cancellation)
7. Median Amount Per Invoice (Basket)/Transaction
8. How many unique items people buy per transactions (Includes Cancellations)
9. Quantity of items (Basket size) per Transaction (Excluding Cancellations and Discarded Items)
10. What Year performed well in terms of Revenue (Excludes Cancellation)? 
11. What month performed well in terms of Revenue ?
12. What Day of Week 
13. Which hour is the busiest hour of the store?
12. Profit/Loss by Month/Year


12. Stretch goal- Do NLP on Item Description and try to do clustering on similar items ?
13 Stretch goal- Market Basket Analysis
'''


Cancelled_df=df.loc[df['Invoice'].isin(Cancelled_Invoice)]
can_bplt=Cancelled_df['Country'].value_counts().sort_values(ascending=0)[:10].plot(kind='bar',figsize=(14,8),title="Cancelled Invoices by Region")
can_bplt.bar_label(can_bplt.containers[0], label_type='edge')
#3
print("No. of transactions where customer ID's are missing "+ str(df['Customer ID'].isnull().sum()))

#4
Unique_Customers = df['Customer ID'].dropna().unique().tolist()
Unique_Customers_Countries=df.groupby(['Country','Customer ID']).size().to_frame(name='count').reset_index()
uni_cust_plt=Unique_Customers_Countries['Country'].value_counts().sort_values(ascending=0)[:10].plot(kind='bar',figsize=(14,8),title="Unique Customers by Region")
uni_cust_plt.bar_label(uni_cust_plt.containers[0], label_type='edge')

#5

print("Top 10 Customers by Revenue are shown below : ")
print(Regular_df.groupby(['Customer ID','Country']).agg({'Amount':sum}).sort_values(by='Amount',ascending=0).head(10))

#5.1

print(df.groupby(['Customer ID','Country']).agg({'Amount':sum}).sort_values(by='Amount',ascending=0).head(10))


#6
print("The average basket size value in pound sterling per transaction : ",np.mean(df.groupby('Invoice').agg({'Amount':sum})['Amount']))

#7
print("The median basket size value in pound sterling per transaction : ",np.median(df.groupby('Invoice').agg({'Amount':sum})['Amount']))

#8
print("Average number of unique items per transaction/basket",np.mean(df.groupby('Invoice').size().to_frame(name='Number of Items per Trx').reset_index('Invoice').sort_values(by='Number of Items per Trx',ascending=0)['Number of Items per Trx']))
print("Median number of unique items per transaction/basket",np.median(df.groupby('Invoice').size().to_frame(name='Number of Items per Trx').reset_index('Invoice').sort_values(by='Number of Items per Trx',ascending=0)['Number of Items per Trx']))

#9
print("Average quantities(load size per basket) per Transactions : ",np.mean(df_l.groupby('Invoice').agg({'Quantity':sum}).reset_index('Invoice')['Quantity']))
print("Median quantities(load size per basket) per Transactions : ",np.median(df_l.groupby('Invoice').agg({'Quantity':sum}).reset_index('Invoice')['Quantity']))

#10
YearMonthplt = Regular_df.groupby('Invoice')['YearMonth'].unique().value_counts().sort_index().plot(kind='bar',figsize=(14,8))
YearMonthplt.set_xlabel('Month',fontsize=15)
YearMonthplt.set_ylabel('Number of Orders',fontsize=15)
YearMonthplt.set_title('Number of orders for different Months (Dec 2009 - Dec 2011)',fontsize=15)
plt.show()

#11

DayMonthplt = Regular_df.groupby('Invoice')['DayofWeek'].unique().value_counts().sort_index().plot(kind='bar',figsize=(14,8))
DayMonthplt.set_xlabel('Day',fontsize=15)
DayMonthplt.set_ylabel('Number of Orders',fontsize=15)
DayMonthplt.set_title('Number of orders for different Days',fontsize=15)
DayMonthplt.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sat','Sun'), rotation='horizontal', fontsize=15)
plt.show()


#12
TrendMonth=Regular_df.groupby('YearMonth').agg({'Amount':sum}).reset_index('YearMonth')
TrendMonthplt=TrendMonth.plot('YearMonth','Amount')
TrendMonthplt.set_xticks(TrendMonth.index,TrendMonth['YearMonth'],rotation=90)

#13

TrendDay=Regular_df.groupby('DayofWeek').agg({'Amount':sum}).reset_index('DayofWeek')
TrendDayplt=TrendDay.plot('DayofWeek','Amount')
TrendDayplt.set_xticks(TrendDay.index+1,TrendDay['DayofWeek'])
TrendDayplt.set_xticklabels(['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])

#14
Regular_df=Regular_df[Regular_df.Invoice != '549245']






Hourplt = Regular_df.groupby('Invoice')['Hour'].unique().value_counts().sort_index().plot(kind='bar',figsize=(14,8))
Hourplt.set_xlabel('Hour',fontsize=15)
Hourplt.set_ylabel('Number of Orders',fontsize=15)
Hourplt.set_title('Busiest Hour of the Day',fontsize=15)
#Hourplt.set_xticklabels(Hourplt['Hour'], rotation='horizontal', fontsize=15)
plt.show()


#
Discarded_Items_df=Regular_df[Regular_df['Quantity']<0]
Discarded_Items_df=Discarded_Items_df.groupby(['Country']).agg({'Quantity':sum})

Discarded_Items_df=Regular_df[Regular_df['Quantity']<0]
Discarded_Items_df=Discarded_Items_df.groupby(['Country','StockCode']).agg({'Quantity':sum}).reset_index().sort_values(by='Quantity')[:10]
Discarded_Items_df['Quantity']=abs(Discarded_Items_df['Quantity'])



'''

1. We observe that Cancelled invoices are 19494. Also there were 6 invoices in total which started
with letter A
2. We observe that UK has maximum number of cancelled invoices
3. We see that there are 5942 unique customers
4. We see that most of our customer approx 5400 live in UK, followed by Germany having a meagre 107
5. We observe that the mDayMonthpltimum spending of the top customer is 57000
6. The average basket size value in pound sterling per transaction :  92.53788371746104
7. The median basket size value in pound sterling per transaction :  30.299999999999997
8. Average items in basket is 19.90324084433505 and Median number of items per transaction/basket 9.0
9.  
11. Maybe Saturday the store remains closed
12. For Invoice No.549245 we see that there are two Invoice Dates. Usually Invoice Date would be 1
'''




