#Görev 1: Veriyi Hazırlama

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns",None)
pd.set_option("display.float_format",lambda x: "%.5f" % x)

df_=pd.read_csv("flo_data_20k.csv")
df = df_.copy()

df.isnull().sum()



def outlier_tresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range=quartile3-quartile1
    up_limit = quartile3+1.5*interquantile_range
    low_limit = quartile1-1.5*interquantile_range
    return low_limit, up_limit

def replace_with_tresholds(dataframe,variable):
    low_limit,up_limit = outlier_tresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit.__round__()
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit.__round__()



replace_with_tresholds(df,"order_num_total_ever_online")
replace_with_tresholds(df,"order_num_total_ever_offline")
replace_with_tresholds(df,"customer_value_total_ever_offline")
replace_with_tresholds(df,"customer_value_total_ever_online")



df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]



import datetime as dt
for i in df.columns:
    if "date" in i:
       df[i] = df[i].astype("datetime64[ns]")


#Görev 2:CLTV Veri Yapısının Oluşturulması

df["last_order_date"].max()
today_date=dt.datetime(2021,6,1)

# recency = her kullanıcı özelinde min ve max satın alma tarihi arasındaki fark (haftalık)
# frequency = tekrar eden toplam satın alma sayısı
# T: müşterinin yaşı.haftalık (analiz tarihinden ne kadar önce ilk satın alma yapılmıs)
# monetary value: satın alma başına ortalama kazanç



cltv_df=df.groupby("master_id").agg({
                                  "first_order_date": lambda x: (today_date-x.min()).days,
                                  "order_num_total":lambda x: x.sum(),
                                  "customer_value_total":lambda x: x.sum()})

cltv_df["recency_cltv_weekly"] = df["last_order_date"]-df["first_order_date"]

cltv_df.head()




cltv_df.columns = ["recency","T","frequency","monetaryTOPLAM"]

cltv_df.describe().T

cltv_df["monetary"] = cltv_df["monetaryTOPLAM"]/cltv_df["frequency"]

cltv_df["recency"]=cltv_df["recency"]/7

cltv_df["T"]=cltv_df["T"]/7


#Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:20]

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

