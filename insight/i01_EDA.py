import pandas as pd

# %% lectura de datos -------------------------------------------
df_cnt = pd.read_csv('datasets/contract.csv')
df_prs = pd.read_csv('datasets/personal.csv')
df_int = pd.read_csv('datasets/internet.csv')
df_phn = pd.read_csv('datasets/phone.csv')

# %% lectura de informcion basica--------------------------------

df_cnt.info()
df_cnt

df_prs.info()
df_prs

df_int.info()
df_int

df_phn.info()
df_phn

# %% combinacion de DF e impresion de información del mismo------

df_full = df_cnt.merge(df_prs, on='customerID', how='left')
df_full = df_full.merge(df_int, on='customerID', how='left')
df_full = df_full.merge(df_phn, on='customerID', how='left')

print(df_full.shape)
df_full.head()

# %% creacion de variable objetivo y verificacion de distribuciones-
df_full['churn'] = df_full['EndDate'].apply(lambda x: 0 if x == 'No' else 1)

df_full['churn'].value_counts()
