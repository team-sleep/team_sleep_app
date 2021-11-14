# import hydralit automatically imports all of Streamlit
# https://github.com/TangleSpace/hydralit
import hydralit as hy
import psycopg2
import pandas as pd


conn = psycopg2.connect(**hy.secrets["postgres"])
sql = "SELECT * from test_table;"
df = pd.read_sql(sql, conn, index_col=None)
conn.close()

hy.info('Hello world')
hy.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
