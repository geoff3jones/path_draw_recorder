"""
Trying to diagnose as weird crash I was getting when reading from parquet 
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def test_pandas_write_read():
    df_out = pd.DataFrame.from_dict([{"A":i} for i in range(3)])
    df_out.to_parquet("crash.parquet")
    df_in  = pd.read_parquet("crash.parquet")
    print(df_in)

def test_arrow_write_read():
    df = pd.DataFrame.from_dict([{"A":i} for i in range(3)])
    table_out = pa.Table.from_pandas(df)
    pq.write_table(table_out, 'crash.parquet')
    table_in = pq.read_table('crash.parquet')
    print(table_in)

if __name__ == "__main__":
    test_pandas_write_read()
    test_arrow_write_read()