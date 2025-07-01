import io,os
import pandas as pd

def df_into_string(df,max_rows=5): # for text anaylsis
    buf=io.StringIO()
    df.info(buf)
    schema=buf.getvalue()
    head=df.head(max_rows).to_markdown(index=False)

    missing=df.isnull().sum()
    missing=missing[missing>0]
    missing_info='No missing values.'if missing.empty else str(missing)
    return f"""
    Schema : \n {schema} \n Preview : \n{head}\n Missing : \n{missing_info}
    """