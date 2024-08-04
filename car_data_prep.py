
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

def clean_model_column(model, manufactor):
    if not isinstance(model, str):
        return None
    import re
    model = re.sub(r'[^\w\s]', '', model)
    model = re.sub(r'\s+', ' ', model).strip()
    manufactor = re.sub(r'[^\w\s]', '', manufactor)
    model = re.sub(fr'\b{manufactor}\b', '', model, flags=re.IGNORECASE).strip()
    model = re.sub(r'\b\d{4}\b', '', model).strip()
    if model == '':
        model = None
    return model

 

def prepare_data(df):
    # יצירת עותק של הנתונים המקוריים
    df = df.copy()
    
    # הסרת כפילויות מלאות
    df = df.drop_duplicates()
    
    # הסרת עמודות מסוימות
    columns_to_drop = ['Test', 'Area','Supply_score']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # המרת ערכים נומריים
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    
    # מיזוג קטגוריות זהות
    df['Engine_type'] = df['Engine_type'].replace({'היבריד': 'היברידי', 'גז': 'אחר', 'טורבו דיזל': 'אחר', 'חשמלי': 'אחר'})

    # הסרת פסיקים מעמודות (במידת הצורך)
    df['Km'] = df['Km'].astype(str).str.replace(',', '')
    df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '')
    
    # המרה חזרה לנומרי
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    
    # טיפול בערכים חסרים
    df['capacity_Engine'] = df.groupby('manufactor')['capacity_Engine'].transform(lambda x: x.fillna(x.median()))
    df['Engine_type'] = df.groupby(['model', 'Year'])['Engine_type'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'אחר'))
    df['Km'] = df.groupby('Year')['Km'].transform(lambda x: x.fillna(x.mean()))
    df = df.dropna(subset=['capacity_Engine'])
    
    # הבטחת סוג המידע של capacity_Engine
    df['capacity_Engine'] = df['capacity_Engine'].astype(int)
    
    # טיפול בערכים חסרים בעמודות קטגוריאליות
    mode_color = df['Color'].mode().iloc[0] if not df['Color'].mode().empty else 'unknown'
    df['Color'] = df['Color'].fillna(mode_color)
    df['Gear'] = df['Gear'].fillna(df['Gear'].mode().iloc[0] if not df['Gear'].mode().empty else 'unknown')
    df['Prev_ownership'] = df['Prev_ownership'].fillna(df['Curr_ownership'])
    df['Curr_ownership'] = df['Curr_ownership'].fillna(df['Prev_ownership'])
    df['Prev_ownership'] = df['Prev_ownership'].fillna('פרטית')
    df['Curr_ownership'] = df['Curr_ownership'].fillna('פרטית')
    median_pic_num = df['Pic_num'].median() if not df['Pic_num'].isna().all() else 0
    df['Pic_num'] = df['Pic_num'].fillna(median_pic_num)
    
    # ניקוי עמודת דגם
    df['model'] = df.apply(lambda row: clean_model_column(row['model'], row['manufactor']), axis=1)
    df['model'] = df['model'].replace('none', np.nan)
    df = df.dropna(subset=['model'])


    # המרת עמודות קטגוריאליות לקטגוריות
    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'City', 'Color']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # הסרת חריגים
    df = df[df['Year'] > 2000]
    df = df[df['Hand'] <= 6]
    df = df[df['Pic_num'] < 10]
    df = df[(df['Km'] > 0) & (df['Km'] < 273000)]
    df = df[(df['capacity_Engine'] > 150) & (df['capacity_Engine'] < 8000)]
    
    return df
