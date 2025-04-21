# filename: api_subway.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd, joblib

app   = FastAPI()
model = joblib.load('subway_xgb_best.pkl')

# ── (1) 학습 때와 동일한 전처리 ─────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date']  = pd.to_datetime(df['date'])
    df['hour']  = df['time'].str.slice(0, 2).astype('int8')
    df['month'] = df['date'].dt.month.astype('int8')
    df['day']   = df['date'].dt.day.astype('int8')
    df = df.drop(columns=['date', 'time', 'station_name'])  # 학습 때 제외한 열
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('missing')
    return df

# ── (2) 요청 스키마 (Pydantic) ─────────────────────────────────────
class PredictItem(BaseModel):
    date          : str   = Field(...,  example="2024-09-01")
    line          : int   = Field(...,  example=2)
    station_code  : int   = Field(...,  example=152)
    station_name  : str   = Field(...,  example="신도림")
    time          : str   = Field(...,  example="18:00")
    일기           : str   = Field(...,  example="맑음")
    시정           : float | None = 15
    운량           : float | None = 1
    중하운량        : float | None = 0
    현재기온         : float | None = 25
    이슬점온도        : float | None = 18
    체감온도         : float | None = 27
    일강수          : float | None = 0
    적설           : float | None = 0
    습도           : float | None = 60
    풍향           : str   = "남"
    풍속           : float | None = 2
    해면기압         : float | None = 1012
    weekday        : int   = Field(...,  example=0)   # 월=0, 화=1 …
    is_holiday     : int   = Field(...,  example=0)   # 휴일=1

@app.post("/predict")
def predict(item: PredictItem):
    df = pd.DataFrame([item.dict()])
    X  = preprocess(df)
    pred = int(model.predict(X)[0].round())
    return {"count_pred": pred}