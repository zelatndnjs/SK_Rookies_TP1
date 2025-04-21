import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
import joblib
from scipy import stats

import datetime as dt
import re
import time
import requests
from bs4 import BeautifulSoup

from folium.plugins import MarkerCluster

# 페이지 설정
st.set_page_config(
    page_title="서울 지하철 승하차 인원 분석 및 예측 시스템",
    page_icon="🚇",
    layout="wide"
)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('seoul_ridership_weather_calendar_2024.csv')
        # 데이터 타입 변환
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['time'].str[:2].astype(int)
        
        # 요일 매핑
        weekday_map = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
        df['weekday_name'] = df['date'].dt.weekday.map(weekday_map)
        
        # 월 추출
        df['month'] = df['date'].dt.month
        
        # 요일 약어 (혼잡도 계산에 필요)
        weekday_short_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
        df['요일'] = df['date'].dt.weekday.map(weekday_short_map)
        # 역코드 문자열로 통일
        df['station_code'] = df['station_code'].astype(str)

        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

# 역 좌표 데이터 로드 함수
@st.cache_data
def load_station_coordinates():
    try:
        # 인코딩 문제를 해결하기 위해 'cp949' 인코딩으로 파일 읽기
        coords_df = pd.read_csv('서울교통공사_1_8호선 역사 좌표(위경도) 정보_20241031.csv', encoding='cp949')
        
        # 컬럼명 직접 지정 (파일에서 첫 행이 헤더)
        coords_df.columns = ['번호', '호선', '역코드', '역명', '위도', '경도', '작성일자']
        
        # 필요한 컬럼만 선택하고 컬럼명 변경
        coords_df = coords_df.rename(columns={
            '역명': 'station_name',
            '호선': 'line',
            '역코드': 'station_code',
            '위도': 'latitude',
            '경도': 'longitude'
        })
        
        # 필요한 컬럼만 선택 (역명, 역코드, 위도, 경도만 유지)
        coords_df = coords_df[['station_name', 'line', 'station_code', 'latitude', 'longitude']]
        
        # 결측치 처리
        coords_df = coords_df.dropna(subset=['station_code', 'latitude', 'longitude'])  # 역코드, 위도, 경도 결측치 제거
        coords_df['line'] = coords_df['line'].astype(int)  # 호선 정수형으로 변환
        coords_df['station_code'] = coords_df['station_code'].astype(str)  # 역코드 문자열로 통일
        
        # 중복 제거 (같은 역명이지만 위치가 다른 경우 첫 번째 값 사용)
        coords_df = coords_df.drop_duplicates(subset=['station_code'], keep='first')
        
        return coords_df
    except Exception as e:
        st.error(f"역 좌표 데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

# 혼잡도 계산 함수 - 승하차 인원 절대값 기준
def calculate_congestion(passengers, station_name, data_df, all_data=None):
    """
    승하차 인원 절대값 기준으로 혼잡도 계산 (1-10 척도)
    
    Parameters:
    ----------
    passengers : int or float
        현재 역의 승하차 인원
    station_name : str
        역 이름
    data_df : pandas.DataFrame
        필터링된 데이터프레임
    all_data : pandas.DataFrame, optional
        필터링되지 않은 전체 데이터프레임 (없을 경우 data_df 사용)
    """
    # 최대 혼잡도 기준값 (16,000명)
    max_congestion = 16000
    
    # 구간별 혼잡도 계산 (1-10 척도)
    if passengers >= max_congestion:
        congestion = 10
    else:
        # 승하차 인원을 1-10 척도로 변환
        congestion = int(1 + (passengers / max_congestion) * 9)
    
    # 백분율 계산 (최대 혼잡도 대비)
    percentile = (passengers / max_congestion) * 100
    
    return congestion, percentile  # 혼잡도, 절대적 백분율

# 예측 모델 함수
def predict_model(features, data_df):
    """
    간단한 예측 함수 - 실제로는 머신러닝 모델 사용 권장
    """
    # 해당 역의 기존 데이터를 바탕으로 예측
    station_data = data_df[data_df['station_name'] == features['station_name']]
    
    # 필터링된 데이터가 없는 경우
    if station_data.empty:
        return 0, 1
    
    # 요일별 평균 계산
    day_data = station_data[station_data['요일'] == features['요일']]
    
    if day_data.empty:
        day_avg = station_data['count'].mean()
    else:
        day_avg = day_data['count'].mean()
    
    # 공휴일 여부에 따른 조정
    if features['is_holiday'] == 1:
        holiday_data = station_data[station_data['is_holiday'] == 1]
        if not holiday_data.empty and station_data[station_data['is_holiday'] == 0].shape[0] > 0:
            holiday_factor = holiday_data['count'].mean() / station_data[station_data['is_holiday'] == 0]['count'].mean()
            day_avg *= holiday_factor
        else:
            # 공휴일 데이터가 없는 경우 일반적인 감소율 적용
            day_avg *= 0.7
    
    # 날씨 영향 (기온, 강수량)
    weather_factor = 1.0
    
    if features['temperature'] > 25:  # 더운 날
        weather_factor *= 0.95
    elif features['temperature'] < 5:  # 추운 날
        weather_factor *= 0.9
    
    if features['rainfall'] > 10:  # 비/눈이 많이 오는 날
        weather_factor *= 0.85
    
    # 최종 예측값
    prediction = int(day_avg * weather_factor)
    
    # 혼잡도 계산
    congestion, _ = calculate_congestion(prediction, features['station_name'], data_df)
    
    return prediction, congestion

# 색상 매핑 함수 - 호선 색상
def get_line_color(line):
    color_map = {
        1: '#0052A4',  # 1호선: 파랑
        2: '#00A84D',  # 2호선: 녹색
        3: '#EF7C1C',  # 3호선: 오렌지
        4: '#00A5DE',  # 4호선: 하늘색
        5: '#996CAC',  # 5호선: 보라
        6: '#CD7C2F',  # 6호선: 황토색
        7: '#747F00',  # 7호선: 갈색
        8: '#E6186C',  # 8호선: 분홍
    }
    return color_map.get(line, '#808080')  # 기본값 회색

# 혼잡도 색상 매핑 함수
def get_congestion_color(congestion):
    if congestion <= 3:
        return "green"
    elif congestion <= 7:
        return "orange"
    else:
        return "red"

# 혼잡도 텍스트 매핑 함수
def get_congestion_text(congestion):
    if congestion <= 3:
        return "여유"
    elif congestion <= 7:
        return "보통"
    else:
        return "매우 혼잡"

# 날씨 상태 간소화 함수
def simplify_weather(weather):
    """
    세부적인 날씨 상태를 더 간단한 카테고리로 그룹화
    """
    weather = str(weather).lower()
    
    if any(keyword in weather for keyword in ['폭우', '호우', '집중호우', '강한비', '강한 비', '폭우주의보']):
        return '강한비'
    elif any(keyword in weather for keyword in ['비', '소나기', '가랑비', '이슬비', '약한비', '약한 비']):
        return '약한비'
    elif any(keyword in weather for keyword in ['눈', '진눈깨비', '우박', '싸락눈', '폭설', '강설']):
        return '눈'
    elif any(keyword in weather for keyword in ['연무', '안개', '박무', '황사', '미세먼지', '먼지', '안개주의보']):
        return '연무/안개'
    elif any(keyword in weather for keyword in ['맑음', '청명', '쾌청']):
        return '맑음'
    elif any(keyword in weather for keyword in ['구름조금', '구름 조금', '약간흐림', '구름깔림']):
        return '구름조금'
    elif any(keyword in weather for keyword in ['구름많음', '구름 많음', '사이사이 구름']):
        return '구름많음'
    elif any(keyword in weather for keyword in ['흐림', '흐릿함', '완전흐림', '구름가득']):
        return '흐림'
    else:
        return '기타'

# ---------------- 공통 상수 ----------------
BASE_URL = "https://www.weather.go.kr/w/observation/land/city-obs.do"
COMMON_QS = {
    "auto_man": "m",
    "stn": "0",
    "dtm": "",
    "type": "t99",
    "reg": "109",  # 서울·인천·경기도
}
COLS = [
    "date", "time",
    "일기", "시정", "운량", "중하운량",
    "현재기온", "이슬점온도", "체감온도",
    "일강수", "적설", "습도", "풍향", "풍속", "해면기압",
]

# ---------------- 유틸 함수 ----------------
def build_params(ts: dt.datetime) -> dict:
    tm_raw = ts.strftime("%Y.%m.%d.%H:00")
    return COMMON_QS | {"tm": tm_raw}

def parse_row(tr, ts):
    if tr is None:
        return None

    tds = tr.find_all("td")
    cells = []
    for td in tds:
        if td.script:  # 풍속 셀
            m = re.search(r"writeWindSpeed\('([\d.]+)'", td.script.string)
            cells.append(float(m.group(1)) if m else None)
        else:
            txt = td.get_text(strip=True).replace("−", "-")
            cells.append(
                float(txt) if re.fullmatch(r"-?\d+(\.\d+)?", txt) else (txt or None)
            )

    # 적설(tds 인덱스 8)이 없어서 cells가 12개라면, 0.0을 끼워넣기
    if len(cells) == 12:
        cells.insert(8, 0.0)

    # 그래도 13개가 안 되면 건너뛰기
    if len(cells) != 13:
        return None

    return {
        "date": ts.strftime("%Y-%m-%d"),
        "time": ts.strftime("%H:%M"),
        **dict(zip(COLS[2:], cells))
    }

def fetch_current_weather(retries=3, backoff=3):
    for attempt in range(1, retries+1):
        try:
            resp = requests.get(BASE_URL, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            full_text = soup.get_text(" ", strip=True)   # <br>, 공백 등을 한 칸으로 치환

            # 2) yyyy.mm.dd.hh:mm 형태에 매칭되는 정규식
            dt_pattern = re.compile(r"\b\d{4}\.\d{2}\.\d{2}\.\d{2}:\d{2}\b")

            # 3) 첫 번째(or 모든) 매칭 가져오기
            matches = dt_pattern.findall(full_text)
            # 4) datetime 객체로 변환
            ts = dt.datetime.strptime(matches[0], "%Y.%m.%d.%H:%M") if matches else None
            link = soup.select_one('a[href*="stn=108"]')  # 서울 stn=108
            if not link:
                print(f"[{ts}] ❌ 서울 관측 데이터 없음")
                return None
            return parse_row(link.find_parent('tr'), ts)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                print(f"[{ts}] 요청 실패({attempt}회): {e}")
    return None

# 저장된 모델을 로드하는 함수
@st.cache_resource
def load_congestion_model():
    """
    저장된 XGBoost 모델을 로드
    """
    model = joblib.load('subway_xgb_best.pkl')
    st.success("XGBoost 모델을 성공적으로 로드했습니다.")
    return model


# 현재 날씨 데이터로 모든 역의 혼잡도 예측
def predict_all_stations_congestion(model, current_weather, all_stations, data_df):
    """
    현재 날씨 데이터를 기반으로 모든 역의 혼잡도 예측
    """
    # 현재 시간과 요일 정보
    now = dt.datetime.now()
    current_weekday = now.weekday()  # 월=0, 일=6
    
    # 추가: 현재 시간에서 hour, month, day 추출
    current_hour = now.hour
    current_month = now.month
    current_day = now.day
    
    # 2025년 한국 공휴일 정보
    holidays_2025 = [
        "2025-01-01",  # 신정
        "2025-01-27",  # 설날 연휴
        "2025-01-28",  # 설날
        "2025-01-29",  # 설날 연휴
        "2025-01-30",  # 설날 연휴
        "2025-03-01",  # 삼일절
        "2025-03-03",  # 대체 공휴일
        "2025-05-05",  # 어린이날
        "2025-05-06",  # 대체 공휴일
        "2025-06-03",  # 현충일
        "2025-06-06",  # 대체 공휴일
        "2025-08-15",  # 광복절
        "2025-10-03",  # 개천절
        "2025-10-05",  # 추석 연휴
        "2025-10-06",  # 추석
        "2025-10-07",  # 추석 연휴
        "2025-10-08",  # 추석 연휴
        "2025-10-09",  # 한글날
        "2025-12-25",  # 성탄절
    ]
    
    # 현재 날짜 문자열 형식으로 가져오기 (YYYY-MM-DD)
    current_date = now.strftime("%Y-%m-%d")
    
    # 휴일 여부 확인 (주말 또는 공휴일)
    is_holiday = 1 if current_date in holidays_2025 else 0
    
    # 결과를 저장할 데이터프레임
    results = []
    
    # 각 역별로 예측
    for _, station in all_stations.iterrows():
        try:
            # 모든 필요한 특성 구성
            input_features = {
                'line': station['line'],
                'station_code': station['station_code'],
                'station_name': station['station_name'],
                'time': current_weather.get('time', now.strftime("%H:%M")),
                '일기': current_weather.get('일기', '맑음'),
                '시정': current_weather.get('시정', 10),
                '운량': current_weather.get('운량', 0),
                '중하운량': current_weather.get('중하운량', 0),
                '현재기온': current_weather.get('현재기온', 20),
                '이슬점온도': current_weather.get('이슬점온도', 10),
                '체감온도': current_weather.get('체감온도', 20),
                '일강수': current_weather.get('일강수', 0),
                '적설': current_weather.get('적설', 0),
                '습도': current_weather.get('습도', 60),
                '풍향': current_weather.get('풍향', '남남서'),
                '풍속': current_weather.get('풍속', 2),
                '해면기압': current_weather.get('해면기압', 1013),
                'weekday': current_weekday,
                'is_holiday': is_holiday,
                # 추가: 누락된 특성 추가
                'hour': current_hour,
                'month': current_month,
                'day': current_day,
            }
            
            # 데이터프레임 생성
            X_pred = pd.DataFrame([input_features])
            
            # 승하차 인원 예측
            predicted_count = model.predict(X_pred)[0]
            
            # 혼잡도 계산 (1-10 스케일)
            congestion, percentile = calculate_congestion(predicted_count, station['station_name'], data_df)
            
            results.append({
                'station_name': station['station_name'],
                'station_code': station['station_code'],
                'line': station['line'],
                'latitude': station['latitude'], 
                'longitude': station['longitude'],
                'predicted_count': int(predicted_count),
                'congestion': congestion,
                'percentile': percentile
            })
        except Exception as e:
            st.error(f"예측 오류 ({station['station_name']}): {e}")
            # 오류 발생 시 기본값 추가
            results.append({
                'station_name': station['station_name'],
                'station_code': station['station_code'],
                'line': station['line'],
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'predicted_count': 0,
                'congestion': 1,
                'percentile': 0
            })
    
    return pd.DataFrame(results)

# 데이터 로드
data = load_data()
station_coords = load_station_coordinates()

# 데이터 로드 확인
if data.empty or station_coords.empty:
    st.error("데이터를 로드할 수 없습니다. 파일 경로와 형식을 확인하세요.")
    st.stop()
else:
    st.success("데이터가 성공적으로 로드되었습니다.")

# 혼잡도 예측 모델 로드
model = load_congestion_model()

# 현재 날씨 데이터 가져오기
current_weather = fetch_current_weather()

# 메인 화면 구성
analysis_mode = st.sidebar.selectbox(
    "분석 모드 선택",
    ["기본 분석", "역별 상세 분석", "실시간 혼잡도 예측"]
)

# 기본 분석 모드
if analysis_mode == "기본 분석":
    st.header("서울 지하철 승하차 데이터 기본 분석")
    
    # 전체 데이터셋 기본 정보
    st.subheader("데이터셋 기본 정보")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stations = data['station_name'].nunique()
        st.metric("총 역 수", f"{total_stations}")
        
    with col2:
        total_lines = data['line'].nunique()
        st.metric("총 노선 수", f"{total_lines}")
        
    with col3:
        total_days = data['date'].nunique()
        st.metric("분석 기간(일)", f"{total_days}")
        
    with col4:
        total_records = len(data)
        st.metric("총 데이터 수", f"{total_records:,}")
    
    # 상위 10개 역 대시보드
    st.subheader("승하차 인원 상위/하위 10개 역")
    
    # 역별 평균 승하차 인원
    station_avg = data.groupby(['station_name', 'line'])['count'].mean().reset_index()
    
    # 상위 10개 역
    top_stations = station_avg.sort_values('count', ascending=False).head(10)
    
    # 하위 10개 역
    bottom_stations = station_avg.sort_values('count', ascending=True).head(10)
    
    # 두 개의 열로 나누어 상위/하위 역 차트 표시
    col1, col2 = st.columns(2)
    
    with col1:
        fig_top = px.bar(
            top_stations,
            x='station_name',
            y='count',
            color='line',
            labels={'count': '평균 승하차 인원', 'station_name': '역명', 'line': '호선'},
            title='평균 승하차 인원 상위 10개 역'
        )
        fig_top.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        fig_bottom = px.bar(
            bottom_stations,
            x='station_name',
            y='count',
            color='line',
            labels={'count': '평균 승하차 인원', 'station_name': '역명', 'line': '호선'},
            title='평균 승하차 인원 하위 10개 역'
        )
        fig_bottom.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bottom, use_container_width=True)
    
    # 노선별 통계
    st.subheader("노선별 승하차 통계")
    
    line_stats = data.groupby('line')['count'].agg(['mean', 'max', 'min', 'std']).reset_index()
    line_stats.columns = ['호선', '평균', '최대', '최소', '표준편차']
    
    # 호선별로 정렬
    line_stats = line_stats.sort_values('호선')
    
    # 호선별 평균 승하차 인원
    fig = px.bar(
        line_stats,
        x='호선',
        y='평균',
        color='호선',
        text='평균',
        labels={'평균': '평균 승하차 인원', '호선': '호선'},
        title='호선별 평균 승하차 인원'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간대별 분석
    st.subheader("시간대별 승하차 패턴")
    
    hourly_data = data.groupby('hour')['count'].mean().reset_index()
    
    fig = px.line(
        hourly_data,
        x='hour',
        y='count',
        markers=True,
        labels={'count': '평균 승하차 인원', 'hour': '시간(시)'},
        title='시간대별 평균 승하차 인원'
    )
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)
    
    # 요일별 분석
    st.subheader("요일별 승하차 패턴")
    
    weekday_data = data.groupby('weekday_name')['count'].mean().reset_index()
    
    # 요일 순서 설정
    weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    weekday_data['weekday_name'] = pd.Categorical(weekday_data['weekday_name'], categories=weekday_order, ordered=True)
    weekday_data = weekday_data.sort_values('weekday_name')
    
    fig = px.bar(
        weekday_data,
        x='weekday_name',
        y='count',
        color='count',
        labels={'count': '평균 승하차 인원', 'weekday_name': '요일'},
        title='요일별 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간대별-요일별 히트맵 추가
    st.subheader("시간대별-요일별 승하차 패턴")
    
    # 시간대와 요일에 따른 평균 승하차 인원 계산
    heatmap_data = data.groupby(['hour', 'weekday_name'])['count'].mean().reset_index()
    
    # 요일 순서 설정
    heatmap_data['weekday_name'] = pd.Categorical(heatmap_data['weekday_name'], categories=weekday_order, ordered=True)
    
    # 피벗 테이블로 변환하여 히트맵 데이터 준비
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='weekday_name', values='count')
    
    # 히트맵 그리기
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="요일", y="시간(시)", color="평균 승하차 인원"),
        x=weekday_order,
        y=heatmap_pivot.index,  # 피벗 테이블의 실제 인덱스 사용
        color_continuous_scale="Viridis",
        title="시간대별-요일별 평균 승하차 인원"
    )
    
    # 레이아웃 조정
    fig.update_layout(
        xaxis_title="요일",
        yaxis_title="시간(시)",
        coloraxis_colorbar=dict(title="평균 승하차 인원"),
    )
    
    # 시간 축 설정
    fig.update_yaxes(tickmode='linear', dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 날씨와 승하차 인원 관계
    st.subheader("날씨와 승하차 인원의 관계")
    
    # 기온과 승하차 인원
    fig = px.scatter(
        data.sample(10000),  # 샘플링하여 속도 향상
        x='현재기온',
        y='count',
        trendline='ols',
        labels={'count': '승하차 인원', '현재기온': '기온(°C)'},
        title='기온과 승하차 인원의 관계'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 날씨 상태별 평균 승하차 인원
    data['simplified_weather'] = data['일기'].apply(simplify_weather)
    weather_data = data.groupby('simplified_weather')['count'].mean().reset_index()
    weather_data = weather_data.sort_values('count', ascending=False)
    
    fig = px.bar(
        weather_data,
        x='simplified_weather',
        y='count',
        color='count',
        color_continuous_scale="Blues",
        labels={'count': '평균 승하차 인원', 'simplified_weather': '날씨'},
        title='날씨별 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 강수량 구간별 승하차 인원
    data['rainfall_group'] = pd.cut(
        data['일강수'], 
        bins=[-0.1, 0, 5, 10, 20, 50, 100], 
        labels=['0 mm', '0.1~5 mm', '5.1~10 mm', '10.1~20 mm', '20.1~50 mm', '50 mm 이상']
    )
    
    rainfall_data = data.groupby('rainfall_group')['count'].mean().reset_index()
    
    fig = px.bar(
        rainfall_data,
        x='rainfall_group',
        y='count',
        color='count',
        color_continuous_scale="Blues",
        labels={'count': '평균 승하차 인원', 'rainfall_group': '강수량 구간'},
        title='강수량 구간별 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 지도 시각화
    st.subheader("서울 지하철 역 위치 및 승하차 인원")
    
    # 역별 위치와 승하차 인원 데이터 준비
    station_avg = data.groupby('station_name')['count'].mean().reset_index()
    station_map_data = station_coords.merge(station_avg, on='station_name', how='inner')
    
    # 마커 크기 조정
    station_map_data['marker_size'] = np.log1p(station_map_data['count']) * 2
    
    # 지도 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    
    # 각 역에 대한 마커 추가
    for _, row in station_map_data.iterrows():
        # 호선별 색상
        line_color = get_line_color(row['line'])
        
        # 마커 아이콘
        icon = folium.Icon(
            icon='subway', 
            prefix='fa', 
            color='white', 
            icon_color=line_color
        )
        
        # 마커 추가
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            tooltip=f"{row['station_name']} ({row['line']}호선): 평균 {row['count']:.1f}명",
            icon=icon
        ).add_to(m)
    
    # 지도 표시
    folium_static(m, width=800, height=600)

# 역별 상세 분석 모드
elif analysis_mode == "역별 상세 분석":
    st.header("역별 상세 분석")
    
    # 역 선택 필터
    line_filter = st.selectbox("호선 선택", sorted(data['line'].unique()))
    stations_in_line = sorted(data[data['line'] == line_filter]['station_name'].unique())
    selected_station = st.selectbox("역 선택", stations_in_line)
    
    # 선택한 역의 데이터만 필터링
    station_data = data[(data['station_name'] == selected_station) & (data['line'] == line_filter)]
    
    # 해당 역의 역코드 가져오기
    station_code = station_data['station_code'].iloc[0] if not station_data.empty else None
    
    # 역 정보 대시보드
    st.subheader(f"{selected_station} ({line_filter}호선) 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_count = station_data['count'].mean()
        st.metric("평균 승하차 인원", f"{avg_count:.1f}")
        
    with col2:
        max_count = station_data['count'].max()
        st.metric("최대 승하차 인원", f"{max_count:.1f}")
        
    with col3:
        total_count = station_data['count'].sum()
        st.metric("총 승하차 인원", f"{total_count:,.1f}")
        
    with col4:
        rank = data.groupby('station_name')['count'].mean().rank(ascending=False)
        station_rank = int(rank[selected_station])
        total_stations = len(rank)
        st.metric("혼잡도 순위", f"{station_rank}/{total_stations}")
    
    # 해당 역의 시간대별 패턴
    st.subheader("시간대별 승하차 패턴")
    
    hourly_station = station_data.groupby('hour')['count'].mean().reset_index()
    
    fig = px.line(
        hourly_station,
        x='hour',
        y='count',
        markers=True,
        labels={'count': '평균 승하차 인원', 'hour': '시간(시)'},
        title=f'{selected_station} 시간대별 평균 승하차 인원'
    )
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)
    
    # 요일별 패턴
    st.subheader("요일별 승하차 패턴")
    
    weekday_station = station_data.groupby('weekday_name')['count'].mean().reset_index()
    
    # 요일 순서 설정
    weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    weekday_station['weekday_name'] = pd.Categorical(weekday_station['weekday_name'], categories=weekday_order, ordered=True)
    weekday_station = weekday_station.sort_values('weekday_name')
    
    fig = px.bar(
        weekday_station,
        x='weekday_name',
        y='count',
        color='weekday_name',
        labels={'count': '평균 승하차 인원', 'weekday_name': '요일'},
        title=f'{selected_station} 요일별 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간대별-요일별 히트맵 추가
    st.subheader("시간대별-요일별 승하차 패턴")
    
    # 시간대와 요일에 따른 평균 승하차 인원 계산
    heatmap_data = station_data.groupby(['hour', 'weekday_name'])['count'].mean().reset_index()
    
    # 요일 순서 설정
    heatmap_data['weekday_name'] = pd.Categorical(heatmap_data['weekday_name'], categories=weekday_order, ordered=True)
    
    # 피벗 테이블로 변환하여 히트맵 데이터 준비
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='weekday_name', values='count')
    
    # 히트맵 그리기
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="요일", y="시간(시)", color="평균 승하차 인원"),
        x=weekday_order,
        y=heatmap_pivot.index,  # 피벗 테이블의 실제 인덱스 사용
        color_continuous_scale="Viridis",
        title=f"{selected_station} 시간대별-요일별 평균 승하차 인원"
    )
    
    # 레이아웃 조정
    fig.update_layout(
        xaxis_title="요일",
        yaxis_title="시간(시)",
        coloraxis_colorbar=dict(title="평균 승하차 인원"),
    )
    
    # 시간 축 설정
    fig.update_yaxes(tickmode='linear', dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 날씨별 패턴
    st.subheader("날씨별 승하차 패턴")
    
    # 간소화된 날씨 적용
    station_data['simplified_weather'] = station_data['일기'].apply(simplify_weather)
    weather_station = station_data.groupby('simplified_weather')['count'].mean().reset_index()
    
    fig = px.bar(
        weather_station,
        x='simplified_weather',
        y='count',
        color='simplified_weather',
        labels={'count': '평균 승하차 인원', 'simplified_weather': '날씨'},
        title=f'{selected_station} 날씨별 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 월별 패턴
    st.subheader("월별 승하차 패턴")
    
    monthly_station = station_data.groupby('month')['count'].mean().reset_index()
    
    fig = px.line(
        monthly_station,
        x='month',
        y='count',
        markers=True,
        labels={'count': '평균 승하차 인원', 'month': '월'},
        title=f'{selected_station} 월별 평균 승하차 인원'
    )
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # 공휴일 영향
    st.subheader("공휴일 영향 분석")
    
    holiday_impact = station_data.groupby('is_holiday')['count'].mean().reset_index()
    holiday_impact['is_holiday'] = holiday_impact['is_holiday'].map({0: '평일', 1: '공휴일'})
    
    fig = px.bar(
        holiday_impact,
        x='is_holiday',
        y='count',
        color='is_holiday',
        labels={'count': '평균 승하차 인원', 'is_holiday': '구분'},
        title=f'{selected_station} 공휴일 여부에 따른 평균 승하차 인원'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 시계열 데이터
    st.subheader("승하차 인원 추세")
    
    # 일별 데이터
    daily_station = station_data.groupby('date')['count'].mean().reset_index()
    
    fig = px.line(
        daily_station,
        x='date',
        y='count',
        labels={'count': '평균 승하차 인원', 'date': '날짜'},
        title=f'{selected_station} 일별 평균 승하차 인원 추세'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 위치 정보
    st.subheader("역 위치 정보")
    
    # 역 위치 정보를 역코드로 검색
    if station_code:
        station_loc = station_coords[station_coords['station_code'] == station_code]
        
        if not station_loc.empty:
            lat, lon = station_loc.iloc[0]['latitude'], station_loc.iloc[0]['longitude']
            
            # 지도 생성
            m = folium.Map(location=[lat, lon], zoom_start=15)
            
            # 호선별 색상
            line_color = get_line_color(line_filter)
            
            # 마커 아이콘
            icon = folium.Icon(
                icon='subway', 
                prefix='fa', 
                color='white', 
                icon_color=line_color
            )
            
            # 마커 추가
            folium.Marker(
                location=[lat, lon],
                tooltip=f"{selected_station} ({line_filter}호선)",
                icon=icon
            ).add_to(m)
            
            # 지도 표시
            folium_static(m, width=700, height=500)
        else:
            st.error(f"역 위치 정보를 찾을 수 없습니다. (역코드: {station_code})")
    else:
        st.error("해당 역의 코드를 찾을 수 없습니다.")

# 실시간 혼잡도 예측 모드
elif analysis_mode == "실시간 혼잡도 예측":
    st.header("실시간 지하철 혼잡도 예측")
    
    # 현재 날짜 및 시간 정보 표시
    now = dt.datetime.now()
    st.subheader(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M')}")
    
    # 현재 날씨 정보 가져오기 및 표시
    with st.spinner("현재 날씨 데이터를 가져오는 중..."):
        current_weather = fetch_current_weather()
    
    if current_weather:
        # 날씨 정보 표시
        st.subheader("현재 서울 날씨 정보")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("현재 기온", f"{current_weather.get('현재기온', '데이터 없음')}°C")
            st.metric("일기 상태", f"{current_weather.get('일기', '데이터 없음')}")
        
        with col2:
            st.metric("강수량", f"{current_weather.get('일강수', 0)}mm")
            st.metric("습도", f"{current_weather.get('습도', '데이터 없음')}%")
        
        with col3:
            st.metric("풍속", f"{current_weather.get('풍속', '데이터 없음')}m/s")
            st.metric("체감온도", f"{current_weather.get('체감온도', '데이터 없음')}°C")
        
        # 모델 로드 및 예측 수행
        with st.spinner("지하철 역별 혼잡도를 예측하는 중..."):
            # 모델 학습이 이미 되어 있음
            congestion_predictions = predict_all_stations_congestion(model, current_weather, station_coords, data)
        
        # 혼잡도 필터링 옵션
        st.subheader("혼잡도 필터링")
        congestion_filter = st.radio(
            "표시할 혼잡도 레벨 선택:",
            ["모두 표시", "여유 (1-3)", "보통 (4-7)", "혼잡 (8-10)"],
            horizontal=True
        )
        
        # 호선 필터링
        line_filter = st.multiselect(
            "표시할 호선 선택 (선택하지 않으면 모든 호선 표시):",
            sorted(station_coords['line'].unique())
        )
        
        # 데이터 필터링
        filtered_predictions = congestion_predictions.copy()
        
        if congestion_filter != "모두 표시":
            if congestion_filter == "여유 (1-3)":
                filtered_predictions = filtered_predictions[filtered_predictions['congestion'] <= 3]
            elif congestion_filter == "보통 (4-7)":
                filtered_predictions = filtered_predictions[(filtered_predictions['congestion'] >= 4) & (filtered_predictions['congestion'] <= 7)]
            elif congestion_filter == "혼잡 (8-10)":
                filtered_predictions = filtered_predictions[filtered_predictions['congestion'] >= 8]
        
        if line_filter:
            filtered_predictions = filtered_predictions[filtered_predictions['line'].isin(line_filter)]
        
        # 예측 결과 지도에 표시
        st.subheader("지하철 역별 현재 혼잡도 예측")
        
        # 지도 생성
        map_center = [37.5665, 126.9780]  # 서울시청 좌표
        m = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")
        
        # 마커 클러스터 설정
        marker_cluster = MarkerCluster().add_to(m)
        
        # 예측 결과 지도에 표시
        for idx, row in filtered_predictions.iterrows():
            # 혼잡도에 따른 색상 결정
            congestion_level = row['congestion']
            color = get_congestion_color(congestion_level)
            congestion_text = get_congestion_text(congestion_level)
            
            # 팝업 텍스트 생성
            popup_html = f"""
            <div style="width:200px">
                <h4>{row['station_name']}역 ({row['station_code']})</h4>
                <p><b>호선:</b> {row['line']}호선</p>
                <p><b>예측 승하차 인원:</b> {row['predicted_count']:,}명</p>
                <p><b>혼잡도:</b> {congestion_level}/10 ({congestion_text})</p>
                <p><b>백분위:</b> 상위 {row['percentile']:.1f}%</p>
            </div>
            """
            
            # 팝업 생성
            popup = folium.Popup(popup_html, max_width=300)
            
            # 혼잡도 아이콘 생성
            icon = folium.DivIcon(
                html=f"""
                <div style="
                    background-color: {color}; 
                    color: white; 
                    border-radius: 50%; 
                    width: 20px; 
                    height: 20px; 
                    line-height: 20px; 
                    text-align: center;
                    font-weight: bold;
                    font-size: 12px;">
                    {congestion_level}
                </div>
                """,
                icon_size=(20, 20)
            )
            
            # 마커 추가
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup,
                tooltip=f"{row['station_name']}역 (혼잡도: {congestion_level})",
                icon=icon
            ).add_to(marker_cluster)
        
        # 지도 표시
        folium_static(m, width=1000, height=600)
        
        # 데이터 테이블로도 표시
        st.subheader("역별 혼잡도 예측 결과")
        
        # 데이터 정렬
        sorted_predictions = filtered_predictions.sort_values(by='congestion', ascending=False)
        
        # 표시할 컬럼 선택 및 이름 변경
        display_columns = {
            'station_name': '역명',
            'line': '호선',
            'predicted_count': '예측 승하차인원',
            'congestion': '혼잡도',
            'percentile': '백분위'
        }
        
        # 데이터프레임 표시
        st.dataframe(
            sorted_predictions[display_columns.keys()].rename(columns=display_columns),
            hide_index=True,
            use_container_width=True,
            column_config={
                '예측 승하차인원': st.column_config.NumberColumn(format="%d명"),
                '혼잡도': st.column_config.ProgressColumn(
                    min_value=1,
                    max_value=10,
                    format="%d"
                ),
                '백분위': st.column_config.NumberColumn(format="%.1f%%")
            }
        )
        
        # 호선별 평균 혼잡도
        st.subheader("호선별 평균 혼잡도")
        line_congestion = filtered_predictions.groupby('line').agg({
            'congestion': 'mean',
            'predicted_count': 'mean',
            'station_name': 'count'  # 역 수 계산
        }).reset_index()
        
        line_congestion = line_congestion.rename(columns={
            'congestion': '평균 혼잡도',
            'predicted_count': '평균 승하차인원',
            'station_name': '역 수'
        })
        
        line_congestion['평균 혼잡도'] = line_congestion['평균 혼잡도'].round(1)
        line_congestion['평균 승하차인원'] = line_congestion['평균 승하차인원'].astype(int)
        line_congestion = line_congestion.sort_values('평균 혼잡도', ascending=False)
        
        # 호선별 혼잡도 차트
        fig = px.bar(
            line_congestion,
            x='line',
            y='평균 혼잡도',
            color='line',
            color_discrete_map={int(line): get_line_color(int(line)) for line in line_congestion['line']},
            text='평균 혼잡도',
            title='호선별 평균 혼잡도',
            labels={'line': '호선', '평균 혼잡도': '평균 혼잡도 (1-10)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("현재 날씨 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
        
        # 샘플 날씨 데이터로 대체
        st.warning("샘플 날씨 데이터로 계속 진행합니다.")
        
        sample_weather = {
            '현재기온': 22.0,
            '일강수': 0.0,
            '일기': '맑음',
            '습도': 60,
            '풍속': 2.5,
            '체감온도': 21.5,
        }
        
        # 샘플 데이터로 예측 수행
        with st.spinner("샘플 날씨 데이터로 혼잡도 예측 중..."):
            congestion_predictions = predict_all_stations_congestion(model, sample_weather, station_coords, data)
            
        st.info("샘플 날씨 데이터: 기온 22.0°C, 맑음, 강수량 0mm")
        
        # 예측 결과 테이블 표시
        st.subheader("샘플 데이터 기준 예측 결과")
        st.dataframe(
            congestion_predictions[['station_name', 'line', 'predicted_count', 'congestion']].head(20),
            hide_index=True
        )

# 푸터
st.markdown("---")
st.markdown("#### 서울 지하철 승하차 데이터와 날씨 데이터 통합 분석 ©️ 2025")