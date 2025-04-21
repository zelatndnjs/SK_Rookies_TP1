import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="지하철 역별 인구 밀집도 분석 및 예측",
    page_icon="🚇",
    layout="wide"
)

# 가상의 데이터 로드 함수 (실제 구현 시 대체 필요)
@st.cache_data
def load_data():
    # 예시 데이터 - 실제로는 크롤링이나 DB에서 데이터를 가져와야 함
    # 공휴일 정보 추가
    df = pd.DataFrame({
        '날짜': pd.date_range(start='2023-01-01', periods=365),
        '역명': np.random.choice(['강남', '홍대입구', '여의도', '신촌', '잠실'], 365),
        '승차인원': np.random.randint(1000, 20000, 365),
        '하차인원': np.random.randint(1000, 20000, 365),
        '기온': np.random.uniform(0, 30, 365),
        '강수량': np.random.uniform(0, 30, 365),
        '요일': np.random.choice(['월', '화', '수', '목', '금', '토', '일'], 365),
        '위도': np.random.uniform(37.5, 37.6, 365),
        '경도': np.random.uniform(126.9, 127.1, 365),
        '공휴일': np.random.choice([0, 1], 365, p=[0.85, 0.15])
    })
    
    # 각 역별로 대략적인 패턴을 만들기 (실제 데이터에서는 필요 없음)
    for station in df['역명'].unique():
        # 각 역에 고유한 평균과 표준편차 부여
        mean_boarding = np.random.randint(5000, 15000)
        std_boarding = mean_boarding * 0.2
        mean_alighting = np.random.randint(5000, 15000)
        std_alighting = mean_alighting * 0.2
        
        # 해당 역의 데이터 수정
        station_mask = df['역명'] == station
        
        # 기본 승하차 인원 설정
        df.loc[station_mask, '승차인원'] = np.random.normal(mean_boarding, std_boarding, station_mask.sum())
        df.loc[station_mask, '하차인원'] = np.random.normal(mean_alighting, std_alighting, station_mask.sum())
        
        # 요일별 변동 추가
        for day in ['월', '화', '수', '목', '금', '토', '일']:
            day_mask = (df['역명'] == station) & (df['요일'] == day)
            
            # 주중/주말 차이 반영
            if day in ['토', '일']:
                weekend_factor = np.random.uniform(0.6, 0.8)  # 주말은 주중보다 적음
                df.loc[day_mask, '승차인원'] *= weekend_factor
                df.loc[day_mask, '하차인원'] *= weekend_factor
            else:
                # 특정 요일에 따른 변동
                day_factor = np.random.uniform(0.9, 1.1)
                df.loc[day_mask, '승차인원'] *= day_factor
                df.loc[day_mask, '하차인원'] *= day_factor
        
        # 공휴일 영향
        holiday_mask = (df['역명'] == station) & (df['공휴일'] == 1)
        holiday_factor = np.random.uniform(0.5, 0.7)  # 공휴일은 평일보다 적음
        df.loc[holiday_mask, '승차인원'] *= holiday_factor
        df.loc[holiday_mask, '하차인원'] *= holiday_factor
    
    # 음수 값이 생길 경우 처리
    df['승차인원'] = df['승차인원'].apply(lambda x: max(1000, int(x)))
    df['하차인원'] = df['하차인원'].apply(lambda x: max(1000, int(x)))
    
    return df

# 가상의 역 정보 데이터 로드 함수
@st.cache_data
def load_station_info():
    # 예시 데이터 - 실제로는 크롤링이나 DB에서 데이터를 가져와야 함
    station_info = pd.DataFrame({
        '역명': ['강남', '홍대입구', '여의도', '신촌', '잠실'],
        '위도': [37.498095, 37.557192, 37.521624, 37.555134, 37.513111],
        '경도': [127.027610, 126.925381, 126.934131, 126.936893, 127.100019],
        '주소': [
            '서울 강남구 강남대로 396', 
            '서울 마포구 양화로 지하 160', 
            '서울 영등포구 여의나루로 지하 42', 
            '서울 서대문구 신촌로 지하 90', 
            '서울 송파구 올림픽로 지하 265'
        ],
        '호선': ['2호선', '2호선, 공항철도', '5호선, 9호선', '2호선', '2호선, 8호선']
    })
    return station_info

# 데이터 로드
df = load_data()
station_info = load_station_info()

# 혼잡도 계산 함수
def calculate_congestion(passengers, station_name):
    # 해당 역의 평균, 최대 승하차 인원 계산
    station_data = df[df['역명'] == station_name]
    avg_passengers = station_data['하차인원'].mean()
    max_passengers = station_data['하차인원'].max()
    
    # 혼잡도 계산 (1-10 척도)
    if passengers <= avg_passengers * 0.5:
        congestion = 1
    elif passengers <= avg_passengers * 0.75:
        congestion = 2
    elif passengers <= avg_passengers * 0.9:
        congestion = 3
    elif passengers <= avg_passengers:
        congestion = 4
    elif passengers <= avg_passengers * 1.1:
        congestion = 5
    elif passengers <= avg_passengers * 1.25:
        congestion = 6
    elif passengers <= avg_passengers * 1.5:
        congestion = 7
    elif passengers <= avg_passengers * 1.75:
        congestion = 8
    elif passengers <= max_passengers * 0.9:
        congestion = 9
    else:
        congestion = 10
    
    return congestion

# 가상의 예측 모델 로드 함수 (실제 구현 시 대체 필요)
def load_model():
    # 실제로는 학습된 모델을 로드해야 함
    # 예: model = joblib.load('model.pkl')
    # 여기서는 간단한 예측 함수로 대체
    def predict_model(features):
        # 예시: 간단한 예측 로직
        # 해당 역의 기존 데이터를 바탕으로 예측
        station_data = df[df['역명'] == features['역명']]
        
        # 요일별 평균 계산
        day_avg = station_data[station_data['요일'] == features['요일']]['하차인원'].mean()
        
        # 공휴일 여부에 따른 조정
        if features['공휴일'] == 1:
            holiday_factor = station_data[station_data['공휴일'] == 1]['하차인원'].mean() / station_data[station_data['공휴일'] == 0]['하차인원'].mean()
            day_avg *= holiday_factor
        
        # 날씨 영향 (기온, 강수량)
        weather_factor = 1.0
        if features['기온'] > 25:  # 더운 날
            weather_factor *= 0.95
        elif features['기온'] < 5:  # 추운 날
            weather_factor *= 0.9
        
        if features['강수량'] > 10:  # 비/눈이 많이 오는 날
            weather_factor *= 0.85
        
        # 최종 예측값
        prediction = int(day_avg * weather_factor)
        
        # 혼잡도 계산
        congestion = calculate_congestion(prediction, features['역명'])
        
        return prediction, congestion
    
    return predict_model

# 메인 함수
def main():
    # 타이틀 및 소개
    st.title("지하철 역별 인구 밀집도 분석 및 예측")
    st.markdown("""
    이 애플리케이션은 지하철역의 날씨, 날짜 등을 활용하여 역별 승하차 인원을 분석하고 예측합니다.
    """)
    
    # 첫 페이지 - 역 선택
    st.subheader("역 선택")
    
    # 역 선택 입력
    selected_station = st.selectbox(
        "분석할 역을 선택하세요",
        options=station_info['역명'].unique(),
        index=0
    )
    
    # 선택된 역 정보 표시
    selected_station_info = station_info[station_info['역명'] == selected_station].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_station}역 정보")
        st.write(f"**호선**: {selected_station_info['호선']}")
        st.write(f"**주소**: {selected_station_info['주소']}")
        
        # 역 위치 지도로 표시
        m = folium.Map(
            location=[selected_station_info['위도'], selected_station_info['경도']],
            zoom_start=15
        )
        
        folium.Marker(
            location=[selected_station_info['위도'], selected_station_info['경도']],
            popup=f"{selected_station}역",
            tooltip=f"{selected_station}역",
            icon=folium.Icon(color='red', icon='subway', prefix='fa')
        ).add_to(m)
        
        folium_static(m, width=350, height=300)
    
    with col2:
        # 해당 역의 전체 데이터에서 평균 승하차 인원 계산
        station_data = df[df['역명'] == selected_station]
        
        avg_boarding = int(station_data['승차인원'].mean())
        avg_alighting = int(station_data['하차인원'].mean())
        
        st.subheader("평균 승하차 인원")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("평균 승차인원", f"{avg_boarding:,}명")
        with col_b:
            st.metric("평균 하차인원", f"{avg_alighting:,}명")
        
        # 평균 혼잡도 계산
        avg_congestion = calculate_congestion(avg_alighting, selected_station)
        
        st.subheader("평균 혼잡도")
        
        # 혼잡도 시각화
        congestion_color = "green" if avg_congestion <= 3 else "orange" if avg_congestion <= 7 else "red"
        
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h1 style="color: {congestion_color}; font-size: 48px;">{avg_congestion}</h1>
                <p>1(여유) ~ 10(매우 혼잡) 척도</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 요일별 평균 승하차 인원
    st.subheader("요일별 평균 승하차 인원")
    
    weekday_order = ['월', '화', '수', '목', '금', '토', '일']
    weekday_df = station_data.groupby('요일').agg({
        '승차인원': 'mean',
        '하차인원': 'mean'
    }).reset_index()
    
    # 요일 순서대로 정렬
    weekday_df['요일_정렬'] = pd.Categorical(weekday_df['요일'], categories=weekday_order, ordered=True)
    weekday_df = weekday_df.sort_values('요일_정렬')
    
    # 그래프로 표시
    fig_weekday = px.bar(
        weekday_df,
        x='요일',
        y=['승차인원', '하차인원'],
        barmode='group',
        title=f"{selected_station}역 요일별 평균 승하차 인원"
    )
    st.plotly_chart(fig_weekday, use_container_width=True)
    
    # 공휴일 여부에 따른 승하차 인원 비교
    st.subheader("공휴일 여부에 따른 승하차 인원 비교")
    
    holiday_df = station_data.groupby('공휴일').agg({
        '승차인원': 'mean',
        '하차인원': 'mean'
    }).reset_index()
    
    holiday_df['공휴일'] = holiday_df['공휴일'].map({0: '평일', 1: '공휴일'})
    
    fig_holiday = px.bar(
        holiday_df,
        x='공휴일',
        y=['승차인원', '하차인원'],
        barmode='group',
        title=f"{selected_station}역 공휴일 여부에 따른 평균 승하차 인원 비교"
    )
    st.plotly_chart(fig_holiday, use_container_width=True)
    
    # 기온과 강수량에 따른 하차인원 변화 (산점도)
    st.subheader("날씨에 따른 하차인원 변화")
    
    fig_weather = px.scatter(
        station_data,
        x='기온',
        y='하차인원',
        color='강수량',
        color_continuous_scale='blues',
        size='하차인원',
        size_max=15,
        hover_data=['날짜', '요일', '공휴일'],
        title=f"{selected_station}역 날씨에 따른 하차인원 변화"
    )
    st.plotly_chart(fig_weather, use_container_width=True)
    
    # 날짜별 승하차 인원 추이
    st.subheader("날짜별 승하차 인원 추이")
    
    # 날짜 범위 선택기
    date_range = st.date_input(
        "날짜 범위 선택",
        [station_data['날짜'].min().date(), station_data['날짜'].max().date()],
        min_value=station_data['날짜'].min().date(),
        max_value=station_data['날짜'].max().date()
    )
    
    if len(date_range) == 2:
        filtered_data = station_data[
            (station_data['날짜'].dt.date >= date_range[0]) & 
            (station_data['날짜'].dt.date <= date_range[1])
        ]
        
        fig_trend = px.line(
            filtered_data,
            x='날짜',
            y=['승차인원', '하차인원'],
            title=f"{selected_station}역 승하차 인원 추이 ({date_range[0]} ~ {date_range[1]})"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # 구분선
    st.markdown("---")
    
    # 예측 섹션
    st.header("미래 날짜의 혼잡도 예측")
    
    # 모델 로드
    model = load_model()
    
    # 예측을 위한 입력값 설정
    col1, col2 = st.columns(2)
    
    with col1:
        predict_date = st.date_input(
            "예측할 날짜 선택",
            value=datetime.date.today() + datetime.timedelta(days=1),
            min_value=datetime.date.today(),
            max_value=datetime.date.today() + datetime.timedelta(days=365)
        )
        
        # 날짜로부터 요일 계산
        weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
        predict_weekday = weekday_map[predict_date.weekday()]
        
        st.write(f"선택한 날짜의 요일: {predict_weekday}")
        
        predict_holiday = st.checkbox("공휴일 여부")
    
    with col2:
        predict_temp = st.slider(
            "예상 기온 (℃)",
            min_value=-10.0,
            max_value=40.0,
            value=20.0,
            step=0.1
        )
        
        predict_rain = st.slider(
            "예상 강수량 (mm)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )
    
    # 예측 실행
    if st.button("혼잡도 예측하기"):
        # 예측을 위한 특성 데이터 준비
        features = {
            '역명': selected_station,
            '날짜': predict_date,
            '요일': predict_weekday,
            '기온': predict_temp,
            '강수량': predict_rain,
            '공휴일': 1 if predict_holiday else 0
        }
        
        # 예측 실행
        prediction, congestion = model(features)
        
        st.subheader("예측 결과")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="예상 하차인원",
                value=f"{prediction:,}명"
            )
        
        with col2:
            # 혼잡도 시각화
            congestion_color = "green" if congestion <= 3 else "orange" if congestion <= 7 else "red"
            congestion_text = "여유" if congestion <= 3 else "보통" if congestion <= 7 else "매우 혼잡"
            
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2>예상 혼잡도</h2>
                    <h1 style="color: {congestion_color}; font-size: 60px;">{congestion}</h1>
                    <h3 style="color: {congestion_color};">{congestion_text}</h3>
                    <p>1(여유) ~ 10(매우 혼잡) 척도</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 요일별 평균과 비교
        st.subheader("요일별 평균 대비 예상 인원")
        
        # 해당 요일의 평균 하차인원
        avg_day_passengers = station_data[station_data['요일'] == predict_weekday]['하차인원'].mean()
        
        # 평균 대비 백분율
        percentage = (prediction / avg_day_passengers) * 100
        
        # 비교 그래프
        comparison_data = pd.DataFrame({
            '구분': ['예상 하차인원', f'{predict_weekday}요일 평균'],
            '인원': [prediction, avg_day_passengers]
        })
        
        fig_comparison = px.bar(
            comparison_data,
            x='구분',
            y='인원',
            title=f"{predict_date} ({predict_weekday}) {selected_station}역 하차인원 예측 vs 평균",
            text_auto=True
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.write(f"예상 하차인원은 {predict_weekday}요일 평균 대비 {percentage:.1f}% 수준입니다.")
        
        # 예측에 영향을 준 요인 설명
        st.subheader("예측에 영향을 준 요인")
        
        factors = []
        
        if predict_holiday:
            factors.append("• 공휴일은 평균적으로 평일보다 하차인원이 적습니다.")
        
        if predict_weekday in ['토', '일']:
            factors.append("• 주말은 평일보다 하차인원이 적은 경향이 있습니다.")
        
        if predict_temp > 25:
            factors.append("• 기온이 높은 날은 하차인원이 소폭 감소하는 경향이 있습니다.")
        elif predict_temp < 5:
            factors.append("• 기온이 낮은 날은 하차인원이 감소하는 경향이 있습니다.")
        
        if predict_rain > 10:
            factors.append("• 강수량이 많은 날은 하차인원이 감소하는 경향이 있습니다.")
        
        for factor in factors:
            st.markdown(factor)
        
        if not factors:
            st.write("특별한 요인이 없습니다. 해당 요일의 평균적인 패턴을 따를 것으로 예상됩니다.")

# 앱 실행
if __name__ == "__main__":
    main()
