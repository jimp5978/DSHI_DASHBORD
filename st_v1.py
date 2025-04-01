import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime


# 페이지 설정
st.set_page_config(
    page_title="DSHI HY",
    page_icon="assets/DSHI_1.jpg",  # 페이지 설정에 로고 추가
    layout="wide"
)

# 사이드바에 로고 추가
st.sidebar.image("assets/DSHI.jpg", use_container_width=True)

# 제목
st.title("대만공항 함양공장 PKG3 생산현황")

# 데이터 준비
@st.cache_data
def load_data():
    return pd.read_excel("assets/PKG3_data.xlsx")

def preprocess_data(df):
    # Weight 컬럼 전처리 (kg to ton)
    df['Weight'] = df['Weight'].apply(lambda x: 0 if isinstance(x, str) else x)
    df['Weight'] = df['Weight'].fillna(0)
    df['Weight'] = df['Weight'] / 1000  # kg to ton
    
    # 날짜 컬럼 처리
    date_cols = ['FIT-UP', 'FINAL', 'GALVA', 'PAINT']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        # 연도와 월 컬럼 추가
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
    
    # 다른 컬럼의 NaN 값 처리
    for col in df.columns:
        if col not in date_cols and col != 'Weight' and not col.endswith('_year') and not col.endswith('_month'):
            df[col] = df[col].fillna('미완료')
    
    return df

# 데이터 로드
df = preprocess_data(load_data())

# 날짜 필터링
st.sidebar.subheader('날짜 필터')
# 연도 선택
years = []
for process in ['FIT-UP', 'FINAL', 'GALVA', 'PAINT']:
    process_years = df[f'{process}_year'].dropna().unique()
    years.extend(process_years)
years = sorted(list(set(years)))
selected_years = st.sidebar.multiselect('연도 선택', options=years, default=years)

# 월 선택
months = list(range(1, 13))
selected_months = st.sidebar.multiselect('월 선택', options=months, default=months)

# Company 선택
st.sidebar.subheader('Company 선택')
companies = sorted(df['Company'].unique())
default_companies = companies[:3] if len(companies) >= 3 else companies
selected_companies = st.sidebar.multiselect('', options=companies, default=default_companies)

# 공정 선택
st.sidebar.subheader('공정 선택')
process_cols = st.sidebar.columns(4)
selected_processes = []
for i, process in enumerate(['FIT-UP', 'FINAL', 'GALVA', 'PAINT']):
    with process_cols[i]:
        if st.checkbox(process, value=True, key=f'process_{process}'):
            selected_processes.append(process)

# ITEM 선택
st.sidebar.subheader('ITEM 선택')
items = sorted(df['ITEM'].unique())
selected_items = st.sidebar.multiselect('', options=items, default=items)

# Zone 선택
st.sidebar.subheader('Zone 선택')
zones = sorted(df['Zone'].unique())
selected_zones = st.sidebar.multiselect('', options=zones, default=zones)

# 필터링된 데이터
filtered_df = df[
    (df['Zone'].isin(selected_zones)) &
    (df['Company'].isin(selected_companies)) &
    (df['ITEM'].isin(selected_items))
]

# 날짜 필터링
if selected_years and selected_months:
    date_mask = pd.Series(False, index=filtered_df.index)
    for process in selected_processes:
        process_mask = (filtered_df[f'{process}_year'].isin(selected_years)) & \
                      (filtered_df[f'{process}_month'].isin(selected_months))
        date_mask = date_mask | process_mask
    filtered_df = filtered_df[date_mask]

# 주요 지표
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
st.subheader("주요 지표")
col1, col2 = st.columns(2)

with col1:
    st.metric("총 Assembly 수", len(filtered_df))
with col2:
    total_weight = filtered_df['Weight'].sum()
    st.metric("총 Weight (Ton)", f"{total_weight:,.2f}")

# 주요 지표 섹션 다음에 추가
st.subheader("월별 생산 추이")

# 월별 데이터 준비
monthly_data = []
for year in selected_years:
    for month in selected_months:
        month_data = {
            '년월': f'{year}-{month:02d}',
            '생산량(Ton)': 0
        }
        for company in selected_companies:
            company_df = filtered_df[
                (filtered_df['Company'] == company) &
                (filtered_df['PAINT_year'] == year) &
                (filtered_df['PAINT_month'] == month)
            ]
            weight = company_df['Weight'].sum()
            if weight > 0:  # 생산량이 있는 경우만 추가
                monthly_data.append({
                    '년월': f'{year}-{month:02d}',
                    'Company': company,
                    '생산량(Ton)': weight
                })

monthly_df = pd.DataFrame(monthly_data)

# 데이터가 있는 경우만 표시
if not monthly_df.empty:
    # 월별 생산 추이 차트 - stack 형태로 변경
    fig_monthly = px.bar(
        monthly_df,
        x='년월',
        y='생산량(Ton)',
        color='Company',
        barmode='stack',  # 'group'에서 'stack'으로 변경
        title='월별 Company별 생산량',
        text_auto=False  # 기본 text_auto 비활성화
    )

    # 텍스트 포맷 설정 (소수점 1자리 표시)
    fig_monthly.update_traces(
        texttemplate='%{y:.1f}',  # y값을 소수점 1자리로 표시
        textposition='outside'  # 텍스트를 막대 외부에 표시
    )

    # x축 레이아웃 수정
    fig_monthly.update_layout(
        xaxis_title="년월",
        yaxis_title="생산량 (Ton)",
        bargap=0.2
    )

    # 차트 표시
    st.plotly_chart(fig_monthly, use_container_width=True)


# 2개의 컬럼으로 레이아웃 구성
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Weight by Zone
    zone_weight = filtered_df.groupby('Zone')['Weight'].sum().reset_index()
    fig_bar = px.bar(
        zone_weight,
        x='Zone',
        y='Weight',
        title='Zone별 총 Weight (Ton)',
        text_auto=True  # 막대그래프에 값 표시
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # Item distribution
    fig_pie = px.pie(filtered_df, 
                     names='ITEM',
                     title='ITEM 별 분포')
    st.plotly_chart(fig_pie, use_container_width=True)

# 진행 상황 표시
st.subheader('공정 진행 현황')
progress_data = []

for col in selected_processes:
    process_data = {'공정': col}
    total_completion = 0
    for company in selected_companies:
        company_df = filtered_df[filtered_df['Company'] == company]
        total = len(company_df)
        completed = len(company_df[company_df[col].notna()])
        completion_rate = (completed / total) * 100 if total > 0 else 0
        progress_data.append({
            '공정': col,
            '완료율': completion_rate,
            'Company': company
        })

progress_df = pd.DataFrame(progress_data)

# 공정별 완료율 (%) 차트
fig_progress = px.bar(
    progress_df, 
    x='공정', 
    y='완료율',
    color='Company',
    barmode='stack',  # 'group'에서 'stack'으로 변경
    title='공정별 완료율 (%)',
    text_auto=False  # 기본 text_auto 비활성화
)

# 텍스트 포맷 설정 (소수점 1자리와 % 추가)
fig_progress.update_traces(
    texttemplate='%{y:.1f}%',  # y값을 소수점 1자리와 %로 표시
    textposition='outside'  # 텍스트를 막대 외부에 표시
)

# 차트 표시
st.plotly_chart(fig_progress, use_container_width=True)

