# =============================================================================
# app.py - 통합 지표 모니터링 대시보드 (Streamlit Cloud 배포용)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 설정
# =============================================================================
DATA_PATH = "data/daily_clipping.xlsm"

DATA_COLUMNS = [
    "날짜", "달러환율", "엔환율", "유로환율", "위안화환율",
    "육지 가격", "육지 거래량", "제주 가격", "제주 거래량",
    "육지 SMP", "제주 SMP", "두바이유", "브렌트유", "WTI",
    "탱크로리용", "연료전지용", "콜금리(1일)", "CD (91일)", "CP (91일)",
    "국고채 (3년)", "국고채 (5년)", "국고채 (10년)", "산금채 (1년)",
    "회사채 (3년)(AA-)", "회사채 (3년)(BBB-)",
    "IRS (3년)", "IRS (5년)", "IRS (10년)", "CRS (1년)", "CRS (3년)"
]

INDICATORS = {
    "환율": {
        "icon": "💱", "color": "#3498db",
        "columns": {
            "달러환율": {"unit": "원", "format": "{:,.1f}"},
            "엔환율": {"unit": "원/100엔", "format": "{:,.2f}"},
            "유로환율": {"unit": "원", "format": "{:,.2f}"},
            "위안화환율": {"unit": "원", "format": "{:,.2f}"},
        }
    },
    "REC": {
        "icon": "📗", "color": "#27ae60",
        "columns": {
            "육지 가격": {"unit": "원/REC", "format": "{:,.0f}"},
            "육지 거래량": {"unit": "REC", "format": "{:,.0f}"},
            "제주 가격": {"unit": "원/REC", "format": "{:,.0f}"},
            "제주 거래량": {"unit": "REC", "format": "{:,.0f}"},
        }
    },
    "SMP": {
        "icon": "⚡", "color": "#f39c12",
        "columns": {
            "육지 SMP": {"unit": "원/kWh", "format": "{:,.2f}"},
            "제주 SMP": {"unit": "원/kWh", "format": "{:,.2f}"},
        }
    },
    "유가": {
        "icon": "🛢️", "color": "#e74c3c",
        "columns": {
            "두바이유": {"unit": "$/배럴", "format": "{:,.2f}"},
            "브렌트유": {"unit": "$/배럴", "format": "{:,.2f}"},
            "WTI": {"unit": "$/배럴", "format": "{:,.2f}"},
        }
    },
    "LNG": {
        "icon": "🔥", "color": "#9b59b6",
        "columns": {
            "탱크로리용": {"unit": "원/MJ", "format": "{:,.4f}"},
            "연료전지용": {"unit": "원/MJ", "format": "{:,.4f}"},
        }
    },
    "금리": {
        "icon": "📊", "color": "#1abc9c",
        "columns": {
            "콜금리(1일)": {"unit": "%", "format": "{:,.3f}"},
            "CD (91일)": {"unit": "%", "format": "{:,.2f}"},
            "CP (91일)": {"unit": "%", "format": "{:,.2f}"},
            "국고채 (3년)": {"unit": "%", "format": "{:,.3f}"},
            "국고채 (5년)": {"unit": "%", "format": "{:,.3f}"},
            "국고채 (10년)": {"unit": "%", "format": "{:,.3f}"},
            "산금채 (1년)": {"unit": "%", "format": "{:,.3f}"},
            "회사채 (3년)(AA-)": {"unit": "%", "format": "{:,.3f}"},
            "회사채 (3년)(BBB-)": {"unit": "%", "format": "{:,.3f}"},
        }
    },
    "스왑": {
        "icon": "🔄", "color": "#34495e",
        "columns": {
            "IRS (3년)": {"unit": "%", "format": "{:,.4f}"},
            "IRS (5년)": {"unit": "%", "format": "{:,.4f}"},
            "IRS (10년)": {"unit": "%", "format": "{:,.4f}"},
            "CRS (1년)": {"unit": "%", "format": "{:,.2f}"},
            "CRS (3년)": {"unit": "%", "format": "{:,.2f}"},
        }
    },
}

CHART_PERIODS = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "전체": None}

ALERT_THRESHOLDS = {
    "환율": 1.0, "REC": 3.0, "SMP": 5.0, "유가": 3.0,
    "LNG": 5.0, "금리": 0.1, "스왑": 0.1,
}

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="📊 데일리 클리핑 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS 스타일
# =============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f3460 0%, #1a1a2e 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #ffffff; font-size: 2rem; margin: 0; }
    .main-header p { color: #aaaaaa; margin: 0.5rem 0 0 0; font-size: 0.9rem; }
    
    .metric-card {
        background: linear-gradient(145deg, #16213e 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #0f3460;
        margin-bottom: 1rem;
    }
    .metric-card:hover { border-color: #e94560; }
    .metric-title { color: #888888; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .metric-value { color: #ffffff; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .metric-change-up { color: #00d26a; font-size: 0.9rem; font-weight: 600; }
    .metric-change-down { color: #ff6b6b; font-size: 0.9rem; font-weight: 600; }
    .metric-change-neutral { color: #888888; font-size: 0.9rem; }
    
    .category-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 1rem;
        background: linear-gradient(90deg, #0f3460 0%, transparent 100%);
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid;
    }
    .category-header h3 { color: #ffffff; margin: 0; font-size: 1.1rem; }
    
    .alert-box {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.2) 0%, transparent 100%);
        border-left: 4px solid #e94560;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .alert-box h4 { color: #e94560; margin: 0 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 데이터 로딩
# =============================================================================
@st.cache_data(ttl=300)
def load_data():
    try:
        df = pd.read_excel(DATA_PATH, sheet_name="Data", skiprows=4, usecols="B:AE", engine='openpyxl')
        df.columns = DATA_COLUMNS
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        df = df.dropna(subset=['날짜'])
        df = df.sort_values('날짜').reset_index(drop=True)
        
        numeric_cols = [col for col in df.columns if col != '날짜']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 빈 데이터 행 제거
        key_cols = ['달러환율', '육지 SMP', '두바이유']
        mask = df[key_cols].notna().any(axis=1)
        df = df[mask].reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return None

def get_summary(df):
    if df is None or len(df) < 2:
        return {}
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    summary = {}
    
    for category, info in INDICATORS.items():
        is_rate = category in ['금리', '스왑']
        summary[category] = {'icon': info['icon'], 'color': info['color'], 'indicators': {}}
        
        for col_name, col_info in info['columns'].items():
            current = latest.get(col_name)
            prev = previous.get(col_name)
            
            if pd.notna(current) and pd.notna(prev) and prev != 0:
                change = current - prev
                change_pct = (change / prev) * 100 if not is_rate else change * 100
                direction = 'up' if change > 0 else ('down' if change < 0 else 'neutral')
            else:
                change, change_pct, direction = None, None, 'neutral'
            
            summary[category]['indicators'][col_name] = {
                'value': current, 'previous': prev, 'change': change,
                'change_pct': change_pct, 'direction': direction,
                'unit': col_info['unit'], 'format': col_info['format']
            }
    
    return summary

def check_alerts(summary):
    alerts = []
    for category, data in summary.items():
        threshold = ALERT_THRESHOLDS.get(category, 5.0)
        is_rate = category in ['금리', '스왑']
        
        for col_name, ind in data['indicators'].items():
            if ind['change_pct'] is None:
                continue
            
            check_val = abs(ind['change']) * 100 if is_rate else abs(ind['change_pct'])
            threshold_val = threshold * 100 if is_rate else threshold
            
            if check_val >= threshold_val:
                alerts.append({
                    'category': category, 'indicator': col_name,
                    'change_pct': ind['change_pct'], 'direction': ind['direction'],
                    'icon': data['icon']
                })
    return alerts

def format_value(value, fmt, unit=""):
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{fmt.format(value)} {unit}"
    except:
        return str(value)

def get_change_html(change, change_pct, direction, is_rate=False):
    if change is None:
        return '<span class="metric-change-neutral">-</span>'
    
    arrow = "▲" if direction == 'up' else ("▼" if direction == 'down' else "―")
    css = "metric-change-up" if direction == 'up' else ("metric-change-down" if direction == 'down' else "metric-change-neutral")
    
    if is_rate:
        return f'<span class="{css}">{arrow} {abs(change)*100:.1f}bp</span>'
    return f'<span class="{css}">{arrow} {abs(change_pct):.2f}%</span>'

def create_metric_card(title, value, change_html):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div>{change_html}</div>
    </div>
    """

# =============================================================================
# 메인 앱
# =============================================================================
def main():
    df = load_data()
    
    if df is None or len(df) == 0:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return
    
    latest_date = df['날짜'].max()
    
    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📂 카테고리 필터")
        categories = list(INDICATORS.keys())
        selected_categories = st.multiselect("표시할 카테고리", categories, default=categories)
        
        st.markdown("---")
        st.markdown("### 📅 차트 기간")
        selected_period = st.selectbox("기간 선택", list(CHART_PERIODS.keys()), index=2)
        
        st.markdown("---")
        st.markdown(f"""
        ### 📋 데이터 정보
        - **최신 날짜:** {latest_date.strftime('%Y-%m-%d')}
        - **총 데이터:** {len(df):,}행
        """)
    
    # 메인 헤더
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 데일리 클리핑 통합 지표 대시보드</h1>
        <p>📅 기준일: {latest_date.strftime('%Y년 %m월 %d일')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_summary(df)
    
    # 알림
    alerts = check_alerts(summary)
    if alerts:
        st.markdown(f'<div class="alert-box"><h4>🚨 급변동 알림 ({len(alerts)}건)</h4></div>', unsafe_allow_html=True)
        cols = st.columns(min(len(alerts), 4))
        for i, alert in enumerate(alerts[:4]):
            with cols[i % 4]:
                direction = "▲" if alert['direction'] == 'up' else "▼"
                color = "#00d26a" if alert['direction'] == 'up' else "#ff6b6b"
                st.markdown(f"""
                <div style="background: rgba(233,69,96,0.1); padding: 0.8rem; border-radius: 8px; border: 1px solid {color};">
                    <div style="color: #888; font-size: 0.8rem;">{alert['icon']} {alert['category']}</div>
                    <div style="color: #fff; font-weight: bold;">{alert['indicator']}</div>
                    <div style="color: {color}; font-weight: bold;">{direction} {abs(alert['change_pct']):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 탭
    tab1, tab2, tab3 = st.tabs(["📈 지표 현황", "📊 차트 분석", "📋 데이터 테이블"])
    
    # TAB 1: 지표 현황
    with tab1:
        for category in selected_categories:
            if category not in summary:
                continue
            data = summary[category]
            
            st.markdown(f"""
            <div class="category-header" style="border-color: {data['color']};">
                <span style="font-size: 1.5rem;">{data['icon']}</span>
                <h3>{category}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(4)
            is_rate = category in ['금리', '스왑']
            
            for i, (col_name, ind) in enumerate(data['indicators'].items()):
                with cols[i % 4]:
                    value_str = format_value(ind['value'], ind['format'], ind['unit'])
                    change_html = get_change_html(ind['change'], ind['change_pct'], ind['direction'], is_rate)
                    st.markdown(create_metric_card(col_name, value_str, change_html), unsafe_allow_html=True)
    
    # TAB 2: 차트 분석
    with tab2:
        st.markdown("### 📈 지표 추이 차트")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            chart_category = st.selectbox("카테고리", selected_categories, key="chart_cat")
            if chart_category:
                available = list(INDICATORS[chart_category]['columns'].keys())
                chart_indicators = st.multiselect("지표 선택", available, default=available[:2])
        
        with col2:
            if chart_category and chart_indicators:
                days = CHART_PERIODS.get(selected_period)
                df_chart = df.copy()
                if days:
                    cutoff = latest_date - timedelta(days=days)
                    df_chart = df_chart[df_chart['날짜'] >= cutoff]
                
                fig = go.Figure()
                colors = px.colors.qualitative.Set2
                for i, ind in enumerate(chart_indicators):
                    fig.add_trace(go.Scatter(
                        x=df_chart['날짜'], y=df_chart[ind],
                        mode='lines', name=ind,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(22,33,62,0.8)',
                    plot_bgcolor='rgba(22,33,62,0.8)',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔄 다중 지표 비교")
        
        compare_options = ['달러환율', '육지 SMP', '두바이유', '국고채 (3년)', 'IRS (3년)']
        compare_indicators = st.multiselect("비교할 지표 (최대 4개)", compare_options, default=['달러환율', '육지 SMP'], max_selections=4)
        
        if compare_indicators:
            days = CHART_PERIODS.get(selected_period)
            df_compare = df.copy()
            if days:
                cutoff = latest_date - timedelta(days=days)
                df_compare = df_compare[df_compare['날짜'] >= cutoff]
            
            df_norm = df_compare[['날짜'] + compare_indicators].copy()
            for col in compare_indicators:
                first = df_norm[col].dropna().iloc[0] if len(df_norm[col].dropna()) > 0 else 1
                df_norm[col] = (df_norm[col] / first) * 100
            
            fig2 = go.Figure()
            for col in compare_indicators:
                fig2.add_trace(go.Scatter(x=df_norm['날짜'], y=df_norm[col], mode='lines', name=col))
            
            fig2.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(22,33,62,0.8)',
                plot_bgcolor='rgba(22,33,62,0.8)',
                height=350,
                yaxis_title="상대 변화율 (시작=100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 3: 데이터 테이블
    with tab3:
        st.markdown("### 📋 원본 데이터 조회")
        
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("날짜 범위", value=(latest_date - timedelta(days=30), latest_date))
        with col2:
            table_category = st.selectbox("카테고리", ['전체'] + list(INDICATORS.keys()))
        
        df_table = df.copy()
        if len(date_range) == 2:
            start, end = date_range
            df_table = df_table[(df_table['날짜'] >= pd.to_datetime(start)) & (df_table['날짜'] <= pd.to_datetime(end))]
        
        if table_category != '전체':
            cols = ['날짜'] + list(INDICATORS[table_category]['columns'].keys())
            df_table = df_table[cols]
        
        df_display = df_table.copy()
        df_display['날짜'] = df_display['날짜'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(df_display.sort_values('날짜', ascending=False), use_container_width=True, height=400)
        
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 CSV 다운로드", csv, f"daily_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        📊 데일리 클리핑 통합 지표 대시보드 | 데이터 출처: 서울외국환중개, 신재생 원스톱 포털, 한국석유공사, 한국가스공사, 경제통계시스템
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
