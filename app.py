# =============================================================================
# app.py - 통합 지표 모니터링 대시보드 v3.0
# 기능: 지표 현황, 상관관계 분석, 회귀분석 예측, 차트 분석
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 설정
# =============================================================================
DATA_PATH = "data/데일리_클리핑_자료.xlsm"

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

# 상관관계 분석용 주요 지표
KEY_INDICATORS = [
    "달러환율", "유로환율", "위안화환율",
    "육지 SMP", "제주 SMP",
    "두바이유", "브렌트유", "WTI",
    "국고채 (3년)", "국고채 (5년)", "국고채 (10년)",
    "IRS (3년)", "IRS (5년)"
]

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="📊 데일리 클리핑 대시보드 v3.0",
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
    
    .alert-item {
        background: rgba(233,69,96,0.1);
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid;
        margin-bottom: 0.5rem;
    }
    
    .insight-box {
        background: linear-gradient(145deg, #1a3a5c 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3498db;
        margin: 1rem 0;
    }
    .insight-box h4 { color: #3498db; margin: 0 0 0.8rem 0; }
    .insight-box p { color: #ffffff; margin: 0.3rem 0; line-height: 1.6; }
    
    .prediction-box {
        background: linear-gradient(145deg, #1a4a3c 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #27ae60;
        margin: 1rem 0;
    }
    .prediction-box h4 { color: #27ae60; margin: 0 0 0.8rem 0; }
    
    .correlation-strong { color: #00d26a; font-weight: bold; }
    .correlation-moderate { color: #f39c12; font-weight: bold; }
    .correlation-weak { color: #888888; }
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

# =============================================================================
# 상관관계 분석 함수
# =============================================================================
def calculate_correlation_matrix(df, columns, days=365):
    """지표 간 상관관계 매트릭스 계산"""
    if days:
        cutoff = df['날짜'].max() - timedelta(days=days)
        df_filtered = df[df['날짜'] >= cutoff]
    else:
        df_filtered = df
    
    df_corr = df_filtered[columns].dropna()
    return df_corr.corr()

def calculate_lagged_correlation(df, leading_col, lagging_col, max_lag=30):
    """시차(Lag) 상관관계 계산"""
    results = []
    df_clean = df[['날짜', leading_col, lagging_col]].dropna()
    
    for lag in range(0, max_lag + 1):
        if lag == 0:
            corr, p_value = stats.pearsonr(df_clean[leading_col], df_clean[lagging_col])
        else:
            leading_shifted = df_clean[leading_col].iloc[:-lag].values
            lagging_current = df_clean[lagging_col].iloc[lag:].values
            
            if len(leading_shifted) > 10:
                corr, p_value = stats.pearsonr(leading_shifted, lagging_current)
            else:
                corr, p_value = np.nan, np.nan
        
        results.append({
            'lag': lag,
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return pd.DataFrame(results)

def find_optimal_lag(lag_df):
    """최적 시차 찾기"""
    valid_df = lag_df.dropna()
    if len(valid_df) == 0:
        return None
    idx = valid_df['correlation'].abs().idxmax()
    return valid_df.loc[idx]

def interpret_correlation(corr):
    """상관계수 해석"""
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "강한"
        css_class = "correlation-strong"
    elif abs_corr >= 0.4:
        strength = "중간"
        css_class = "correlation-moderate"
    else:
        strength = "약한"
        css_class = "correlation-weak"
    
    direction = "양의" if corr > 0 else "음의"
    return strength, direction, css_class

# =============================================================================
# 회귀분석 예측 함수
# =============================================================================
def build_regression_model(df, target_col, feature_cols, train_days=365):
    """
    회귀 분석 모델 구축
    - target_col: 예측 대상 (후행지표)
    - feature_cols: 설명 변수들 (선행지표들)
    - train_days: 학습 데이터 기간
    """
    # 데이터 준비
    cutoff = df['날짜'].max() - timedelta(days=train_days)
    df_train = df[df['날짜'] >= cutoff].copy()
    
    # 결측치 제거
    cols_needed = [target_col] + feature_cols
    df_clean = df_train[cols_needed].dropna()
    
    if len(df_clean) < 30:
        return None, None, None, "데이터가 부족합니다 (최소 30개 필요)"
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    # 스케일링
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # 예측 및 평가
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # 계수 정보
    coef_info = []
    for i, col in enumerate(feature_cols):
        coef_info.append({
            'feature': col,
            'coefficient': model.coef_[i],
            'importance': abs(model.coef_[i])
        })
    coef_df = pd.DataFrame(coef_info).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'r2': r2,
        'mae': mae,
        'coefficients': coef_df,
        'y_actual': y,
        'y_pred': y_pred,
        'dates': df_train[df_train[target_col].notna()]['날짜'].iloc[-len(y):].values
    }, X, y, None

def predict_future(model_info, df, feature_cols, days_ahead=7):
    """
    미래 예측 (단순 추세 기반)
    """
    if model_info is None:
        return None
    
    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']
    
    # 최근 데이터로 예측
    latest = df[feature_cols].dropna().iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler_X.transform(latest)
    
    pred_scaled = model.predict(latest_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    
    return pred

# =============================================================================
# 기존 함수들
# =============================================================================
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
        <h1>📊 데일리 클리핑 통합 지표 대시보드 v3.0</h1>
        <p>📅 기준일: {latest_date.strftime('%Y년 %m월 %d일')} | 🆕 회귀분석 예측 기능 추가</p>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_summary(df)
    
    # =========================================================================
    # 급변동 알림 (전체 표시)
    # =========================================================================
    alerts = check_alerts(summary)
    if alerts:
        st.markdown(f'<div class="alert-box"><h4>🚨 급변동 알림 ({len(alerts)}건)</h4></div>', unsafe_allow_html=True)
        
        # 알림 전체를 스크롤 가능한 영역에 표시
        num_cols = 4
        num_rows = (len(alerts) + num_cols - 1) // num_cols  # 올림 나눗셈
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                alert_idx = row * num_cols + col_idx
                if alert_idx < len(alerts):
                    alert = alerts[alert_idx]
                    with cols[col_idx]:
                        direction = "▲" if alert['direction'] == 'up' else "▼"
                        color = "#00d26a" if alert['direction'] == 'up' else "#ff6b6b"
                        st.markdown(f"""
                        <div class="alert-item" style="border-color: {color};">
                            <div style="color: #888; font-size: 0.8rem;">{alert['icon']} {alert['category']}</div>
                            <div style="color: #fff; font-weight: bold;">{alert['indicator']}</div>
                            <div style="color: {color}; font-weight: bold;">{direction} {abs(alert['change_pct']):.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # 탭 (예측 탭 추가)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 지표 현황", "🔬 상관관계 분석", "🎯 예측 분석", "📊 차트 분석", "📋 데이터 테이블"])
    
    # =========================================================================
    # TAB 1: 지표 현황
    # =========================================================================
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
    
    # =========================================================================
    # TAB 2: 상관관계 분석
    # =========================================================================
    with tab2:
        st.markdown("## 🔬 선행/후행 지표 상관관계 분석")
        st.markdown("지표 간의 상관관계와 시차(Lag)를 분석하여 **선행지표 변화 → 후행지표 예측**에 활용합니다.")
        
        st.markdown("---")
        
        # ----- 섹션 1: 상관관계 히트맵 -----
        st.markdown("### 📊 지표 간 상관관계 매트릭스")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            heatmap_period = st.selectbox(
                "분석 기간",
                ["3개월", "6개월", "1년", "전체"],
                index=2,
                key="heatmap_period"
            )
            
            heatmap_indicators = st.multiselect(
                "분석 지표 선택",
                KEY_INDICATORS,
                default=["달러환율", "육지 SMP", "두바이유", "국고채 (3년)", "IRS (3년)"],
                key="heatmap_indicators"
            )
        
        with col2:
            if len(heatmap_indicators) >= 2:
                days = CHART_PERIODS.get(heatmap_period)
                corr_matrix = calculate_correlation_matrix(df, heatmap_indicators, days)
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    labels=dict(color="상관계수"),
                    x=heatmap_indicators,
                    y=heatmap_indicators,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    text_auto='.2f'
                )
                
                fig_heatmap.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(22,33,62,0.8)',
                    plot_bgcolor='rgba(22,33,62,0.8)',
                    height=400,
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("2개 이상의 지표를 선택해주세요.")
        
        st.markdown("---")
        
        # ----- 섹션 2: 시차(Lag) 상관관계 분석 -----
        st.markdown("### 🕐 시차(Lag) 상관관계 분석")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            leading_indicator = st.selectbox(
                "🔵 선행지표 (먼저 움직이는 지표)",
                KEY_INDICATORS,
                index=KEY_INDICATORS.index("두바이유") if "두바이유" in KEY_INDICATORS else 0,
                key="leading"
            )
        
        with col2:
            lagging_indicator = st.selectbox(
                "🔴 후행지표 (따라오는 지표)",
                KEY_INDICATORS,
                index=KEY_INDICATORS.index("육지 SMP") if "육지 SMP" in KEY_INDICATORS else 1,
                key="lagging"
            )
        
        with col3:
            max_lag = st.slider("최대 시차 (일)", 1, 60, 30, key="max_lag")
        
        if leading_indicator != lagging_indicator:
            lag_df = calculate_lagged_correlation(df, leading_indicator, lagging_indicator, max_lag)
            optimal = find_optimal_lag(lag_df)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_lag = go.Figure()
                fig_lag.add_trace(go.Scatter(
                    x=lag_df['lag'], y=lag_df['correlation'],
                    mode='lines+markers', name='상관계수',
                    line=dict(color='#3498db', width=2), marker=dict(size=6)
                ))
                
                if optimal is not None:
                    fig_lag.add_vline(x=optimal['lag'], line_dash="dash", line_color="#e94560",
                                     annotation_text=f"최적 Lag: {int(optimal['lag'])}일")
                
                fig_lag.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig_lag.update_layout(
                    title=f"{leading_indicator} → {lagging_indicator} 시차별 상관계수",
                    xaxis_title="시차 (일)", yaxis_title="상관계수",
                    template='plotly_dark',
                    paper_bgcolor='rgba(22,33,62,0.8)',
                    plot_bgcolor='rgba(22,33,62,0.8)',
                    height=350, yaxis=dict(range=[-1, 1])
                )
                st.plotly_chart(fig_lag, use_container_width=True)
            
            with col2:
                if optimal is not None and not np.isnan(optimal['correlation']):
                    strength, direction, css_class = interpret_correlation(optimal['correlation'])
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>💡 분석 결과</h4>
                        <p><strong>최적 시차:</strong> <span style="color: #e94560; font-size: 1.3rem;">{int(optimal['lag'])}일</span></p>
                        <p><strong>상관계수:</strong> <span class="{css_class}">{optimal['correlation']:.3f}</span></p>
                        <p><strong>해석:</strong> {strength} {direction} 상관관계</p>
                        <p><strong>통계적 유의성:</strong> {'✅ 유의함' if optimal['significant'] else '⚠️ 유의하지 않음'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("선행지표와 후행지표를 다르게 선택해주세요.")
    
    # =========================================================================
    # TAB 3: 예측 분석 (신규)
    # =========================================================================
    with tab3:
        st.markdown("## 🎯 회귀분석 기반 예측")
        st.markdown("선행지표들을 활용하여 후행지표의 값을 예측합니다.")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ 예측 설정")
            
            # 예측 대상 선택
            target_col = st.selectbox(
                "🎯 예측 대상 (후행지표)",
                KEY_INDICATORS,
                index=KEY_INDICATORS.index("육지 SMP") if "육지 SMP" in KEY_INDICATORS else 0,
                key="pred_target"
            )
            
            # 설명 변수 선택
            available_features = [x for x in KEY_INDICATORS if x != target_col]
            feature_cols = st.multiselect(
                "📊 설명 변수 (선행지표들)",
                available_features,
                default=["두바이유", "달러환율", "국고채 (3년)"] if all(x in available_features for x in ["두바이유", "달러환율", "국고채 (3년)"]) else available_features[:3],
                key="pred_features"
            )
            
            # 학습 기간
            train_period = st.selectbox(
                "📅 학습 데이터 기간",
                ["3개월", "6개월", "1년", "전체"],
                index=2,
                key="train_period"
            )
            
            train_days = CHART_PERIODS.get(train_period)
            
            run_prediction = st.button("🚀 예측 모델 실행", use_container_width=True)
        
        with col2:
            if run_prediction and len(feature_cols) >= 1:
                with st.spinner("모델 학습 중..."):
                    model_info, X, y, error = build_regression_model(
                        df, target_col, feature_cols, 
                        train_days if train_days else len(df)
                    )
                
                if error:
                    st.error(f"❌ {error}")
                elif model_info:
                    # 모델 성능
                    st.markdown("### 📊 모델 성능")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        r2_color = "#00d26a" if model_info['r2'] >= 0.7 else ("#f39c12" if model_info['r2'] >= 0.4 else "#ff6b6b")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">R² (설명력)</div>
                            <div class="metric-value" style="color: {r2_color};">{model_info['r2']:.3f}</div>
                            <div style="color: #888;">{'좋음' if model_info['r2'] >= 0.7 else ('보통' if model_info['r2'] >= 0.4 else '낮음')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with perf_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">MAE (평균 오차)</div>
                            <div class="metric-value">{model_info['mae']:.2f}</div>
                            <div style="color: #888;">절대 평균 오차</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with perf_col3:
                        # 현재 값 기준 예측
                        current_pred = predict_future(model_info, df, feature_cols)
                        actual_latest = df[target_col].dropna().iloc[-1]
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h4>🎯 현재 예측값</h4>
                            <p style="font-size: 1.5rem; font-weight: bold;">{current_pred:.2f}</p>
                            <p style="color: #888;">실제값: {actual_latest:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 변수 중요도
                    st.markdown("### 📈 변수 중요도")
                    coef_df = model_info['coefficients']
                    
                    fig_coef = go.Figure(go.Bar(
                        x=coef_df['importance'],
                        y=coef_df['feature'],
                        orientation='h',
                        marker_color=['#00d26a' if c > 0 else '#ff6b6b' for c in coef_df['coefficient']]
                    ))
                    fig_coef.update_layout(
                        title="변수별 영향력 (절대값)",
                        template='plotly_dark',
                        paper_bgcolor='rgba(22,33,62,0.8)',
                        plot_bgcolor='rgba(22,33,62,0.8)',
                        height=250,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)
                    
                    # 실제 vs 예측 차트
                    st.markdown("### 📉 실제값 vs 예측값")
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=model_info['dates'], y=model_info['y_actual'],
                        mode='lines', name='실제값',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=model_info['dates'], y=model_info['y_pred'],
                        mode='lines', name='예측값',
                        line=dict(color='#e94560', width=2, dash='dot')
                    ))
                    fig_pred.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(22,33,62,0.8)',
                        plot_bgcolor='rgba(22,33,62,0.8)',
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # 해석
                    st.markdown("### 💡 분석 인사이트")
                    
                    top_feature = coef_df.iloc[0]
                    direction = "양의" if top_feature['coefficient'] > 0 else "음의"
                    
                    st.info(f"""
                    **모델 해석:**
                    - **{target_col}** 예측에 가장 큰 영향을 미치는 변수는 **{top_feature['feature']}** 입니다.
                    - {top_feature['feature']}와 {target_col}은 **{direction} 관계**입니다.
                    - 모델의 설명력(R²)은 **{model_info['r2']*100:.1f}%** 입니다.
                    """)
                    
                    if model_info['r2'] < 0.4:
                        st.warning("⚠️ 모델 설명력이 낮습니다. 다른 설명 변수를 추가하거나 학습 기간을 조정해보세요.")
            
            elif run_prediction:
                st.warning("설명 변수를 1개 이상 선택해주세요.")
            else:
                st.info("👈 왼쪽에서 설정 후 '예측 모델 실행' 버튼을 클릭하세요.")
    
    # =========================================================================
    # TAB 4: 차트 분석
    # =========================================================================
    with tab4:
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
        compare_indicators = st.multiselect("비교할 지표 (최대 4개)", compare_options, default=['달러환율', '육지 SMP'], max_selections=4, key="compare")
        
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
    
    # =========================================================================
    # TAB 5: 데이터 테이블
    # =========================================================================
    with tab5:
        st.markdown("### 📋 원본 데이터 조회")
        
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("날짜 범위", value=(latest_date - timedelta(days=30), latest_date))
        with col2:
            table_category = st.selectbox("카테고리", ['전체'] + list(INDICATORS.keys()), key="table_cat")
        
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
        📊 데일리 클리핑 통합 지표 대시보드 v3.0 | 데이터 출처: 서울외국환중개, 신재생 원스톱 포털, 한국석유공사, 한국가스공사, 경제통계시스템
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
