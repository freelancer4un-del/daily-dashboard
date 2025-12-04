# =============================================================================
# app.py - 통합 지표 모니터링 대시보드 v4.0
# 친환경·순환경제·인프라 자산운용사 맞춤 버전
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
    page_title="📊 IFAM 대시보드 v4.0",
    page_icon="🌱",
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
        border: 1px solid #27ae60;
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
    .metric-card:hover { border-color: #27ae60; }
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
    
    .signal-buy {
        background: linear-gradient(145deg, #1a4a3c 0%, #16213e 100%);
        border: 2px solid #00d26a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(145deg, #4a1a1a 0%, #16213e 100%);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .signal-hold {
        background: linear-gradient(145deg, #3a3a1a 0%, #16213e 100%);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .summary-card {
        background: linear-gradient(145deg, #1a2a4a 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3498db;
        margin: 0.5rem 0;
    }
    
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
# LNG 데이터 처리 (월별 데이터 대응)
# =============================================================================
def get_latest_lng_data(df):
    """LNG는 월별 데이터이므로 가장 최근 유효값을 가져옴"""
    lng_cols = ['탱크로리용', '연료전지용']
    result = {}
    
    for col in lng_cols:
        # 유효한 값이 있는 가장 최근 행 찾기
        valid_data = df[df[col].notna()][['날짜', col]]
        if len(valid_data) > 0:
            latest = valid_data.iloc[-1]
            # 이전 값 (전월)
            if len(valid_data) > 1:
                prev = valid_data.iloc[-2]
            else:
                prev = latest
            
            result[col] = {
                'value': latest[col],
                'previous': prev[col],
                'date': latest['날짜']
            }
        else:
            result[col] = {'value': None, 'previous': None, 'date': None}
    
    return result

# =============================================================================
# 기존 함수들
# =============================================================================
def get_summary(df):
    if df is None or len(df) < 2:
        return {}
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    summary = {}
    
    # LNG 최신 데이터 가져오기
    lng_data = get_latest_lng_data(df)
    
    for category, info in INDICATORS.items():
        is_rate = category in ['금리', '스왑']
        summary[category] = {'icon': info['icon'], 'color': info['color'], 'indicators': {}}
        
        for col_name, col_info in info['columns'].items():
            # LNG는 별도 처리
            if category == 'LNG' and col_name in lng_data:
                lng_info = lng_data[col_name]
                current = lng_info['value']
                prev = lng_info['previous']
                
                if current is not None and prev is not None and prev != 0:
                    change = current - prev
                    change_pct = (change / prev) * 100
                    direction = 'up' if change > 0 else ('down' if change < 0 else 'neutral')
                else:
                    change, change_pct, direction = None, None, 'neutral'
                
                summary[category]['indicators'][col_name] = {
                    'value': current, 'previous': prev, 'change': change,
                    'change_pct': change_pct, 'direction': direction,
                    'unit': col_info['unit'], 'format': col_info['format'],
                    'note': f"({lng_info['date'].strftime('%m월') if lng_info['date'] else ''})"
                }
            else:
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
                    'unit': col_info['unit'], 'format': col_info['format'],
                    'note': ''
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

def create_metric_card(title, value, change_html, note=""):
    note_html = f'<div style="color: #666; font-size: 0.75rem;">{note}</div>' if note else ''
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div>{change_html}</div>
        {note_html}
    </div>
    """

# =============================================================================
# 상관관계 분석 함수
# =============================================================================
def calculate_correlation_matrix(df, columns, days=365):
    if days:
        cutoff = df['날짜'].max() - timedelta(days=days)
        df_filtered = df[df['날짜'] >= cutoff]
    else:
        df_filtered = df
    df_corr = df_filtered[columns].dropna()
    return df_corr.corr()

def calculate_lagged_correlation(df, leading_col, lagging_col, max_lag=30):
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
        results.append({'lag': lag, 'correlation': corr, 'p_value': p_value,
                       'significant': p_value < 0.05 if not np.isnan(p_value) else False})
    return pd.DataFrame(results)

def find_optimal_lag(lag_df):
    valid_df = lag_df.dropna()
    if len(valid_df) == 0:
        return None
    idx = valid_df['correlation'].abs().idxmax()
    return valid_df.loc[idx]

def interpret_correlation(corr):
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "강한", "양의" if corr > 0 else "음의", "correlation-strong"
    elif abs_corr >= 0.4:
        return "중간", "양의" if corr > 0 else "음의", "correlation-moderate"
    return "약한", "양의" if corr > 0 else "음의", "correlation-weak"

# =============================================================================
# 회귀분석 예측 함수
# =============================================================================
def build_regression_model(df, target_col, feature_cols, train_days=365):
    cutoff = df['날짜'].max() - timedelta(days=train_days) if train_days else df['날짜'].min()
    df_train = df[df['날짜'] >= cutoff].copy()
    
    cols_needed = [target_col] + feature_cols
    df_clean = df_train[cols_needed].dropna()
    
    if len(df_clean) < 30:
        return None, None, None, "데이터가 부족합니다 (최소 30개 필요)"
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    coef_info = [{'feature': col, 'coefficient': model.coef_[i], 'importance': abs(model.coef_[i])}
                 for i, col in enumerate(feature_cols)]
    coef_df = pd.DataFrame(coef_info).sort_values('importance', ascending=False)
    
    return {
        'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'r2': r2, 'mae': mae, 'coefficients': coef_df,
        'y_actual': y, 'y_pred': y_pred,
        'dates': df_train[df_train[target_col].notna()]['날짜'].iloc[-len(y):].values
    }, X, y, None

def predict_future(model_info, df, feature_cols):
    if model_info is None:
        return None
    latest = df[feature_cols].dropna().iloc[-1].values.reshape(1, -1)
    latest_scaled = model_info['scaler_X'].transform(latest)
    pred_scaled = model_info['model'].predict(latest_scaled)
    return model_info['scaler_y'].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

# =============================================================================
# 신재생에너지 수익성 시뮬레이터
# =============================================================================
def calculate_renewable_revenue(smp, rec_price, capacity_mw, cf=0.15, rec_weight=1.0):
    """
    신재생에너지 발전 수익 계산
    - smp: 계통한계가격 (원/kWh)
    - rec_price: REC 가격 (원/REC)
    - capacity_mw: 설비용량 (MW)
    - cf: 이용률 (Capacity Factor, 태양광 기본 15%)
    - rec_weight: REC 가중치 (태양광 기본 1.0)
    """
    # 연간 발전량 (MWh)
    annual_generation = capacity_mw * 1000 * 24 * 365 * cf / 1000  # MWh
    
    # SMP 수익
    smp_revenue = annual_generation * smp * 1000  # 원
    
    # REC 수익 (1MWh = 1REC)
    rec_count = annual_generation * rec_weight
    rec_revenue = rec_count * rec_price
    
    # 총 수익
    total_revenue = smp_revenue + rec_revenue
    
    return {
        'annual_generation_mwh': annual_generation,
        'smp_revenue': smp_revenue,
        'rec_revenue': rec_revenue,
        'total_revenue': total_revenue,
        'revenue_per_mw': total_revenue / capacity_mw if capacity_mw > 0 else 0
    }

# =============================================================================
# 투자 시그널 생성
# =============================================================================
def generate_investment_signals(df, days=30):
    """투자 의사결정 시그널 생성"""
    signals = []
    
    if len(df) < days:
        return signals
    
    latest = df.iloc[-1]
    
    # 최근 N일 데이터
    recent = df.tail(days)
    
    # 1. SMP 시그널
    smp_current = latest.get('육지 SMP')
    smp_avg = recent['육지 SMP'].mean()
    smp_std = recent['육지 SMP'].std()
    
    if pd.notna(smp_current) and pd.notna(smp_avg):
        if smp_current < smp_avg - smp_std:
            signals.append({
                'category': '신재생에너지',
                'indicator': 'SMP',
                'signal': 'BUY',
                'reason': f'SMP가 30일 평균 대비 저점 (현재: {smp_current:.1f}, 평균: {smp_avg:.1f})',
                'strength': 'STRONG' if smp_current < smp_avg - 2*smp_std else 'MODERATE'
            })
        elif smp_current > smp_avg + smp_std:
            signals.append({
                'category': '신재생에너지',
                'indicator': 'SMP',
                'signal': 'SELL',
                'reason': f'SMP가 30일 평균 대비 고점 (현재: {smp_current:.1f}, 평균: {smp_avg:.1f})',
                'strength': 'STRONG' if smp_current > smp_avg + 2*smp_std else 'MODERATE'
            })
    
    # 2. REC 시그널
    rec_current = latest.get('육지 가격')
    rec_avg = recent['육지 가격'].mean()
    rec_std = recent['육지 가격'].std()
    
    if pd.notna(rec_current) and pd.notna(rec_avg) and rec_std > 0:
        if rec_current < rec_avg - rec_std:
            signals.append({
                'category': '신재생에너지',
                'indicator': 'REC',
                'signal': 'BUY',
                'reason': f'REC 가격 저점 매수 기회 (현재: {rec_current:,.0f}, 평균: {rec_avg:,.0f})',
                'strength': 'STRONG' if rec_current < rec_avg - 2*rec_std else 'MODERATE'
            })
    
    # 3. 금리 시그널 (인프라 투자)
    rate_current = latest.get('국고채 (3년)')
    rate_avg = recent['국고채 (3년)'].mean()
    
    if pd.notna(rate_current) and pd.notna(rate_avg):
        if rate_current > rate_avg + 0.1:
            signals.append({
                'category': '인프라',
                'indicator': '금리',
                'signal': 'HOLD',
                'reason': f'금리 상승 중 - 신규 차입 주의 (현재: {rate_current:.2f}%, 평균: {rate_avg:.2f}%)',
                'strength': 'MODERATE'
            })
        elif rate_current < rate_avg - 0.1:
            signals.append({
                'category': '인프라',
                'indicator': '금리',
                'signal': 'BUY',
                'reason': f'금리 하락 - 차입 적기 (현재: {rate_current:.2f}%, 평균: {rate_avg:.2f}%)',
                'strength': 'MODERATE'
            })
    
    # 4. 환율 시그널 (해외 투자)
    fx_current = latest.get('달러환율')
    fx_avg = recent['달러환율'].mean()
    fx_std = recent['달러환율'].std()
    
    if pd.notna(fx_current) and pd.notna(fx_avg) and fx_std > 0:
        if fx_current > fx_avg + fx_std:
            signals.append({
                'category': '해외투자',
                'indicator': '환율',
                'signal': 'HOLD',
                'reason': f'원화 약세 - 해외 신규 투자 주의 (현재: {fx_current:,.0f}원)',
                'strength': 'MODERATE'
            })
        elif fx_current < fx_avg - fx_std:
            signals.append({
                'category': '해외투자',
                'indicator': '환율',
                'signal': 'BUY',
                'reason': f'원화 강세 - 해외 투자 적기 (현재: {fx_current:,.0f}원)',
                'strength': 'MODERATE'
            })
    
    return signals

# =============================================================================
# 시장 트렌드 요약
# =============================================================================
def generate_market_summary(df, days=7):
    """주간 시장 트렌드 요약"""
    if len(df) < days:
        return None
    
    recent = df.tail(days)
    prev_period = df.iloc[-(days*2):-days] if len(df) >= days*2 else df.head(days)
    
    summary = {}
    
    indicators = {
        '달러환율': {'name': '달러/원 환율', 'unit': '원', 'format': '{:,.1f}'},
        '육지 SMP': {'name': 'SMP (육지)', 'unit': '원/kWh', 'format': '{:,.1f}'},
        '육지 가격': {'name': 'REC 가격', 'unit': '원', 'format': '{:,.0f}'},
        '두바이유': {'name': '두바이유', 'unit': '$/배럴', 'format': '{:,.1f}'},
        '국고채 (3년)': {'name': '국고채 3년', 'unit': '%', 'format': '{:,.2f}'},
    }
    
    for col, info in indicators.items():
        current_avg = recent[col].mean()
        prev_avg = prev_period[col].mean()
        current_last = recent[col].iloc[-1]
        
        if pd.notna(current_avg) and pd.notna(prev_avg) and prev_avg != 0:
            change_pct = (current_avg - prev_avg) / prev_avg * 100
            trend = '상승' if change_pct > 0.5 else ('하락' if change_pct < -0.5 else '보합')
            
            summary[col] = {
                'name': info['name'],
                'current': current_last,
                'avg': current_avg,
                'prev_avg': prev_avg,
                'change_pct': change_pct,
                'trend': trend,
                'unit': info['unit'],
                'format': info['format']
            }
    
    return summary

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
        <h1>🌱 IFAM 대시보드 v4.0</h1>
        <p>📅 기준일: {latest_date.strftime('%Y년 %m월 %d일')} | 인프라프론티어자산운용(주)</p>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_summary(df)
    
    # 급변동 알림
    alerts = check_alerts(summary)
    if alerts:
        st.markdown(f'<div class="alert-box"><h4>🚨 급변동 알림 ({len(alerts)}건)</h4></div>', unsafe_allow_html=True)
        num_cols = 4
        num_rows = (len(alerts) + num_cols - 1) // num_cols
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
    
    # 탭
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 지표 현황", "🌱 수익성 시뮬레이터", "🔔 투자 시그널",
        "🔬 상관관계 분석", "🎯 예측 분석", "📋 데이터"
    ])
    
    # =========================================================================
    # TAB 1: 지표 현황
    # =========================================================================
    with tab1:
        # 시장 트렌드 요약
        st.markdown("### 📊 주간 시장 트렌드")
        market_summary = generate_market_summary(df, days=7)
        
        if market_summary:
            cols = st.columns(5)
            for i, (col_name, data) in enumerate(market_summary.items()):
                with cols[i % 5]:
                    trend_color = "#00d26a" if data['trend'] == '상승' else ("#ff6b6b" if data['trend'] == '하락' else "#888")
                    trend_arrow = "↑" if data['trend'] == '상승' else ("↓" if data['trend'] == '하락' else "→")
                    st.markdown(f"""
                    <div class="summary-card">
                        <div style="color: #888; font-size: 0.8rem;">{data['name']}</div>
                        <div style="color: #fff; font-size: 1.3rem; font-weight: bold;">{data['format'].format(data['current'])} {data['unit']}</div>
                        <div style="color: {trend_color};">{trend_arrow} {data['trend']} ({data['change_pct']:+.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 카테고리별 지표
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
                    note = ind.get('note', '')
                    st.markdown(create_metric_card(col_name, value_str, change_html, note), unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: 수익성 시뮬레이터
    # =========================================================================
    with tab2:
        st.markdown("## 🌱 신재생에너지 수익성 시뮬레이터")
        st.markdown("SMP와 REC 가격 시나리오별 예상 수익을 계산합니다.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ 프로젝트 설정")
            
            project_type = st.selectbox("발전 유형", ["태양광", "풍력(육상)", "풍력(해상)", "연료전지", "바이오"])
            
            # 유형별 기본값
            defaults = {
                "태양광": {"cf": 0.15, "rec_weight": 1.0},
                "풍력(육상)": {"cf": 0.25, "rec_weight": 1.0},
                "풍력(해상)": {"cf": 0.30, "rec_weight": 2.0},
                "연료전지": {"cf": 0.85, "rec_weight": 2.0},
                "바이오": {"cf": 0.80, "rec_weight": 1.5},
            }
            
            capacity = st.number_input("설비용량 (MW)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
            cf = st.slider("이용률 (%)", 5, 95, int(defaults[project_type]["cf"]*100)) / 100
            rec_weight = st.number_input("REC 가중치", min_value=0.5, max_value=5.0, 
                                         value=defaults[project_type]["rec_weight"], step=0.1)
            
            st.markdown("### 📊 시나리오 설정")
            
            # 현재 값 가져오기
            current_smp = df['육지 SMP'].dropna().iloc[-1] if len(df['육지 SMP'].dropna()) > 0 else 100
            current_rec = df['육지 가격'].dropna().iloc[-1] if len(df['육지 가격'].dropna()) > 0 else 70000
            
            smp_scenarios = st.multiselect(
                "SMP 시나리오 (원/kWh)",
                [80, 100, 120, 150, 180, 200, 220],
                default=[100, 150, 200]
            )
            
            rec_scenario = st.number_input("REC 가격 (원/REC)", 
                                           min_value=10000, max_value=200000, 
                                           value=int(current_rec), step=1000)
        
        with col2:
            st.markdown("### 📈 수익 시뮬레이션 결과")
            
            if smp_scenarios:
                results = []
                for smp in smp_scenarios:
                    rev = calculate_renewable_revenue(smp, rec_scenario, capacity, cf, rec_weight)
                    results.append({
                        'SMP (원/kWh)': smp,
                        '연간발전량 (MWh)': f"{rev['annual_generation_mwh']:,.0f}",
                        'SMP 수익 (억원)': f"{rev['smp_revenue']/100000000:.2f}",
                        'REC 수익 (억원)': f"{rev['rec_revenue']/100000000:.2f}",
                        '총 수익 (억원)': f"{rev['total_revenue']/100000000:.2f}",
                        'MW당 수익 (억원)': f"{rev['revenue_per_mw']/100000000:.2f}"
                    })
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # 차트
                fig = go.Figure()
                
                revenues = [calculate_renewable_revenue(smp, rec_scenario, capacity, cf, rec_weight)['total_revenue']/100000000 
                           for smp in smp_scenarios]
                
                fig.add_trace(go.Bar(
                    x=[f"SMP {s}" for s in smp_scenarios],
                    y=revenues,
                    marker_color='#27ae60',
                    text=[f"{r:.1f}억" for r in revenues],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"{project_type} {capacity}MW 연간 예상 수익",
                    yaxis_title="총 수익 (억원)",
                    template='plotly_dark',
                    paper_bgcolor='rgba(22,33,62,0.8)',
                    plot_bgcolor='rgba(22,33,62,0.8)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 손익분기점 분석
                st.markdown("### 💰 손익분기점 분석")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    capex_per_mw = st.number_input("CAPEX (억원/MW)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
                with col_b:
                    opex_rate = st.slider("OPEX (수익 대비 %)", 5, 30, 15)
                
                total_capex = capex_per_mw * capacity
                current_rev = calculate_renewable_revenue(current_smp, rec_scenario, capacity, cf, rec_weight)
                annual_opex = current_rev['total_revenue'] * opex_rate / 100
                net_revenue = current_rev['total_revenue'] - annual_opex
                
                if net_revenue > 0:
                    payback_years = total_capex * 100000000 / net_revenue
                    st.success(f"📊 **현재 SMP({current_smp:.0f}원) 기준 투자회수 기간: {payback_years:.1f}년**")
                else:
                    st.error("현재 조건에서는 수익이 발생하지 않습니다.")
    
    # =========================================================================
    # TAB 3: 투자 시그널
    # =========================================================================
    with tab3:
        st.markdown("## 🔔 투자 의사결정 시그널")
        st.markdown("시장 지표 분석을 통한 투자 타이밍 시그널입니다.")
        
        signals = generate_investment_signals(df, days=30)
        
        if signals:
            for signal in signals:
                if signal['signal'] == 'BUY':
                    css_class = 'signal-buy'
                    icon = '🟢'
                    label = '매수 적기'
                elif signal['signal'] == 'SELL':
                    css_class = 'signal-sell'
                    icon = '🔴'
                    label = '매도 고려'
                else:
                    css_class = 'signal-hold'
                    icon = '🟡'
                    label = '관망'
                
                st.markdown(f"""
                <div class="{css_class}">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="color: #fff; font-size: 1.2rem; font-weight: bold;">{signal['category']} - {signal['indicator']}</div>
                    <div style="color: #fff; font-size: 1.5rem; font-weight: bold;">{label}</div>
                    <div style="color: #aaa; margin-top: 0.5rem;">{signal['reason']}</div>
                    <div style="color: #888; font-size: 0.8rem;">신호 강도: {signal['strength']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("현재 특별한 투자 시그널이 없습니다. 시장이 안정적입니다.")
        
        # 종합 분석
        st.markdown("---")
        st.markdown("### 📋 종합 시장 분석")
        
        latest = df.iloc[-1]
        
        analysis_points = []
        
        # SMP 분석
        smp_current = latest.get('육지 SMP')
        smp_avg_90d = df.tail(90)['육지 SMP'].mean()
        if pd.notna(smp_current) and pd.notna(smp_avg_90d):
            smp_vs_avg = (smp_current / smp_avg_90d - 1) * 100
            if smp_vs_avg > 10:
                analysis_points.append(f"⚡ SMP가 90일 평균 대비 **{smp_vs_avg:.1f}% 높음** - 신재생 발전 수익성 양호")
            elif smp_vs_avg < -10:
                analysis_points.append(f"⚡ SMP가 90일 평균 대비 **{abs(smp_vs_avg):.1f}% 낮음** - 수익성 주의 필요")
        
        # 금리 분석
        rate_current = latest.get('국고채 (3년)')
        rate_avg_90d = df.tail(90)['국고채 (3년)'].mean()
        if pd.notna(rate_current) and pd.notna(rate_avg_90d):
            if rate_current > rate_avg_90d + 0.2:
                analysis_points.append(f"📊 금리 상승 추세 (현재 {rate_current:.2f}%) - 신규 PF 조달비용 상승 예상")
            elif rate_current < rate_avg_90d - 0.2:
                analysis_points.append(f"📊 금리 하락 추세 (현재 {rate_current:.2f}%) - PF 리파이낸싱 검토 적기")
        
        # 유가 분석
        oil_current = latest.get('두바이유')
        oil_avg_90d = df.tail(90)['두바이유'].mean()
        if pd.notna(oil_current) and pd.notna(oil_avg_90d):
            oil_vs_avg = (oil_current / oil_avg_90d - 1) * 100
            if oil_vs_avg > 15:
                analysis_points.append(f"🛢️ 유가 상승 추세 - SMP 상승 가능성, 연료전지 발전 비용 증가 예상")
            elif oil_vs_avg < -15:
                analysis_points.append(f"🛢️ 유가 하락 추세 - SMP 하락 가능성 주의")
        
        if analysis_points:
            for point in analysis_points:
                st.markdown(f"- {point}")
        else:
            st.info("시장이 전반적으로 안정적입니다.")
    
    # =========================================================================
    # TAB 4: 상관관계 분석
    # =========================================================================
    with tab4:
        st.markdown("## 🔬 선행/후행 지표 상관관계 분석")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            heatmap_period = st.selectbox("분석 기간", ["3개월", "6개월", "1년", "전체"], index=2, key="hm_period")
            heatmap_indicators = st.multiselect(
                "분석 지표",
                KEY_INDICATORS,
                default=["달러환율", "육지 SMP", "두바이유", "국고채 (3년)"],
                key="hm_ind"
            )
        
        with col2:
            if len(heatmap_indicators) >= 2:
                days = CHART_PERIODS.get(heatmap_period)
                corr_matrix = calculate_correlation_matrix(df, heatmap_indicators, days)
                
                fig_heatmap = px.imshow(
                    corr_matrix, labels=dict(color="상관계수"),
                    x=heatmap_indicators, y=heatmap_indicators,
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1, text_auto='.2f'
                )
                fig_heatmap.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(22,33,62,0.8)',
                    plot_bgcolor='rgba(22,33,62,0.8)',
                    height=400
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🕐 시차(Lag) 분석")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            leading = st.selectbox("선행지표", KEY_INDICATORS, index=5, key="lead")
        with col2:
            lagging = st.selectbox("후행지표", KEY_INDICATORS, index=3, key="lag")
        with col3:
            max_lag = st.slider("최대 시차", 1, 60, 30, key="mlag")
        
        if leading != lagging:
            lag_df = calculate_lagged_correlation(df, leading, lagging, max_lag)
            optimal = find_optimal_lag(lag_df)
            
            fig_lag = go.Figure()
            fig_lag.add_trace(go.Scatter(x=lag_df['lag'], y=lag_df['correlation'],
                                        mode='lines+markers', line=dict(color='#3498db')))
            if optimal is not None:
                fig_lag.add_vline(x=optimal['lag'], line_dash="dash", line_color="#e94560")
            fig_lag.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_lag.update_layout(
                title=f"{leading} → {lagging}",
                template='plotly_dark',
                paper_bgcolor='rgba(22,33,62,0.8)',
                plot_bgcolor='rgba(22,33,62,0.8)',
                height=300, yaxis=dict(range=[-1, 1])
            )
            st.plotly_chart(fig_lag, use_container_width=True)
            
            if optimal is not None and not np.isnan(optimal['correlation']):
                strength, direction, _ = interpret_correlation(optimal['correlation'])
                st.info(f"📌 최적 시차: **{int(optimal['lag'])}일** | 상관계수: **{optimal['correlation']:.3f}** ({strength} {direction} 상관관계)")
    
    # =========================================================================
    # TAB 5: 예측 분석
    # =========================================================================
    with tab5:
        st.markdown("## 🎯 회귀분석 기반 예측")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target = st.selectbox("예측 대상", KEY_INDICATORS, index=3, key="pred_t")
            features = st.multiselect(
                "설명 변수",
                [x for x in KEY_INDICATORS if x != target],
                default=["두바이유", "달러환율"],
                key="pred_f"
            )
            train_period = st.selectbox("학습 기간", ["3개월", "6개월", "1년"], index=2, key="train_p")
            run_pred = st.button("🚀 예측 실행", use_container_width=True)
        
        with col2:
            if run_pred and features:
                train_days = CHART_PERIODS.get(train_period)
                model_info, _, _, error = build_regression_model(df, target, features, train_days)
                
                if error:
                    st.error(error)
                elif model_info:
                    st.markdown(f"**R² (설명력): {model_info['r2']:.3f}** | MAE: {model_info['mae']:.2f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=model_info['dates'], y=model_info['y_actual'],
                                            mode='lines', name='실제값', line=dict(color='#3498db')))
                    fig.add_trace(go.Scatter(x=model_info['dates'], y=model_info['y_pred'],
                                            mode='lines', name='예측값', line=dict(color='#e94560', dash='dot')))
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(22,33,62,0.8)',
                        plot_bgcolor='rgba(22,33,62,0.8)',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    pred = predict_future(model_info, df, features)
                    actual = df[target].dropna().iloc[-1]
                    st.success(f"**현재 예측값: {pred:.2f}** (실제: {actual:.2f})")
            elif run_pred:
                st.warning("설명 변수를 선택하세요.")
    
    # =========================================================================
    # TAB 6: 데이터
    # =========================================================================
    with tab6:
        st.markdown("### 📋 원본 데이터")
        
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("날짜 범위", value=(latest_date - timedelta(days=30), latest_date))
        with col2:
            table_cat = st.selectbox("카테고리", ['전체'] + list(INDICATORS.keys()), key="tbl_cat")
        
        df_table = df.copy()
        if len(date_range) == 2:
            start, end = date_range
            df_table = df_table[(df_table['날짜'] >= pd.to_datetime(start)) & (df_table['날짜'] <= pd.to_datetime(end))]
        
        if table_cat != '전체':
            cols = ['날짜'] + list(INDICATORS[table_cat]['columns'].keys())
            df_table = df_table[cols]
        
        df_display = df_table.copy()
        df_display['날짜'] = df_display['날짜'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display.sort_values('날짜', ascending=False), use_container_width=True, height=400)
        
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 CSV 다운로드", csv, f"data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🌱 IFAM 대시보드 v4.0 | 신재생에너지·순환경제·금융지표 대쉬보드
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
