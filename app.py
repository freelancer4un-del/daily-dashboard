# =============================================================================
# app.py - 통합 지표 모니터링 대시보드 v5.0
# 친환경·순환경제·인프라 자산운용사 맞춤 버전 + 사용 메뉴얼
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
    page_title="📊 친환경·인프라 투자 대시보드 v5.0",
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
    
    .manual-section {
        background: linear-gradient(145deg, #1a2a3a 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3498db;
        margin: 1rem 0;
    }
    .manual-section h4 { color: #3498db; margin: 0 0 1rem 0; }
    
    .example-box {
        background: rgba(39, 174, 96, 0.1);
        border-left: 4px solid #27ae60;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .tip-box {
        background: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
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
        
        key_cols = ['달러환율', '육지 SMP', '두바이유']
        mask = df[key_cols].notna().any(axis=1)
        df = df[mask].reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return None

# =============================================================================
# LNG 데이터 처리 (월별 데이터 - 전월 대비 등락률)
# =============================================================================
def get_latest_lng_data(df):
    lng_cols = ['탱크로리용', '연료전지용']
    result = {}
    
    for col in lng_cols:
        valid_data = df[df[col].notna()][['날짜', col]].copy()
        if len(valid_data) > 0:
            # 월별로 그룹화하여 각 월의 마지막 값 가져오기
            valid_data['년월'] = valid_data['날짜'].dt.to_period('M')
            monthly_data = valid_data.groupby('년월').last().reset_index()
            
            if len(monthly_data) >= 2:
                latest = monthly_data.iloc[-1]
                prev = monthly_data.iloc[-2]
                
                # 전월 대비 등락 (원 단위 차이, Daily 탭과 동일하게)
                change = latest[col] - prev[col]
                
                result[col] = {
                    'value': latest[col], 
                    'previous': prev[col], 
                    'change': change,
                    'date': latest['날짜'],
                    'prev_month': str(prev['년월']),
                    'curr_month': str(latest['년월'])
                }
            elif len(monthly_data) == 1:
                latest = monthly_data.iloc[-1]
                result[col] = {
                    'value': latest[col], 
                    'previous': None, 
                    'change': None,
                    'date': latest['날짜'],
                    'prev_month': None,
                    'curr_month': str(latest['년월'])
                }
            else:
                result[col] = {'value': None, 'previous': None, 'change': None, 'date': None, 'prev_month': None, 'curr_month': None}
        else:
            result[col] = {'value': None, 'previous': None, 'change': None, 'date': None, 'prev_month': None, 'curr_month': None}
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
    lng_data = get_latest_lng_data(df)
    
    for category, info in INDICATORS.items():
        is_rate = category in ['금리', '스왑']
        summary[category] = {'icon': info['icon'], 'color': info['color'], 'indicators': {}}
        
        for col_name, col_info in info['columns'].items():
            if category == 'LNG' and col_name in lng_data:
                lng_info = lng_data[col_name]
                current = lng_info['value']
                prev = lng_info['previous']
                change = lng_info['change']
                
                if change is not None:
                    direction = 'up' if change > 0 else ('down' if change < 0 else 'neutral')
                    # LNG는 전월 대비 원 단위 차이로 표시 (Daily 탭과 동일)
                    change_pct = change  # 원 단위 차이를 그대로 사용
                else:
                    direction = 'neutral'
                    change_pct = None
                
                # 월 표시 (예: "10월→11월")
                if lng_info.get('prev_month') and lng_info.get('curr_month'):
                    prev_m = lng_info['prev_month'].split('-')[1] if '-' in str(lng_info['prev_month']) else ''
                    curr_m = lng_info['curr_month'].split('-')[1] if '-' in str(lng_info['curr_month']) else ''
                    note = f"({prev_m}월→{curr_m}월)"
                else:
                    note = ""
                
                summary[category]['indicators'][col_name] = {
                    'value': current, 'previous': prev, 'change': change,
                    'change_pct': change_pct, 'direction': direction,
                    'unit': col_info['unit'], 'format': col_info['format'],
                    'note': note, 'is_lng': True
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
            
            # 금리/스왑은 bp 기준, 나머지는 % 기준
            check_val = abs(ind['change']) * 100 if is_rate else abs(ind['change_pct'])
            threshold_val = threshold * 100 if is_rate else threshold
            
            if check_val >= threshold_val:
                alerts.append({
                    'category': category,
                    'indicator': col_name,
                    'change_pct': ind['change_pct'],
                    'direction': ind['direction'],
                    'icon': data['icon'],
                    # 🔽 여기 추가된 부분들 때문에 전일/현재 값 표시 가능
                    'current': ind.get('value'),
                    'previous': ind.get('previous'),
                    'fmt': ind.get('format', '{:,.2f}'),
                    'unit': ind.get('unit', '')
                })
    return alerts


def format_value(value, fmt, unit=""):
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{fmt.format(value)} {unit}"
    except:
        return str(value)

def get_change_html(change, change_pct, direction, is_rate=False, is_lng=False):
    if change is None:
        return '<span class="metric-change-neutral">-</span>'
    
    arrow = "▲" if direction == 'up' else ("▼" if direction == 'down' else "―")
    css = "metric-change-up" if direction == 'up' else ("metric-change-down" if direction == 'down' else "metric-change-neutral")
    
    if is_rate:
        return f'<span class="{css}">{arrow} {abs(change)*100:.1f}bp</span>'
    elif is_lng:
        # LNG는 원 단위 차이로 표시 (Daily 탭과 동일)
        return f'<span class="{css}">{arrow} {abs(change):.2f}</span>'
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
# 상관관계/회귀분석 함수
# =============================================================================
def calculate_correlation_matrix(df, columns, days=365):
    if days:
        cutoff = df['날짜'].max() - timedelta(days=days)
        df_filtered = df[df['날짜'] >= cutoff]
    else:
        df_filtered = df
    return df_filtered[columns].dropna().corr()

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

def build_regression_model(df, target_col, feature_cols, train_days=365):
    cutoff = df['날짜'].max() - timedelta(days=train_days) if train_days else df['날짜'].min()
    df_train = df[df['날짜'] >= cutoff].copy()
    
    cols_needed = [target_col] + feature_cols
    df_clean = df_train[cols_needed].dropna()
    
    if len(df_clean) < 30:
        return None, None, None, "데이터가 부족합니다"
    
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
    annual_generation = capacity_mw * 1000 * 24 * 365 * cf / 1000
    smp_revenue = annual_generation * smp * 1000
    rec_count = annual_generation * rec_weight
    rec_revenue = rec_count * rec_price
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
    signals = []
    if len(df) < days:
        return signals
    
    latest = df.iloc[-1]
    recent = df.tail(days)
    
    # SMP 시그널
    smp_current = latest.get('육지 SMP')
    smp_avg = recent['육지 SMP'].mean()
    smp_std = recent['육지 SMP'].std()
    
    if pd.notna(smp_current) and pd.notna(smp_avg):
        if smp_current < smp_avg - smp_std:
            signals.append({
                'category': '신재생에너지', 'indicator': 'SMP', 'signal': 'BUY',
                'reason': f'SMP가 30일 평균 대비 저점 (현재: {smp_current:.1f}, 평균: {smp_avg:.1f})',
                'strength': 'STRONG' if smp_current < smp_avg - 2*smp_std else 'MODERATE'
            })
        elif smp_current > smp_avg + smp_std:
            signals.append({
                'category': '신재생에너지', 'indicator': 'SMP', 'signal': 'SELL',
                'reason': f'SMP가 30일 평균 대비 고점 (현재: {smp_current:.1f}, 평균: {smp_avg:.1f})',
                'strength': 'STRONG' if smp_current > smp_avg + 2*smp_std else 'MODERATE'
            })
    
    # REC 시그널
    rec_current = latest.get('육지 가격')
    rec_avg = recent['육지 가격'].mean()
    rec_std = recent['육지 가격'].std()
    
    if pd.notna(rec_current) and pd.notna(rec_avg) and rec_std > 0:
        if rec_current < rec_avg - rec_std:
            signals.append({
                'category': '신재생에너지', 'indicator': 'REC', 'signal': 'BUY',
                'reason': f'REC 가격 저점 매수 기회 (현재: {rec_current:,.0f}, 평균: {rec_avg:,.0f})',
                'strength': 'STRONG' if rec_current < rec_avg - 2*rec_std else 'MODERATE'
            })
    
    # 금리 시그널
    rate_current = latest.get('국고채 (3년)')
    rate_avg = recent['국고채 (3년)'].mean()
    
    if pd.notna(rate_current) and pd.notna(rate_avg):
        if rate_current > rate_avg + 0.1:
            signals.append({
                'category': '인프라', 'indicator': '금리', 'signal': 'HOLD',
                'reason': f'금리 상승 중 - 신규 차입 주의 (현재: {rate_current:.2f}%, 평균: {rate_avg:.2f}%)',
                'strength': 'MODERATE'
            })
        elif rate_current < rate_avg - 0.1:
            signals.append({
                'category': '인프라', 'indicator': '금리', 'signal': 'BUY',
                'reason': f'금리 하락 - 차입 적기 (현재: {rate_current:.2f}%, 평균: {rate_avg:.2f}%)',
                'strength': 'MODERATE'
            })
    
    # 환율 시그널
    fx_current = latest.get('달러환율')
    fx_avg = recent['달러환율'].mean()
    fx_std = recent['달러환율'].std()
    
    if pd.notna(fx_current) and pd.notna(fx_avg) and fx_std > 0:
        if fx_current > fx_avg + fx_std:
            signals.append({
                'category': '해외투자', 'indicator': '환율', 'signal': 'HOLD',
                'reason': f'원화 약세 - 해외 신규 투자 주의 (현재: {fx_current:,.0f}원)',
                'strength': 'MODERATE'
            })
        elif fx_current < fx_avg - fx_std:
            signals.append({
                'category': '해외투자', 'indicator': '환율', 'signal': 'BUY',
                'reason': f'원화 강세 - 해외 투자 적기 (현재: {fx_current:,.0f}원)',
                'strength': 'MODERATE'
            })
    
    return signals

# =============================================================================
# 시장 트렌드 요약
# =============================================================================
def generate_market_summary(df, days=7):
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
                'name': info['name'], 'current': current_last, 'avg': current_avg,
                'prev_avg': prev_avg, 'change_pct': change_pct, 'trend': trend,
                'unit': info['unit'], 'format': info['format']
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
        - **기준 날짜:** {latest_date.strftime('%Y-%m-%d')}
        - **총 데이터:** {len(df):,}행
        - **버전:** v5.0
        """)
    
    # 메인 헤더 (기준일 + 오늘 날짜)
    today = datetime.now()
    st.markdown(f"""
    <div class="main-header">
        <h1>🌱 친환경·인프라 투자 대시보드 v5.0</h1>
        <p>📅 기준일: {latest_date.strftime('%Y년 %m월 %d일')} | 🗓️ 오늘: {today.strftime('%Y년 %m월 %d일')} | 인프라프론티어자산운용(주) </p>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_summary(df)
    
   
   # 급변동 알림
    alerts = check_alerts(summary)
    if alerts:
        st.markdown(
            f'<div class="alert-box"><h4>🚨 급변동 알림 ({len(alerts)}건) - 기준일 대비</h4></div>',
            unsafe_allow_html=True
        )
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

                        prev_str = format_value(
                            alert.get('previous'),
                            alert.get('fmt', '{:,.2f}'),
                            alert.get('unit', '')
                        )
                        curr_str = format_value(
                            alert.get('current'),
                            alert.get('fmt', '{:,.2f}'),
                            alert.get('unit', '')
                        )

                        st.markdown(f"""
                        <div class="alert-item" style="border-color: {color};">
                            <div style="color: #888; font-size: 0.8rem;">
                                {alert['icon']} {alert['category']}
                            </div>
                            <div style="color: #fff; font-weight: bold; margin-top: 2px;">
                                {alert['indicator']}
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top: 6px;">
                                <div style="color: {color}; font-weight: bold; font-size: 0.95rem;">
                                    {direction} {abs(alert['change_pct']):.2f}%
                                </div>
                                <div style="text-align: right; font-size: 0.75rem; line-height: 1.3;">
                                    <div style="color:#aaaaaa;">전일: <span style="color:#ffffff;">{prev_str}</span></div>
                                    <div style="color:#aaaaaa;">현재: <span style="color:#ffffff;">{curr_str}</span></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


    
    # 탭 (메뉴얼 탭 맨 앞에 추가)
    tab0, tab1, tab4, tab5, tab6, tab2, tab3 = st.tabs([
    "📖 사용 메뉴얼",     # tab0
    "📈 지표 현황",       # tab1
    "🔬 상관관계 분석",   # tab4
    "🎯 예측 분석",       # tab5
    "📋 데이터",          # tab6
    "🌱 시뮬레이션(미완성)", # tab2  (끝에서 두 번째)
    "🔔 투자 시그널(미완성)"       # tab3  (맨 끝)
])

    
    # =========================================================================
    # TAB 0: 사용 메뉴얼
    # =========================================================================
    with tab0:
        st.markdown("## 📖 대시보드 사용 메뉴얼")
        st.markdown("친환경·순환경제·인프라 자산운용사를 위한 통합 지표 모니터링 대시보드입니다.")
        
        st.markdown("---")
        
        # 1. 개요
        st.markdown("### 1️⃣ 대시보드 개요")
        st.markdown("""
        <div class="manual-section">
        <h4>📊 데이터 소스 및 업데이트</h4>
        <p>• <strong>데이터 출처:</strong> 데일리 클리핑 자료 (경영지원팀 제공)</p>
        <p>• <strong>지표 수:</strong> 30개 (환율, REC, SMP, 유가, LNG, 금리, 스왑)</p>
        <p>• <strong>데이터 기간:</strong> 2021년 4월 ~ 현재</p>
        <p>• <strong>업데이트:</strong> 매 영업일 (데일리 클리핑 자료 업데이트 시)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="manual-section">
        <h4>🚨 급변동 알림 기준</h4>
        <p>상단의 급변동 알림은 <strong>전일(직전 거래일) 대비</strong> 변동률이 임계값을 초과한 지표를 표시합니다.</p>
        <table style="color: #fff; width: 100%;">
        <tr><th style="text-align:left;">카테고리</th><th style="text-align:left;">임계값</th><th style="text-align:left;">예시</th></tr>
        <tr><td>환율</td><td>±1.0%</td><td>달러 1,400원 → 1,414원 (1% 상승)</td></tr>
        <tr><td>REC</td><td>±3.0%</td><td>육지 가격 70,000원 → 72,100원</td></tr>
        <tr><td>SMP</td><td>±5.0%</td><td>육지 SMP 100원 → 105원</td></tr>
        <tr><td>유가</td><td>±3.0%</td><td>두바이유 $80 → $82.4</td></tr>
        <tr><td>LNG</td><td>±5.0%</td><td>탱크로리용 15원 → 15.75원</td></tr>
        <tr><td>금리/스왑</td><td>±10bp</td><td>국고채 3.0% → 3.1%</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. 지표 현황 탭
        st.markdown("### 2️⃣ 📈 지표 현황 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p>• <strong>주간 시장 트렌드:</strong> 최근 7일간 핵심 5개 지표의 평균 변동률</p>
        <p>• <strong>카테고리별 지표:</strong> 7개 카테고리(환율, REC, SMP, 유가, LNG, 금리, 스왑)의 현재 값과 전일 대비 변동</p>
        <p>• <strong>LNG 참고:</strong> LNG는 월별 데이터로, 가장 최근 유효값을 표시합니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <strong>💼 활용 예시: 아침 투자회의</strong><br><br>
        "오늘 지표 현황 보니까 SMP가 전일 대비 16% 급등했네요. 유가도 상승 추세고, 
        우리 바이오매스 발전소 수익성이 단기적으로 좋아질 것 같습니다. 
        다만 금리도 11bp 올랐으니 신규 PF 조달 시점은 재검토가 필요해 보입니다."
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
        <strong>💡 활용 팁</strong><br>
        • 매일 아침 회의 전 주간 트렌드를 먼저 확인하세요<br>
        • 급변동 알림이 있으면 해당 지표가 포트폴리오에 미치는 영향을 즉시 점검하세요<br>
        • 사이드바에서 관심 카테고리만 필터링할 수 있습니다
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 3. 상관관계 분석 탭
        st.markdown("### 3️⃣ 🔬 상관관계 분석 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p><strong>1. 상관관계 매트릭스:</strong></p>
        <p>• 선택한 지표들 간의 상관계수를 히트맵으로 표시</p>
        <p>• 빨간색: 양의 상관관계 / 파란색: 음의 상관관계</p>
        <p>• 색이 진할수록 상관관계가 강함 (±0.7 이상: 강함, ±0.4~0.7: 중간)</p>
        <br>
        <p><strong>2. 시차(Lag) 분석:</strong></p>
        <p>• 선행지표가 며칠 후에 후행지표에 영향을 미치는지 분석</p>
        <p>• 예: "유가가 3일 전에 움직이면 SMP가 따라서 움직인다"</p>
        <p>• 최적 시차와 상관계수를 자동으로 계산</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <strong>💼 활용 예시: 시장 예측</strong><br><br>
        <strong>분석 결과:</strong> 두바이유 → 육지 SMP, 최적 시차 3일, 상관계수 0.72<br><br>
        <strong>해석:</strong> "두바이유가 상승하면 3일 후 SMP도 상승하는 경향이 있습니다 (강한 양의 상관관계).
        오늘 두바이유가 5% 급등했으니, 3일 후 SMP 상승을 예상하고 
        현물 전력 판매 계약 협상을 서두르는 게 좋겠습니다."
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
        <strong>💡 활용 팁</strong><br>
        • 신재생 투자자가 가장 많이 보는 조합: 두바이유 → SMP, 환율 → SMP<br>
        • 상관계수 0.7 이상이면 예측에 활용 가치가 높습니다<br>
        • 시차가 0일이면 동시에 움직이는 것으로, 예측보다는 확인용입니다
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 4. 예측 분석 탭
        st.markdown("### 4️⃣ 🎯 예측 분석 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p>• <strong>회귀분석:</strong> 선행지표들을 이용해 후행지표 값을 예측하는 모델</p>
        <p>• <strong>R² (설명력):</strong> 모델이 실제 데이터를 얼마나 잘 설명하는지 (0~1, 높을수록 좋음)</p>
        <p>• <strong>MAE (평균 오차):</strong> 예측값과 실제값의 평균적인 차이</p>
        <p>• <strong>변수 중요도:</strong> 어떤 설명 변수가 예측에 가장 큰 영향을 미치는지</p>
        <br>
        <p><strong>권장 조합:</strong></p>
        <p>• SMP 예측: 두바이유 + 달러환율 + 국고채 → R² 0.6~0.7 기대</p>
        <p>• 국고채 예측: IRS + 달러환율 → R² 0.8 이상 기대</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <strong>💼 활용 예시: 수익 전망 보고</strong><br><br>
        <strong>분석:</strong> SMP 예측 모델 (설명변수: 두바이유, 달러환율)<br>
        <strong>결과:</strong> R² = 0.68, 현재 예측값 102.5원/kWh (실제 98.8원/kWh)<br><br>
        <strong>보고:</strong> "회귀모델 기준 SMP가 현재 저평가 상태입니다. 
        모델 예측값(102.5원)과 실제값(98.8원) 차이가 있어, 
        단기적으로 SMP 상승 여력이 있는 것으로 판단됩니다.
        신재생 발전자산 실적이 다음 분기에 개선될 것으로 전망합니다."
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
        <strong>💡 활용 팁</strong><br>
        • R² 0.5 이상이면 참고용으로 활용 가능, 0.7 이상이면 신뢰도 높음<br>
        • 학습 기간을 1년으로 설정하면 계절성이 반영됩니다<br>
        • 예측값이 실제값보다 높으면 저평가, 낮으면 고평가 상태
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 5. 데이터 탭
        st.markdown("### 5️⃣ 📋 데이터 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p>• 원본 데이터 조회 및 필터링</p>
        <p>• 날짜 범위, 카테고리별 필터링 가능</p>
        <p>• CSV 다운로드 기능 (별도 분석용)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

# 6. 수익성 시뮬레이터 탭
        st.markdown("### 6️⃣ 🌱 수익성 시뮬레이터 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p>• <strong>발전 유형:</strong> 태양광, 풍력(육상/해상), 연료전지, 바이오 선택</p>
        <p>• <strong>설비 용량:</strong> MW 단위 입력</p>
        <p>• <strong>이용률:</strong> 발전 유형별 기본값 제공 (태양광 15%, 풍력 25~30%, 연료전지 85%)</p>
        <p>• <strong>REC 가중치:</strong> 발전 유형별 REC 가중치 적용</p>
        <p>• <strong>시나리오 분석:</strong> 다양한 SMP 가격 시나리오별 수익 비교</p>
        <p>• <strong>손익분기점:</strong> CAPEX, OPEX 입력 시 투자회수 기간 자동 계산</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <strong>💼 활용 예시: 신규 투자 검토</strong><br><br>
        <strong>상황:</strong> 10MW 태양광 발전소 인수 검토 중<br><br>
        <strong>시뮬레이션 결과:</strong><br>
        • SMP 100원 시나리오: 연간 2.5억원<br>
        • SMP 150원 시나리오: 연간 3.2억원<br>
        • 현재 SMP(98원) 기준 투자회수 기간: 6.2년<br><br>
        <strong>의사결정:</strong> "현재 SMP가 평균 대비 낮은 수준이라 보수적 시나리오(SMP 100원)로 
        검토해도 7년 내 회수 가능. 인수 진행 추천"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
        <strong>💡 활용 팁</strong><br>
        • 보수적/기본/낙관적 3개 시나리오로 항상 검토하세요<br>
        • REC 가격은 현재 시장가 기준으로 입력하되, 하락 시나리오도 고려하세요<br>
        • CAPEX는 EPC 견적 + 개발비 + 인허가비용을 포함하세요
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 7. 투자 시그널 탭
        st.markdown("### 7️⃣ 🔔 투자 시그널 탭")
        st.markdown("""
        <div class="manual-section">
        <h4>기능 설명</h4>
        <p>시그널은 최근 30일 평균 대비 현재 값의 위치를 기준으로 자동 생성됩니다.</p>
        <table style="color: #fff; width: 100%;">
        <tr><th style="text-align:left;">시그널</th><th style="text-align:left;">기준</th><th style="text-align:left;">의미</th></tr>
        <tr><td>🟢 BUY (매수 적기)</td><td>평균 - 1σ 이하</td><td>저점 매수 기회</td></tr>
        <tr><td>🔴 SELL (매도 고려)</td><td>평균 + 1σ 이상</td><td>고점 매도 검토</td></tr>
        <tr><td>🟡 HOLD (관망)</td><td>특이사항 감지</td><td>추가 분석 필요</td></tr>
        </table>
        <br>
        <p><strong>분석 대상:</strong></p>
        <p>• <strong>SMP:</strong> 신재생 발전 수익성 → 저점 시 발전자산 매수, 고점 시 PPA 재협상</p>
        <p>• <strong>REC:</strong> REC 현물 매매 → 저점 시 REC 매수 비축</p>
        <p>• <strong>금리:</strong> PF 조달 → 저점 시 리파이낸싱, 고점 시 고정금리 전환</p>
        <p>• <strong>환율:</strong> 해외 투자 → 원화 강세 시 해외투자 적기</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <strong>💼 활용 예시: 투자심의위원회</strong><br><br>
        <strong>시그널:</strong> "🟢 인프라 - 금리: 금리 하락 - 차입 적기"<br><br>
        <strong>보고:</strong> "현재 국고채 3년물이 30일 평균 대비 15bp 낮은 수준입니다. 
        보유 중인 A발전소 PF 리파이낸싱을 이번 달 내 실행하면 연간 이자비용 약 2억원 절감 예상됩니다.
        리파이낸싱 승인 요청드립니다."
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
        <strong>💡 활용 팁</strong><br>
        • 시그널은 참고용이며, 최종 의사결정은 종합적 판단이 필요합니다<br>
        • STRONG 시그널(2σ 이상)은 특히 주의 깊게 검토하세요<br>
        • 하단의 "종합 시장 분석"을 함께 참고하세요
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 8. FAQ
        st.markdown("### ❓ 자주 묻는 질문 (FAQ)")
        
        with st.expander("Q. 데이터는 얼마나 자주 업데이트되나요?"):
            st.markdown("**A.** 경영지원팀에서 데일리 클리핑 자료를 업데이트할 때마다 반영됩니다. 엑셀 파일을 교체하고 '데이터 새로고침' 버튼을 누르면 최신 데이터가 로드됩니다.")
        
        with st.expander("Q. 급변동 알림의 임계값을 변경할 수 있나요?"):
            st.markdown("**A.** 현재 버전에서는 코드 수정이 필요합니다. `ALERT_THRESHOLDS` 딕셔너리에서 카테고리별 임계값을 조정할 수 있습니다.")
        
        with st.expander("Q. LNG 데이터가 다른 지표와 다르게 표시되는 이유는?"):
            st.markdown("**A.** LNG(탱크로리용, 연료전지용)는 월별로 업데이트되는 데이터입니다. 따라서 가장 최근 유효값을 표시하며, 해당 월을 괄호로 표기합니다.")
        
        with st.expander("Q. 투자 시그널을 그대로 따라도 되나요?"):
            st.markdown("**A.** 시그널은 통계적 분석 결과일 뿐, 투자 조언이 아닙니다. 반드시 다른 요소(시장 상황, 규제 변화, 내부 전략 등)와 종합적으로 판단하세요.")
        
        with st.expander("Q. 상관관계가 높으면 항상 예측이 맞나요?"):
            st.markdown("**A.** 아닙니다. 상관관계는 과거 데이터 기반이며, 미래에도 동일한 패턴이 유지된다는 보장이 없습니다. 특히 시장 구조 변화(정책 변경, 외부 충격 등) 시 상관관계가 깨질 수 있습니다.")
        
        st.markdown("---")        
        st.markdown("""
        <div style="text-align: center; color: #888; padding: 1rem;">
        📧 문의: 박연준(yjpark@ifasset.co.kr) | 📅 최종 업데이트: 2025.12
        </div>
        """, unsafe_allow_html=True)
    
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
                    is_lng = ind.get('is_lng', False)
                    change_html = get_change_html(ind['change'], ind['change_pct'], ind['direction'], is_rate, is_lng)
                    note = ind.get('note', '')
                    st.markdown(create_metric_card(col_name, value_str, change_html, note), unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: 수익성 시뮬레이터
    # =========================================================================
    with tab2:
        st.markdown("## 🌱 신재생에너지 수익성 시뮬레이터")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ 프로젝트 설정")
            
            project_type = st.selectbox("발전 유형", ["태양광", "풍력(육상)", "풍력(해상)", "연료전지", "바이오"])
            
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
            
            current_smp = df['육지 SMP'].dropna().iloc[-1] if len(df['육지 SMP'].dropna()) > 0 else 100
            current_rec = df['육지 가격'].dropna().iloc[-1] if len(df['육지 가격'].dropna()) > 0 else 70000
            
            smp_scenarios = st.multiselect("SMP 시나리오 (원/kWh)", [80, 100, 120, 150, 180, 200, 220], default=[100, 150, 200])
            rec_scenario = st.number_input("REC 가격 (원/REC)", min_value=10000, max_value=200000, value=int(current_rec), step=1000)
        
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
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
                fig = go.Figure()
                revenues = [calculate_renewable_revenue(smp, rec_scenario, capacity, cf, rec_weight)['total_revenue']/100000000 for smp in smp_scenarios]
                fig.add_trace(go.Bar(x=[f"SMP {s}" for s in smp_scenarios], y=revenues, marker_color='#27ae60',
                                    text=[f"{r:.1f}억" for r in revenues], textposition='outside'))
                fig.update_layout(title=f"{project_type} {capacity}MW 연간 예상 수익", yaxis_title="총 수익 (억원)",
                                 template='plotly_dark', paper_bgcolor='rgba(22,33,62,0.8)', plot_bgcolor='rgba(22,33,62,0.8)', height=350)
                st.plotly_chart(fig, use_container_width=True)
                
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
        st.markdown("최근 30일 평균 대비 현재 값의 위치를 기준으로 시그널을 생성합니다.")
        
        signals = generate_investment_signals(df, days=30)
        
        if signals:
            for signal in signals:
                if signal['signal'] == 'BUY':
                    css_class, icon, label = 'signal-buy', '🟢', '매수 적기'
                elif signal['signal'] == 'SELL':
                    css_class, icon, label = 'signal-sell', '🔴', '매도 고려'
                else:
                    css_class, icon, label = 'signal-hold', '🟡', '관망'
                
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
            st.info("현재 특별한 투자 시그널이 없습니다.")
        
        st.markdown("---")
        st.markdown("### 📋 종합 시장 분석")
        
        latest = df.iloc[-1]
        analysis_points = []
        
        smp_current = latest.get('육지 SMP')
        smp_avg_90d = df.tail(90)['육지 SMP'].mean()
        if pd.notna(smp_current) and pd.notna(smp_avg_90d):
            smp_vs_avg = (smp_current / smp_avg_90d - 1) * 100
            if smp_vs_avg > 10:
                analysis_points.append(f"⚡ SMP가 90일 평균 대비 **{smp_vs_avg:.1f}% 높음** - 신재생 발전 수익성 양호")
            elif smp_vs_avg < -10:
                analysis_points.append(f"⚡ SMP가 90일 평균 대비 **{abs(smp_vs_avg):.1f}% 낮음** - 수익성 주의")
        
        rate_current = latest.get('국고채 (3년)')
        rate_avg_90d = df.tail(90)['국고채 (3년)'].mean()
        if pd.notna(rate_current) and pd.notna(rate_avg_90d):
            if rate_current > rate_avg_90d + 0.2:
                analysis_points.append(f"📊 금리 상승 추세 ({rate_current:.2f}%) - PF 조달비용 상승 예상")
            elif rate_current < rate_avg_90d - 0.2:
                analysis_points.append(f"📊 금리 하락 추세 ({rate_current:.2f}%) - 리파이낸싱 적기")
        
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
            heatmap_period = st.selectbox("분석 기간", ["3개월", "6개월", "1년", "전체"], index=2, key="hm_p")
            heatmap_indicators = st.multiselect("분석 지표", KEY_INDICATORS,
                default=["달러환율", "육지 SMP", "두바이유", "국고채 (3년)"], key="hm_i")
        
        with col2:
            if len(heatmap_indicators) >= 2:
                days = CHART_PERIODS.get(heatmap_period)
                corr_matrix = calculate_correlation_matrix(df, heatmap_indicators, days)
                
                fig = px.imshow(corr_matrix, labels=dict(color="상관계수"), x=heatmap_indicators, y=heatmap_indicators,
                               color_continuous_scale='RdBu_r', zmin=-1, zmax=1, text_auto='.2f')
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(22,33,62,0.8)', plot_bgcolor='rgba(22,33,62,0.8)', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🕐 시차(Lag) 분석")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            leading = st.selectbox("선행지표", KEY_INDICATORS, index=5, key="ld")
        with col2:
            lagging = st.selectbox("후행지표", KEY_INDICATORS, index=3, key="lg")
        with col3:
            max_lag = st.slider("최대 시차", 1, 60, 30, key="ml")
        
        if leading != lagging:
            lag_df = calculate_lagged_correlation(df, leading, lagging, max_lag)
            optimal = find_optimal_lag(lag_df)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lag_df['lag'], y=lag_df['correlation'], mode='lines+markers', line=dict(color='#3498db')))
            if optimal is not None:
                fig.add_vline(x=optimal['lag'], line_dash="dash", line_color="#e94560")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(title=f"{leading} → {lagging}", template='plotly_dark',
                             paper_bgcolor='rgba(22,33,62,0.8)', plot_bgcolor='rgba(22,33,62,0.8)', height=300, yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig, use_container_width=True)
            
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
            # 1) 예측 대상 선택
            target = st.selectbox("예측 대상", KEY_INDICATORS, index=3, key="pt")
            
            # 2) 설명 변수 옵션 리스트 (타깃은 제외)
            feature_options = [x for x in KEY_INDICATORS if x != target]
            
            # 3) 기본 추천 설명 변수 후보
            base_default = ["두바이유", "달러환율"]
            #    → 실제 옵션에 존재하는 것만 기본값으로 사용
            default_features = [x for x in base_default if x in feature_options]
            
            # 4) 멀티셀렉트 (에러 안 나게 default를 옵션에 맞게 조정)
            features = st.multiselect(
                "설명 변수",
                feature_options,
                default=default_features,
                key="pf",
            )
            
            # 5) 학습 기간 / 실행 버튼
            train_period = st.selectbox("학습 기간", ["3개월", "6개월", "1년"], index=2, key="tp")
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
                    fig.add_trace(go.Scatter(
                        x=model_info['dates'],
                        y=model_info['y_actual'],
                        mode='lines',
                        name='실제값',
                        line=dict(color='#3498db')
                    ))
                    fig.add_trace(go.Scatter(
                        x=model_info['dates'],
                        y=model_info['y_pred'],
                        mode='lines',
                        name='예측값',
                        line=dict(color='#e94560', dash='dot')
                    ))
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
            table_cat = st.selectbox("카테고리", ['전체'] + list(INDICATORS.keys()), key="tc")
        
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
        🌱 친환경·인프라 투자 대시보드 v5.0 | 신재생에너지·순환경제·금융 지표 대시보드
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
