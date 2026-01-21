import streamlit as st
import pandas as pd
from vnstock import listing_companies, stock_historical_data, Quote, Listing
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import time

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="T+ KEEPER - CLOUD",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Colors & CSS for Dark Mode Professional UI
st.markdown("""
<style>
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00e676; /* Green Accent */
    }
    [data-testid="stMetricDelta"] svg {
        fill: #00e676;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f2f6; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2962ff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0039cb;
        border-color: #00e676;
    }
    
    /* Highlights */
    .highlight-row { 
        background-color: rgba(41, 98, 255, 0.1); 
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCS ====================

def calculate_rsi(series, period=14):
    """Tính RSI chuẩn"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900) # 15 min cache
def get_market_metrics():
    """Lấy chỉ số VNINDEX cơ bản để hiển thị Dashboard (Mock hoặc Real nếu API cho phép)"""
    try:
        # VNStock Quote API có thể lấy snapshot, ở đây lấy lịch sử ngày gần nhất của VNINDEX
        quote = Quote(symbol='VNINDEX', source='vci')
        df = quote.history(start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), 
                           end=datetime.now().strftime('%Y-%m-%d'), interval='D')
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = latest['close'] - prev['close']
            pct_change = (change / prev['close']) * 100
            
            # Ước lượng thanh khoản (tỷ VND)
            vol_val = (latest['volume'] * latest['close']) / 1e9 
            return latest['close'], change, pct_change, vol_val
    except:
        pass
    return 1250.0, 5.0, 0.4, 15000.0 # Fallback mock

@st.cache_data(ttl=3600)
def get_scan_list():
    """Lấy danh sách VN30 + Top Midcap (50 mã)"""
    # Vì vnstock listing khá nặng, ta hardcode VN30 để tối ưu Cloud
    vn30 = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
    ]
    # Thêm 20 mã hot khác
    midcap = [
        "DGC", "DXG", "DIG", "PDR", "NVL", "KBC", "VGC", "VIX", "GEX", "HAG",
        "DBC", "HSG", "NKG", "VND", "HCM", "FRT", "FTS", "BSI", "ORS", "TCH"
    ]
    return vn30 + midcap

def scan_stock(symbol):
    """Xử lý từng mã: lấy data -> tính chỉ báo"""
    try:
        end_str = datetime.now().strftime('%Y-%m-%d')
        start_str = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
        
        quote = Quote(symbol=symbol, source='vci', show_log=False)
        df = quote.history(start=start_str, end=end_str, interval='D')
        
        if df is None or len(df) < 20:
            return None
            
        # Indicator Calc
        close = df['close']
        rsi = calculate_rsi(close).iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        
        # T+ Score Logic (Simple)
        score = 0
        if rsi < 40: score += 1
        if close.iloc[-1] > ma20: score += 1
        if vol > avg_vol * 1.2: score += 1
        
        return {
            "Mã": symbol,
            "Giá": close.iloc[-1],
            "RSI": round(rsi, 2),
            "MA20": round(ma20, 2),
            "Vol đột biến": f"{round(vol/avg_vol, 1)}x" if avg_vol > 0 else "N/A",
            "T+ Score": score
        }
    except:
        return None

# ==================== MAIN UI ====================

st.title("🛡️ T+ KEEPER | Cloud Edition")

# --- Sidebar ---
st.sidebar.header("⚙️ Cấu hình Scan")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Nhập API Key để dùng AI Analyst")
rsi_filter = st.sidebar.slider("Ngưỡng RSI Max", 30, 80, 70)
st.sidebar.info("Phiên bản Cloud giới hạn quét 50 mã (VN30 + Midcap) để đảm bảo tốc độ.")

# --- Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
idx_price, idx_change, idx_pct, idx_vol = get_market_metrics()

col1.metric("VNINDEX", f"{idx_price:,.2f}", f"{idx_change:+.2f} ({idx_pct:+.2f}%)")
col2.metric("Thanh khoản", f"{idx_vol:,.0f} tỷ", "Trung bình")
col3.metric("Trạng thái", "Tích lũy", "Neutral")

st.divider()

# --- Scanner Section ---
if st.button("🚀 SCAN MARKET (Top 50 Cap)"):
    with st.spinner("Đang quét thị trường... Vui lòng đợi..."):
        symbols = get_scan_list()
        results = []
        progress_bar = st.progress(0)
        
        for i, sym in enumerate(symbols):
            res = scan_stock(sym)
            if res:
                # Filter logic
                if res['RSI'] <= rsi_filter:
                    results.append(res)
            progress_bar.progress((i + 1) / len(symbols))
            time.sleep(0.05) # Tránh rate limit
            
        progress_bar.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="T+ Score", ascending=False)
            
            # Styling DataFrame
            st.success(f"Tìm thấy {len(df_res)} cơ hội tiềm năng!")
            st.dataframe(
                df_res,
                column_config={
                    "T+ Score": st.column_config.ProgressColumn(
                        "Điểm T+",
                        help="Điểm số cơ hội ngắn hạn (0-3)",
                        format="%d",
                        min_value=0,
                        max_value=3,
                    ),
                    "Giá": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            st.session_state['scan_results'] = df_res
        else:
            st.warning("Không tìm thấy mã nào thỏa mãn bộ lọc hiện tại.")

# --- Analysis Section ---
if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
    st.divider()
    st.subheader("🤖 AI Analyst")
    
    selected_stock = st.selectbox("Chọn mã để phân tích chi tiết:", st.session_state['scan_results']['Mã'])
    
    if st.button("Phân tích & Vẽ biểu đồ"):
        # Draw Chart
        quote = Quote(symbol=selected_stock, source='vci', show_log=False)
        df_hist = quote.history(start=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'), 
                                end=datetime.now().strftime('%Y-%m-%d'), interval='D')
        
        if df_hist is not None:
            # Candlestick Chart with Plotly
            fig = go.Figure(data=[go.Candlestick(x=df_hist.index,
                open=df_hist['open'],
                high=df_hist['high'],
                low=df_hist['low'],
                close=df_hist['close'])])
            
            fig.update_layout(title=f"Biểu đồ {selected_stock}", height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Gemini Analyst
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    latest_data = df_hist.iloc[-1]
                    rsi_now = calculate_rsi(df_hist['close']).iloc[-1]
                    
                    prompt = f"""
                    Bạn là một chuyên gia giao dịch T+ tại thị trường chứng khoán Việt Nam.
                    Hãy phân tích ngắn gọn về mã {selected_stock} dựa trên dữ liệu sau:
                    - Giá hiện tại: {latest_data['close']}
                    - RSI: {rsi_now:.2f}
                    - Volume: {latest_data['volume']}
                    
                    Đưa ra nhận định:
                    1. Xu hướng ngắn hạn (Tăng/Giảm/Đi ngang)
                    2. Vùng mua/bán khuyến nghị
                    3. Rủi ro cần chú ý
                    
                    Trả lời ngắn gọn dưới 150 từ, dùng emoji, style chuyên nghiệp.
                    """
                    
                    with st.spinner("Gemini đang suy nghĩ..."):
                        response = model.generate_content(prompt)
                        st.markdown(f"""
                        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #00e676;">
                            <b>🦅 Gemini Insight:</b><br><br>
                            {response.text}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Lỗi AI: {str(e)}")
            else:
                st.info("Nhập API Key bên trái để xem nhận định từ Gemini AI.")

# Footer
st.markdown("---")
st.caption("Powered by Vnstock & Streamlit Cloud | Dev: T+ Keeper")
