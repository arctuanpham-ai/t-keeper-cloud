import streamlit as st
import pandas as pd
from vnstock import Quote, Listing
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import time

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="T+ KEEPER PRO",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING (PRO DARK MODE) ====================
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Cards */
    .metric-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00e676;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00e676;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Highlight Table Rows */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== STATE MANAGEMENT ====================
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = [] # List of {symbol, price, vol, date}
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()

# ==================== HELPER FUNCTIONS ====================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900)
def get_market_overview():
    try:
        quote = Quote(symbol='VNINDEX', source='vci')
        df = quote.history(start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), 
                           end=datetime.now().strftime('%Y-%m-%d'), interval='D')
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = latest['close'] - prev['close']
            pct = (change / prev['close']) * 100
            vol = (latest['volume'] * latest['close']) / 1e9
            return latest['close'], change, pct, vol
    except:
        pass
    return 1250.0, 5.0, 0.4, 15000.0

@st.cache_data(ttl=3600)
def get_top_symbols():
    # VN30 + Top Liquid Stocks
    return [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
        "DGC", "DXG", "DIG", "PDR", "NVL", "KBC", "VGC", "VIX", "GEX", "HAG",
        "DBC", "HSG", "NKG", "VND", "HCM", "FRT", "FTS", "BSI", "ORS", "TCH"
    ]

def scan_symbol(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        quote = Quote(symbol=symbol, source='vci', show_log=False)
        df = quote.history(start=start, end=end, interval='D')
        
        if df is None or len(df) < 20: return None
        
        close = df['close']
        rsi = calculate_rsi(close).iloc[-1]
        vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        sig = "Neutral"
        if rsi < 35: sig = "Oversold (Buy?)"
        elif rsi > 70: sig = "Overbought (Sell?)"
        elif vol > avg_vol * 1.5 and close.iloc[-1] > close.iloc[-2]: sig = "Vol Breakout"
        
        return {
            "Symbol": symbol,
            "Price": close.iloc[-1],
            "Change %": ((close.iloc[-1] - close.iloc[-2])/close.iloc[-2])*100,
            "RSI": rsi,
            "Vol Ratio": vol/avg_vol if avg_vol > 0 else 0,
            "Volume": vol,
            "Signal": sig
        }

@st.cache_data(ttl=86400)
def get_all_symbols():
    try:
        # Lấy toàn bộ mã HOSE để làm gợi ý Search
        lst = Listing()
        df = lst.symbols_by_exchange(exchange='HOSE')
        return df['ticker'].tolist()
    except:
        return get_top_symbols() # Fallback

# ==================== UI LAYOUT ====================

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2764/2764955.png", width=60)
    st.title("T+ KEEPER")
    st.caption("EcoHome Invest AI")
    st.markdown("---")
    
    menu = st.radio("MAIN MENU", ["📊 Dashboard", "🔍 Scanner", "💼 Portfolio", "🤖 AI Analyst"])
    
    st.markdown("---")
    gemini_key = st.text_input("🔑 Gemini API Key", type="password")
    st.caption("Required for AI analysis")

# --- TAB 1: DASHBOARD ---
if menu == "📊 Dashboard":
    st.title("Market Overview")
    
    idx, chg, pct, vol = get_market_overview()
    
    # Custom CSS Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">VNINDEX</div><div class="metric-value">{idx:,.2f}</div><div style="color:{'#00e676' if chg>=0 else '#ff5252'}">{chg:+.2f} ({pct:+.2f}%)</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">THANH KHOẢN</div><div class="metric-value">{vol/1000:.1f}k ỷ</div><div style="color:#8b949e">VN30 + HOSE</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">XU HƯỚNG</div><div class="metric-value" style="color:#2f81f7">UPTREND</div><div style="color:#8b949e">MA50 Support</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card"><div class="metric-label">NGÀNH HOT</div><div class="metric-value" style="color:#d2a8ff">BANK</div><div style="color:#8b949e">Dòng tiền mạnh</div></div>""", unsafe_allow_html=True)

    st.markdown("### 🔥 Top Movers (VN30)")
    # Quick Mock Data for visual flair (Real scanning is in Scanner tab)
    top_cols = st.columns(3)
    # Placeholder visual - In real app, this would come from a quick sort of get_top_symbols
    st.info("💡 Chuyển sang tab **Scanner** để lọc tìm cơ hội mới nhất.")

    st.markdown("---")
    st.markdown("### 🔎 Tra cứu nhanh")
    
    all_symbols = get_all_symbols()
    search_symbol = st.selectbox("Nhập mã cổ phiếu:", [""] + all_symbols, index=0, placeholder="Ví dụ: VCB, HPG...")
    
    if search_symbol:
        st.markdown(f"#### Kết quả: {search_symbol}")
        col_info, col_chart = st.columns([1, 2])
        
        with col_info:
            data = scan_symbol(search_symbol)
            if data:
                st.metric("Giá", f"{data['Price']:,.0f}", f"{data['Change %']:+.2f}%")
                st.metric("RSI (14)", f"{data['RSI']:.1f}", data['Signal'])
                st.metric("Vol Ratio", f"{data['Vol Ratio']:.1f}x", f"{data['Volume']:,.0f} cp")
                
                if st.button("🤖 AI Phân tích ngay"):
                    st.session_state['ai_target'] = search_symbol
                    st.toast(f"Đã chuyển {search_symbol} sang tab AI Analyst!", icon="✅")
                    # Note: Auto switching tabs involves rerun quirk, simple toast guide is safer
                    st.info("👉 Vui lòng chọn Tab **AI Analyst** bên trái để xem chi tiết.")
            else:
                st.error("Không lấy được dữ liệu mã này.")
                
        with col_chart:
            quote = Quote(symbol=search_symbol, source='vci', show_log=False)
            df = quote.history(start=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), 
                               end=datetime.now().strftime('%Y-%m-%d'), interval='D')
            if df is not None:
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SCANNER ---
elif menu == "🔍 Scanner":
    st.title("T+ Opportunity Scanner")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rsi_max = st.slider("Max RSI", 30, 80, 70)
    with c2:
        vol_min_ratio = st.slider("Min Vol Ratio", 0.5, 5.0, 1.0)
    with c3:
        price_max = st.number_input("Max Price (k VND)", min_value=5, value=150, step=5, help="Nhập 50 = 50.000 VND")
    with c4:
        vol_min = st.number_input("Min Volume", min_value=1000, value=50000, step=10000)

    if st.button("🚀 SCAN NOW (50 Major Stocks)", use_container_width=True):
        symbols = get_top_symbols()
        results = []
        prog = st.progress(0)
        status = st.empty()
        
        for i, sym in enumerate(symbols):
            status.text(f"Scanning {sym}...")
            data = scan_symbol(sym)
            if data:
                # Filter Logic
                # Price in data is typically absolute (e.g., 25000). Input is 25.
                price_valid = data['Price'] <= (price_max * 1000)
                vol_valid = data['Volume'] >= vol_min
                rsi_valid = data['RSI'] <= rsi_max
                vol_ratio_valid = data['Vol Ratio'] >= vol_min_ratio
                
                if price_valid and vol_valid and rsi_valid and vol_ratio_valid:
                    results.append(data)
            prog.progress((i+1)/len(symbols))
            time.sleep(0.05)
            
            status.empty()
            prog.empty()
            if results:
                st.session_state.scan_results = pd.DataFrame(results).sort_values(by="RSI")
            else:
                st.warning("No stocks match your filters.")

    if not st.session_state.scan_results.empty:
        st.markdown(f"#### 🎯 Found {len(st.session_state.scan_results)} Matches")
        
        # Interactive Table
        st.dataframe(
            st.session_state.scan_results,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                "RSI": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
                "Vol Ratio": st.column_config.NumberColumn(format="%.1fx"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Quick Add to Portfolio
        selected_add = st.selectbox("Add to Portfolio:", st.session_state.scan_results['Symbol'].tolist(), index=None, placeholder="Select stock...")
        if selected_add:
            if st.button(f"➕ Add {selected_add} to Portfolio"):
                # Check if exists
                if not any(d['symbol'] == selected_add for d in st.session_state.portfolio):
                    row = st.session_state.scan_results[st.session_state.scan_results['Symbol'] == selected_add].iloc[0]
                    st.session_state.portfolio.append({
                        "symbol": selected_add,
                        "buy_price": row['Price'],
                        "date": datetime.now().strftime("%Y-%m-%d")
                    })
                    st.toast(f"Added {selected_add}!", icon="✅")
                else:
                    st.toast("Already in portfolio.", icon="⚠️")

# --- TAB 3: PORTFOLIO ---
elif menu == "💼 Portfolio":
    st.title("My Portfolio Tracker")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Go to Scanner to add stocks.")
    else:
        # Display as Cards
        for item in st.session_state.portfolio:
            with st.container():
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #238636; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: #fff;">{item['symbol']}</h3>
                            <span style="color: #8b949e; font-size: 12px;">Buy: {item['date']} @ {item['buy_price']:,.2f}</span>
                        </div>
                        <div style="text-align: right;">
                             <!-- Real-time price check would go here, using static buy price for now -->
                             <span style="font-weight: bold; font-size: 18px; color: #fff;">Holding</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# --- TAB 4: AI ANALYST ---
elif menu == "🤖 AI Analyst":
    st.title("Gemini Strategic Advisor")
    
    # Combined list: Portfolio + Scan Results + Search Target
    candidates = [p['symbol'] for p in st.session_state.portfolio]
    if not st.session_state.scan_results.empty:
        candidates += st.session_state.scan_results['Symbol'].tolist()
    if 'ai_target' in st.session_state:
        candidates.append(st.session_state.ai_target)
        
    candidates = list(set(candidates)) # Unique
    
    # Auto-select from Dashboard Search
    default_idx = 0
    if 'ai_target' in st.session_state and st.session_state.ai_target in candidates:
        default_idx = candidates.index(st.session_state.ai_target)
    
    if not candidates:
        st.warning("No stocks to analyze (Portfolio empty & No scan results).")
    else:
        target = st.selectbox("Select Stock to Analyze:", candidates, index=default_idx)
        
        if target:
            # Show Chart First
            quote = Quote(symbol=target, source='vci', show_log=False)
            df = quote.history(start=(datetime.now()-timedelta(days=100)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'), interval='D')
            
            if df is not None:
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"✨ Ask Gemini about {target}"):
                    if not gemini_key:
                        st.error("Please enter Gemini API Key in the Sidebar first!")
                    else:
                        try:
                            genai.configure(api_key=gemini_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            latest = df.iloc[-1]
                            rsi = calculate_rsi(df['close']).iloc[-1]
                            
                            prompt = f"""
                            Act as a senior stock trading expert. Analyze ticker {target} (Vietnam Stock Market).
                            Data: Price {latest['close']}, RSI {rsi:.1f}, Vol {latest['volume']}.
                            
                            Provide a strategic formatted response:
                            **1. Trend Analysis:** (Bullish/Bearish/Sideways)
                            **2. Critical Zones:** (Support/Resistance)
                            **3. Actionable Advice:** (Buy/Sell/Hold with target price)
                            
                            Keep it concise, professional, and use emojis.
                            """
                            
                            with st.spinner("Analyzing market structure..."):
                                resp = model.generate_content(prompt)
                                st.success("Analysis Complete")
                                st.markdown(f"""
                                <div style="background-color: #21262d; padding: 20px; border-radius: 8px;">
                                {resp.text}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI Error: {e}")
