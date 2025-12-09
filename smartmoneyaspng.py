import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import requests

# 🧠 AI Çekirdeği
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

# ============================================================
# 🏛️ V15: TITAN PURE TABLE DASHBOARD
# ============================================================
CONFIG = {
    # --- 1. OPTİMİZE EDİLMİŞ PARAMETRELER ---
    "EMA_TREND": 20,
    "ATR_MULT": 2.0,
    "TRAILING_PCT": 0.05,

    # --- 2. GÜVENLİK DUVARI ---
    "MIN_MARKET_CAP": 50_000_000_000,
    "MIN_LIQUIDITY": 100_000_000,

    # --- 3. SMART MONEY ---
    "SQUEEZE_THRESHOLD": 0.85,
    "DRYNESS_THRESHOLD": 0.80,
    "STABILITY_THRESHOLD": 1.5,

    # --- 4. AI (Machine Learning) ---
    "AI_HORIZON": 5,
    "BACKTEST_DAYS": 120
}

# HEDEF LİSTE (BIST 100 Karması)
TICKERS = [
        "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS",
        "ALTNY.IS", "ANSGR.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BALSU.IS",
        "BIMAS.IS", "BINHO.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS",
        "CANTE.IS", "CCOLA.IS", "CIMSA.IS", "CLEBI.IS", "CWENE.IS", "DAPGM.IS",
        "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "ECILC.IS", "EFOR.IS", "EGEEN.IS",
        "EKGYO.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS",
        "FENER.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLRMK.IS",
        "GRSEL.IS", "GRTHO.IS", "GSRAY.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS",
        "IEYHO.IS", "ISCTR.IS", "ISMEN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS",
        "TRALT.IS", "KRDMD.IS", "KTLEV.IS", "KUYAS.IS", "MAGEN.IS", "MAVI.IS",
        "MGROS.IS", "MIATK.IS", "MPARK.IS", "OBAMS.IS", "ODAS.IS", "OTKAR.IS",
        "OYAKC.IS", "PASEU.IS", "PATEK.IS", "PETKM.IS", "PGSUS.IS", "RALYH.IS",
        "REEDR.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SOKM.IS",
        "TABGD.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS",
        "TRALT.IS", "TRENJ.IS", "TRMET.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS",
        "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TUREX.IS", "TURSG.IS", "ULKER.IS",
        "VAKBN.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS","XU100.IS"
    ]

# ============================================================
# 🛠️ FEATURE ENGINEERING
# ============================================================
def calculate_features(df):
    df = df.copy()
    # 1. Trend
    df['EMA_Trend'] = df['Close'].ewm(span=CONFIG['EMA_TREND'], adjust=False).mean()
    df['Trend_Dist'] = (df['Close'] - df['EMA_Trend']) / df['EMA_Trend']
    # 2. Squeeze
    df['Mid'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['BB_Width'] = ((df['Mid'] + 2*df['Std']) - (df['Mid'] - 2*df['Std'])) / df['Mid']
    df['Squeeze_Idx'] = df['BB_Width'] / df['BB_Width'].rolling(120).mean()
    # 3. Dryness
    df['Vol_Dry_Idx'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(50).mean()
    # 4. Stability
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()
    body_size = (df['Close'] - df['Open']).abs()
    df['Stability_Idx'] = body_size.rolling(10).mean() / df['ATR']
    # 5. Intensity
    df['Intensity'] = df['Volume'] / df['Volume'].rolling(50).mean()
    # 6. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    return df.dropna()

# ============================================================
# 🧠 AI ENGINE
# ============================================================
def get_ai_prediction(df):
    data = df.copy()
    data['Target'] = (data['Close'].shift(-CONFIG['AI_HORIZON']) > data['Close'] * 1.02).astype(int)
    data = data.dropna()
    if len(data) < 150: return 50.0

    features = ['Squeeze_Idx', 'Vol_Dry_Idx', 'Stability_Idx', 'Trend_Dist', 'Intensity', 'RSI']
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    try:
        model.fit(X_scaled[:-5], y[:-5])
        current_features = X_scaled[-1].reshape(1, -1)
        return model.predict_proba(current_features)[0][1] * 100
    except:
        return 50.0

# ============================================================
# 🧪 BACKTEST ENGINE
# ============================================================
def run_optimized_backtest(df):
    data = df.tail(CONFIG['BACKTEST_DAYS']).copy()
    balance = 10000.0
    position = 0.0
    entry_price = 0.0
    highest_price = 0.0

    for i in range(len(data)):
        row = data.iloc[i]
        price = row['Close']
        if position == 0:
            if row['Close'] > row['EMA_Trend'] and row['Intensity'] > 1.0:
                position = balance / price
                entry_price = price
                balance = 0
                highest_price = price
        elif position > 0:
            if price > highest_price: highest_price = price
            stop_price = highest_price * (1 - CONFIG['TRAILING_PCT'])
            atr_stop = entry_price - (row['ATR'] * CONFIG['ATR_MULT'])
            final_stop = max(stop_price, atr_stop)
            if price < final_stop or price < row['EMA_Trend']:
                balance = position * price
                position = 0
    if position > 0: balance = position * data.iloc[-1]['Close']
    return (balance - 10000) / 10000 * 100

# ============================================================
# 🎨 GÖRSELLEŞTİRME MOTORU (PURE TABLE)
# ============================================================
def create_dashboard_image(df_results):
    if df_results.empty: return

    # İlk 25 hisseyi al (Daha uzun liste)
    top_df = df_results.head(25).copy()

    # Dark Theme Ayarları
    plt.style.use('dark_background')
    
    # Boyutu ayarla (Uzun ve okunaklı)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')

    # Başlık
    plt.text(0.5, 0.96, f"TITAN MOMENTUM RADAR | {datetime.now().strftime('%d-%m-%Y')}", 
             ha='center', va='center', fontsize=22, color='#00ff99', weight='bold', fontfamily='sans-serif')
    
    plt.text(0.5, 0.93, "Borsa İstanbul Momentum & AI Analiz Raporu", 
             ha='center', va='center', fontsize=12, color='#aaaaaa')

    # --- KOLON AYARLARI (Hizalama için) ---
    # x_pos: yatay konum, align: hizalama türü
    cols = [
        {"name": "HİSSE",      "x": 0.05, "align": "left"},
        {"name": "FİYAT",      "x": 0.20, "align": "right"},
        {"name": "5G SKOR",    "x": 0.30, "align": "center"},
        {"name": "MOMENTUM",   "x": 0.40, "align": "center"},
        {"name": "AI PROB",    "x": 0.50, "align": "center"},
        {"name": "ROI (KANIT)","x": 0.60, "align": "center"},
        {"name": "ETİKETLER",  "x": 0.68, "align": "left"},
    ]

    # --- BAŞLIKLARI YAZ ---
    header_y = 0.88
    # Başlık arka plan çizgisi
    plt.plot([0.02, 0.98], [header_y - 0.015, header_y - 0.015], color='white', linewidth=1)

    for col in cols:
        plt.text(col["x"], header_y, col["name"], 
                 color='white', weight='bold', fontsize=12, 
                 ha=col["align"], va='center', fontfamily='monospace')

    # --- SATIRLARI DÖNGÜYE AL ---
    y_start = 0.84
    row_height = 0.032  # Satır aralığı (düzgün aralık)

    for idx, row in top_df.iterrows():
        y = y_start - (idx * row_height)
        
        # Renk Kodları
        mom_val = row['Momentum']
        if mom_val > 0: mom_color = '#00ff00' # Yeşil
        elif mom_val < 0: mom_color = '#ff3333' # Kırmızı
        else: mom_color = '#888888' # Gri
        
        mom_text = f"+{int(mom_val)}" if mom_val > 0 else f"{int(mom_val)}"

        # AI Rengi
        ai_val = row['AI_Prob']
        if ai_val > 70: ai_color = '#00ff00' # Çok iyi
        elif ai_val > 60: ai_color = '#ffff00' # Orta
        else: ai_color = 'white'

        # ROI Rengi
        roi_val = row['ROI_Kanit']
        roi_color = '#ff3333' if roi_val < 0 else 'white'

        # Verileri Yazdırma (Hizalamaya Dikkat Et)
        # 1. Hisse
        plt.text(cols[0]["x"], y, row['Hisse'], color='white', fontsize=11, weight='bold', ha='left', va='center')
        
        # 2. Fiyat (Sağa yaslı - finansal format)
        plt.text(cols[1]["x"], y, f"{row['Fiyat']:.2f}", color='white', fontsize=11, ha='right', va='center', fontfamily='monospace')
        
        # 3. Skor (Ortalı)
        plt.text(cols[2]["x"], y, f"{int(row['Total_Score_5D'])}", color='cyan', fontsize=11, weight='bold', ha='center', va='center')
        
        # 4. Momentum (Ortalı)
        plt.text(cols[3]["x"], y, mom_text, color=mom_color, fontsize=11, weight='bold', ha='center', va='center')
        
        # 5. AI Prob
        plt.text(cols[4]["x"], y, f"%{ai_val:.1f}", color=ai_color, fontsize=11, ha='center', va='center')
        
        # 6. ROI
        plt.text(cols[5]["x"], y, f"%{roi_val:.1f}", color=roi_color, fontsize=11, ha='center', va='center')
        
        # 7. Etiketler
        plt.text(cols[6]["x"], y, row['Etiketler'][:45], color='#aaaaaa', fontsize=10, ha='left', va='center')

        # İnce ayırıcı çizgi (Her 5 satırda bir daha belirgin çizgi atalım)
        line_alpha = 0.3 if (idx + 1) % 5 == 0 else 0.1
        plt.plot([0.02, 0.98], [y - 0.015, y - 0.015], color='white', alpha=line_alpha, linewidth=0.5)

    # Footer
    plt.text(0.05, 0.02, "ℹ️ Skor: Son 5 günde toplanan teknik puanlar. | Momentum: Skorun dünden bugüne değişimi (Yeşil: Güçleniyor).", 
             color='#666666', fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig('titan_momentum_table.png', dpi=200, facecolor='black', bbox_inches='tight')
    send_telegram_png('titan_momentum_table.png')
    print("\n📸  Mükemmel Tablo Oluşturuldu: titan_momentum_table.png")
    plt.close()

def send_telegram_png(file_path):
    url = f"https://api.telegram.org/bot8396968001:AAEKIdyqNIEWMyGwbVzBonzlDF9kCd3D9DA/sendPhoto"
    with open(file_path, "rb") as photo:
        requests.post(
            url,
            data={"chat_id": -5050366031},
            files={"photo": photo}
        )
# ============================================================
# 🚀 TITAN RUNNER
# ============================================================
def run_titan_prod():
    print(f"\n🏛️  V15: TITAN PURE TABLE | {datetime.now().strftime('%d-%m-%Y')}")
    print("⏳ Veri işleniyor...")

    raw_data = yf.download(TICKERS, period="2y", group_by='ticker', auto_adjust=False, progress=False)
    results = []

    for t in TICKERS:
        if t == "XU100.IS": continue
        try:
            try:
                info = yf.Ticker(t).fast_info
                mcap = info['market_cap']
                if mcap < CONFIG['MIN_MARKET_CAP']: continue
            except: continue

            df = raw_data[t].copy().dropna()
            if len(df) < 200: continue

            df = calculate_features(df)
            last = df.iloc[-1]

            if last['Close'] * last['Volume'] < CONFIG['MIN_LIQUIDITY']: continue
            if last['Close'] < last['EMA_Trend']: continue

            # --- SKORLAMA ---
            recent_days = df.tail(20).copy()
            recent_days['Daily_Score'] = (
                (recent_days['Squeeze_Idx'] < CONFIG['SQUEEZE_THRESHOLD']).astype(int) +
                (recent_days['Vol_Dry_Idx'] < CONFIG['DRYNESS_THRESHOLD']).astype(int) +
                (recent_days['Stability_Idx'] < CONFIG['STABILITY_THRESHOLD']).astype(int) +
                (recent_days['Intensity'] > 1.5).astype(int)
            )

            recent_days['Rolling_Score_5D'] = recent_days['Daily_Score'].rolling(5).sum()
            cum_score_now = recent_days['Rolling_Score_5D'].iloc[-1]
            cum_score_prev = recent_days['Rolling_Score_5D'].iloc[-2]
            momentum = cum_score_now - cum_score_prev

            ai_prob = get_ai_prediction(df)
            roi = run_optimized_backtest(df)
            
            reasons = []
            if last['Squeeze_Idx'] < CONFIG['SQUEEZE_THRESHOLD']: reasons.append("SIKIŞMA")
            if last['Vol_Dry_Idx'] < CONFIG['DRYNESS_THRESHOLD']: reasons.append("KURUMA")
            if last['Stability_Idx'] < CONFIG['STABILITY_THRESHOLD']: reasons.append("STABİL")
            if last['Intensity'] > 1.5: reasons.append("🔥 HACİM ŞOKU")

            results.append({
                "Hisse": t.replace(".IS",""),
                "Fiyat": last['Close'],
                "Total_Score_5D": cum_score_now,
                "Momentum": momentum,
                "AI_Prob": ai_prob,
                "ROI_Kanit": roi,
                "Etiketler": " + ".join(reasons) if reasons else "TREND TAKİBİ"
            })

        except Exception as e: continue

    if not results:
        print("❌ Uygun hisse yok.")
        return

    # Sıralama
    df_res = pd.DataFrame(results).sort_values(["Total_Score_5D", "Momentum", "AI_Prob"], ascending=[False, False, False])
    
    # Text Rapor (Yedek)
    print(df_res[["Hisse", "Total_Score_5D", "Momentum", "AI_Prob"]].head(10))
    
    # 📸 PNG TABLE
    create_dashboard_image(df_res)

if __name__ == "__main__":
    run_titan_prod()