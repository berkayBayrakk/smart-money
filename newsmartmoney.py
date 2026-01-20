import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import requests

warnings.filterwarnings("ignore")

# ============================================================
# ⚙️ CONFIGURATION (RALLY OPTIMIZED)
# ============================================================
CONFIG = {
    "EMA_TREND": 20,
    "ATR_MULT": 2.0,
    "TRAILING_PCT": 0.05,
    "MIN_MARKET_CAP": 50_000_000_000,
    "MIN_LIQUIDITY": 150_000_000, # Rallide likidite beklentisi artar
    "SQUEEZE_THRESHOLD": 0.85,
    "DRYNESS_THRESHOLD": 0.80,
    "AI_HORIZON": 5,
    "BACKTEST_DAYS": 120
}

TICKERS = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS",
    "ALTNY.IS", "ANSGR.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS",
    "BRSAN.IS", "BRYAT.IS", "CCOLA.IS", "CIMSA.IS", "CWENE.IS", "DOAS.IS",
    "DOHOL.IS", "EGEEN.IS", "EKGYO.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS",
    "EREGL.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GESAN.IS", "GUBRF.IS",
    "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "ISMEN.IS", "KCAER.IS", "KCHOL.IS",
    "KONTR.IS", "KRDMD.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS", "MPARK.IS",
    "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PASEU.IS", "PATEK.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SOKM.IS",
    "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS",
    "TTKOM.IS", "TTRAK.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS",
    "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS"
]

# ============================================================
# 🛡️ RISK & PEAK DETECTION (ZİRVE ANALİZİ)
# ============================================================
def calculate_risk_score(df):
    """
    SMC ve Momentum bazlı risk hesaplama.
    Max 100 puan: 0-40 Güvenli, 40-70 Dikkat, 70+ Kritik
    """
    df = df.copy()
    risk = 0
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. Buying Climax: Hacim rekoru + Üst fitil
    upper_wick = last['High'] - max(last['Open'], last['Close'])
    body = abs(last['Close'] - last['Open'])
    if last['Intensity'] > 2.5 and upper_wick > (body * 1.5):
        risk += 30

    # 2. Bearish Divergence: Fiyat yeni zirve, RSI değil
    recent_20_high = df['Close'].tail(20).max()
    recent_20_rsi_high = df['RSI'].tail(20).max()
    if last['Close'] >= recent_20_high and last['RSI'] < recent_20_rsi_high:
        risk += 25

    # 3. RSI Overbought
    if last['RSI'] > 75: risk += 20
    if last['RSI'] > 85: risk += 15

    # 4. Churning: Hacim var ama fiyat gitmiyor
    if last['Intensity'] > 2.0 and abs(last['Close']/prev['Close'] - 1) < 0.003:
        risk += 10

    return min(risk, 100)

# ============================================================
# 🛠️ FEATURE ENGINEERING & AI
# ============================================================
def calculate_features(df):
    df = df.copy()
    df['EMA_Trend'] = df['Close'].ewm(span=CONFIG['EMA_TREND'], adjust=False).mean()
    df['Trend_Dist'] = (df['Close'] - df['EMA_Trend']) / df['EMA_Trend']

    # RSI & Squeeze
    df['Mid'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['BB_Width'] = (4 * df['Std']) / df['Mid']
    df['Squeeze_Idx'] = df['BB_Width'] / df['BB_Width'].rolling(120).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))

    df['Intensity'] = df['Volume'] / df['Volume'].rolling(50).mean()
    df['Vol_Dry_Idx'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(50).mean()

    return df.dropna()

def get_ai_prediction(df):
    data = df.copy()
    data['Target'] = (data['Close'].shift(-CONFIG['AI_HORIZON']) > data['Close'] * 1.02).astype(int)
    data = data.dropna()
    if len(data) < 150: return 50.0

    features = ['Squeeze_Idx', 'Trend_Dist', 'Intensity', 'RSI']
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    try:
        model.fit(X_scaled[:-5], y[:-5])
        return model.predict_proba(X_scaled[-1].reshape(1, -1))[0][1] * 100
    except: return 50.0

# ============================================================
# 🎨 DASHBOARD (V18: GUARDIAN EDITION)
# ============================================================
def create_dashboard_image(df_score, df_mom):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.patch.set_facecolor('#050505')
    ax.axis('off')

    plt.text(0.5, 0.97, "TITAN V18: RALLY & GUARDIAN", ha='center', fontsize=28, color='#00ff99', weight='bold')
    plt.text(0.5, 0.94, f"Zirve Analizi ve Akıllı Para Takibi | {datetime.now().strftime('%d/%m/%Y %H:%M')}", ha='center', fontsize=12, color='gray')

    plt.axvline(x=0.5, ymin=0.05, ymax=0.92, color='#222222', linewidth=2)

    # --- SOL: TOP 10 SKOR ---
    plt.text(0.25, 0.90, "🏆 TEKNİK LİDERLER", color='#00ddeb', fontsize=20, weight='bold', ha='center')
    y_pos = 0.82
    for i in range(min(10, len(df_score))):
        row = df_score.iloc[i]
        # Risk Rengi
        r_color = '#00ff99' if row['Risk'] < 40 else ('#ffcc00' if row['Risk'] < 70 else '#ff4444')
        risk_tag = "GÜVENLİ" if row['Risk'] < 40 else ("DİKKAT" if row['Risk'] < 70 else "KRİTİK")

        plt.text(0.05, y_pos, f"{i+1}. {row['Hisse']}", color='white', fontsize=17, weight='bold')
        plt.text(0.28, y_pos, f"Puan: {int(row['Score'])}", color='#00ddeb', fontsize=16, fontfamily='monospace')
        plt.text(0.45, y_pos, f"[{risk_tag}]", color=r_color, fontsize=13, weight='bold', ha='right')
        plt.plot([0.05, 0.45], [y_pos-0.02, y_pos-0.02], color='#111111', linewidth=1)
        y_pos -= 0.075

    # --- SAĞ: TOP 10 MOMENTUM ---
    plt.text(0.75, 0.90, "🚀 MOMENTUM / ARTIŞ", color='#ff4444', fontsize=20, weight='bold', ha='center')
    y_pos = 0.82
    for i in range(min(10, len(df_mom))):
        row = df_mom.iloc[i]
        plt.text(0.55, y_pos, f"{i+1}. {row['Hisse']}", color='white', fontsize=17)
        plt.text(0.80, y_pos, f"+{int(row['Mom'])}", color='#ff4444', fontsize=20, weight='bold', fontfamily='monospace')
        plt.text(0.95, y_pos, f"AI: %{int(row['AI'])}", color='gray', fontsize=12, ha='right')
        plt.plot([0.55, 0.95], [y_pos-0.02, y_pos-0.02], color='#111111', linewidth=1)
        y_pos -= 0.075

    plt.text(0.5, 0.02, "Risk Skorları: 70+ Üzeri 'Kritik' dağıtım aşamasını, 40- 'Güvenli' sağlıklı trendi temsil eder.", color='#444444', fontsize=10, ha='center', style='italic')
    plt.savefig('titan_v18_guardian.png', dpi=200, bbox_inches='tight', facecolor='#050505')
    plt.close()
    send_telegram_png('titan_v18_guardian.png')

def send_telegram_png(file_path):
    url = f"https://api.telegram.org/bot8396968001:AAEKIdyqNIEWMyGwbVzBonzlDF9kCd3D9DA/sendPhoto"
    with open(file_path, "rb") as photo:
        requests.post(
            url,
            data={"chat_id": -5050366031},
            files={"photo": photo}
        )
# ============================================================
# 🚀 RUNNER
# ============================================================
def run_titan_v18():
    print("⏳ TITAN V18 Guardian Başlatılıyor...")
    raw_data = yf.download(TICKERS, period="2y", group_by='ticker', progress=False)
    final_list = []

    for t in TICKERS:
        try:
            df = raw_data[t].copy().dropna()
            if len(df) < 150: continue

            df = calculate_features(df)
            last = df.iloc[-1]

            # Filtreler: Likidite ve Trend Altı Kontrolü
            if (last['Close'] * last['Volume']) < CONFIG['MIN_LIQUIDITY']: continue
            if last['Close'] < last['EMA_Trend']: continue

            # Skorlama (Rally Edition)
            recent = df.tail(10).copy()
            recent['DS'] = ( (recent['Close'] > recent['EMA_Trend']).astype(int) * 2 +
                             (recent['Intensity'] > 1.8).astype(int) +
                             (recent['RSI'] > 60).astype(int) )

            score_now = recent['DS'].rolling(5).sum().iloc[-1]
            score_prev = recent['DS'].rolling(5).sum().iloc[-2]

            final_list.append({
                "Hisse": t.replace(".IS",""),
                "Score": score_now,
                "Mom": score_now - score_prev,
                "AI": get_ai_prediction(df),
                "Risk": calculate_risk_score(df)
            })
        except: continue

    df_res = pd.DataFrame(final_list)
    df_score = df_res.sort_values("Score", ascending=False)
    df_mom = df_res.sort_values("Mom", ascending=False)

    create_dashboard_image(df_score, df_mom)
    print("✅ Dashboard 'titan_v18_guardian.png' olarak kaydedildi.")

if __name__ == "__main__":
    run_titan_v18()