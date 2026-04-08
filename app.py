import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Optimizador de Cartera · CEDEARs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  ESTILOS CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-title { color: #8888aa; font-size: 13px; margin-bottom: 6px; font-weight: 500; }
    .metric-value { font-size: 28px; font-weight: 700; margin: 0; }
    .metric-sub   { color: #666688; font-size: 12px; margin-top: 4px; }
    .color-minvar { color: #00ccff; }
    .color-sharpe { color: #ff6b35; }
    .color-equal  { color: #aaffaa; }
    .color-target { color: #a855f7; }

    .main-header {
        background: linear-gradient(135deg, #0f0f2e 0%, #1a0a3e 100%);
        border-bottom: 2px solid #2d2d5e;
        padding: 24px 32px;
        margin: -1rem -1rem 2rem -1rem;
    }
    .main-header h1 { color: white; font-size: 2rem; margin: 0; }
    .main-header p  { color: #8888aa; margin: 6px 0 0 0; font-size: 0.95rem; }

    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #2d2d5e; }

    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 12px 32px; font-size: 16px; font-weight: 600;
        width: 100%; cursor: pointer; transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.85; }
    hr { border-color: #2d2d5e; }

    .info-box {
        background: #111827; border: 1px solid #1f2d40;
        border-left: 4px solid #4f46e5;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 13px; color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Optimizador de Cartera · CEDEARs</h1>
    <p>Frontera Eficiente de Markowitz &nbsp;·&nbsp; Datos diarios en USD &nbsp;·&nbsp; Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR — PARÁMETROS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros")
    st.markdown("---")

    st.markdown("### 📌 Tickers de las acciones")
    tickers_input = st.text_area(
        "Separados por coma o por línea",
        value="AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, KO",
        height=110,
        help="Usá los tickers en USD de Yahoo Finance (acción subyacente del CEDEAR). Ej: AAPL, GOOGL, MELI"
    )

    st.markdown("### 📅 Período histórico")
    periodo = st.select_slider(
        "Años de historia a descargar",
        options=[1, 2, 3, 5, 7, 10],
        value=3,
        help="Se descargan precios de cierre diarios ajustados por dividendos y splits."
    )

    st.markdown("### 🎯 Rendimiento objetivo (opcional)")
    usar_target = st.checkbox(
        "Fijar rendimiento mínimo anual",
        value=False,
        help="El sistema buscará la cartera con menor riesgo posible que alcance ese rendimiento."
    )
    target_ret = None
    if usar_target:
        target_ret = st.slider(
            "Rendimiento mínimo anual (%)",
            min_value=0.0, max_value=60.0, value=15.0, step=0.5,
            format="%.1f%%"
        ) / 100.0

    st.markdown("### 🏦 Tasa libre de riesgo")
    rf = st.number_input(
        "Tasa anual (%)",
        min_value=0.0, max_value=20.0, value=4.3, step=0.1, format="%.1f",
        help="Referencia: T-Bills USA a 3 meses (~4.3% en 2025). Usada para calcular el Sharpe Ratio."
    ) / 100.0

    st.markdown("### 🎲 Simulaciones Monte Carlo")
    n_sim = st.select_slider(
        "Carteras a simular",
        options=[3000, 5000, 10000, 20000],
        value=10000,
        help="Más simulaciones = frontera más detallada, pero tarda más."
    )

    st.markdown("---")
    correr = st.button("🚀 Optimizar cartera", type="primary")

# ─────────────────────────────────────────
#  CONSTANTES Y FUNCIONES
# ─────────────────────────────────────────
DIAS = 252  # días hábiles por año

def metricas(pesos, mu, Sigma, rf):
    """Rendimiento, volatilidad y Sharpe de una cartera dado su vector de pesos."""
    pesos = np.array(pesos)
    ret   = np.dot(pesos, mu)
    vol   = np.sqrt(np.dot(pesos.T, np.dot(Sigma, pesos)))
    sharpe = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sharpe

def optimizar(fn_objetivo, mu, Sigma, rf, n, constraints_extra=None):
    """Optimización con reintentos si no converge en el primer intento."""
    cons   = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if constraints_extra:
        cons += constraints_extra
    bounds = [(0.0, 1.0)] * n
    x0     = np.array([1/n] * n)
    for _ in range(8):
        res = minimize(fn_objetivo, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            return res.x
        x0 = np.random.dirichlet(np.ones(n))
    return np.array([1/n] * n)

def tarjeta(titulo, color_class, ret, vol, sharpe):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{titulo}</div>
        <div class="metric-value {color_class}">{ret*100:.1f}%</div>
        <div class="metric-sub">Rendimiento esperado anual</div>
        <div style="margin-top:12px;display:flex;justify-content:space-around;">
            <div>
                <div class="metric-sub">Volatilidad</div>
                <div style="color:#ccccdd;font-size:20px;font-weight:700;">{vol*100:.1f}%</div>
            </div>
            <div>
                <div class="metric-sub">Sharpe</div>
                <div style="color:#ccccdd;font-size:20px;font-weight:700;">{sharpe:.3f}</div>
            </div>
        </div>
    </div>"""

# ─────────────────────────────────────────
#  PANTALLA INICIAL
# ─────────────────────────────────────────
if not correr:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#666688;">
        <div style="font-size:64px; margin-bottom:20px;">📈</div>
        <h2 style="color:#8888aa;">Configurá tu cartera en el panel izquierdo</h2>
        <p style="font-size:16px; max-width:520px; margin:12px auto;">
            Ingresá los tickers de las acciones subyacentes de tus CEDEARs,
            elegí cuántos años de datos históricos usar y presioná
            <strong style="color:#a855f7;">Optimizar cartera</strong>.
        </p>
        <div style="margin-top:28px; color:#444466; font-size:14px; line-height:2;">
            Tickers más comunes de CEDEARs:<br>
            AAPL · GOOGL · MSFT · AMZN · TSLA · META · NVDA<br>
            KO · DIS · JPM · GS · XOM · PFE · BABA · MELI · UBER
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────
#  1. PARSEAR TICKERS
# ─────────────────────────────────────────
tickers_raw = tickers_input.replace("\n", ",").split(",")
tickers = list(dict.fromkeys([t.strip().upper() for t in tickers_raw if t.strip()]))

if len(tickers) < 2:
    st.error("⚠️ Ingresá al menos 2 tickers para poder optimizar una cartera.")
    st.stop()

# ─────────────────────────────────────────
#  2. DESCARGA DE PRECIOS — ticker por ticker
#     (evita problemas con MultiIndex de yfinance)
# ─────────────────────────────────────────
fecha_fin = datetime.today()
fecha_ini = fecha_fin - timedelta(days=int(periodo * 365.25))

precios_dict  = {}
errores_dict  = {}

progress_bar  = st.progress(0, text="📥 Descargando datos...")

for idx, ticker in enumerate(tickers):
    try:
        tkr  = yf.Ticker(ticker)
        hist = tkr.history(
            start=fecha_ini.strftime("%Y-%m-%d"),
            end=fecha_fin.strftime("%Y-%m-%d"),
            interval="1d",          # datos diarios explícito
            auto_adjust=True,        # ajusta dividendos y splits
            actions=False,
        )

        if hist.empty or len(hist) < 50:
            errores_dict[ticker] = "sin datos suficientes"
        else:
            # Extraer solo el precio de cierre, índice en UTC naive
            serie = hist["Close"].copy()
            serie.index = pd.to_datetime(serie.index).tz_localize(None)
            serie.name  = ticker
            precios_dict[ticker] = serie

    except Exception as e:
        errores_dict[ticker] = str(e)

    progress_bar.progress((idx + 1) / len(tickers),
                          text=f"📥 Descargando {ticker}... ({idx+1}/{len(tickers)})")

progress_bar.empty()

# Armar DataFrame alineando fechas (outer join → rellenar huecos)
if not precios_dict:
    st.error("❌ No se pudo descargar datos para ningún ticker. Verificá los nombres.")
    st.stop()

precios = pd.DataFrame(precios_dict)
precios = precios.ffill().dropna()    # forward-fill días sin cotización y eliminar NaN

# Avisar sobre tickers que fallaron
tickers_ok  = list(precios.columns)
tickers_err = [t for t in tickers if t not in tickers_ok] + list(errores_dict.keys())
tickers_err = list(set(tickers_err))

if tickers_err:
    st.warning(f"⚠️ No se encontraron datos para: **{', '.join(tickers_err)}**. "
               f"Verificá que el ticker sea exactamente como aparece en Yahoo Finance.")

if len(tickers_ok) < 2:
    st.error("❌ No hay suficientes activos con datos válidos para optimizar.")
    st.stop()

# ─────────────────────────────────────────
#  3. CALCULAR RENDIMIENTOS DIARIOS
#     Usamos retornos simples para mostrar al usuario
#     y log-retornos para la optimización (más precisos)
# ─────────────────────────────────────────
ret_simples = precios.pct_change().dropna()          # % diario simple  (para mostrar)
ret_log     = np.log(precios / precios.shift(1)).dropna()  # log-retorno diario (para optimizar)

# Parámetros anualizados (basados en log-retornos)
mu_anual    = ret_log.mean() * DIAS
sigma_anual = ret_log.std()  * np.sqrt(DIAS)
cov_anual   = ret_log.cov()  * DIAS

mu    = mu_anual.values
Sigma = cov_anual.values
n     = len(tickers_ok)
pesos_eq = np.array([1/n] * n)

# ─────────────────────────────────────────
#  TABS DE RESULTADOS
# ─────────────────────────────────────────
tab_opt, tab_datos, tab_stats = st.tabs([
    "🎯 Optimización",
    "📋 Datos descargados",
    "📊 Estadísticas individuales",
])

# ══════════════════════════════════════════
#  TAB 2 — DATOS DESCARGADOS (verificación)
# ══════════════════════════════════════════
with tab_datos:
    st.markdown(f"""
    <div class="info-box">
        ✅ Se descargaron <strong>{len(precios):,} días hábiles</strong> de datos diarios
        desde <strong>{precios.index[0].strftime('%d/%m/%Y')}</strong>
        hasta <strong>{precios.index[-1].strftime('%d/%m/%Y')}</strong>
        para <strong>{len(tickers_ok)} activos</strong>: {', '.join(tickers_ok)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💵 Precios de cierre ajustados (USD)")
    st.markdown("*Ajustados por dividendos y splits. Últimos 20 días de negociación:*")
    st.dataframe(
        precios.tail(20).round(2).iloc[::-1],
        use_container_width=True
    )

    st.markdown("### 📈 Evolución de precios — base 100")
    st.markdown("*Normalizado a 100 en la fecha de inicio para comparar entre activos.*")
    precios_norm = precios / precios.iloc[0] * 100
    fig_precios  = go.Figure()
    colores_linea = px.colors.qualitative.Plotly
    for i, t in enumerate(tickers_ok):
        fig_precios.add_trace(go.Scatter(
            x=precios_norm.index, y=precios_norm[t].round(2),
            mode="lines", name=t,
            line=dict(width=1.8, color=colores_linea[i % len(colores_linea)]),
            hovertemplate=f"<b>{t}</b><br>%{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>",
        ))
    fig_precios.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=420,
        xaxis=dict(title="Fecha", gridcolor="#1e2030"),
        yaxis=dict(title="Precio base 100", gridcolor="#1e2030"),
        legend=dict(bgcolor="rgba(20,20,40,0.8)", bordercolor="#2d2d5e", font=dict(color="white")),
        margin=dict(l=50, r=20, t=20, b=50),
        hovermode="x unified",
    )
    st.plotly_chart(fig_precios, use_container_width=True)

    st.markdown("### 📉 Rendimientos diarios (%) — últimos 20 días")
    st.markdown("*Variación porcentual simple de cada día respecto al anterior.*")
    ret_mostrar = (ret_simples * 100).tail(20).round(4).iloc[::-1]
    st.dataframe(ret_mostrar.style.format("{:.2f}%"), use_container_width=True)

    st.markdown("### 📊 Distribución de rendimientos diarios")
    fig_hist = go.Figure()
    for i, t in enumerate(tickers_ok):
        fig_hist.add_trace(go.Histogram(
            x=(ret_simples[t] * 100).round(3),
            name=t, opacity=0.6, nbinsx=60,
            marker_color=colores_linea[i % len(colores_linea)],
            hovertemplate=f"<b>{t}</b><br>Rend: %{{x:.2f}}%<br>Días: %{{y}}<extra></extra>",
        ))
    fig_hist.update_layout(
        barmode="overlay", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=360,
        xaxis=dict(title="Rendimiento diario (%)", ticksuffix="%", gridcolor="#1e2030"),
        yaxis=dict(title="Cantidad de días", gridcolor="#1e2030"),
        legend=dict(bgcolor="rgba(20,20,40,0.8)", font=dict(color="white")),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 3 — ESTADÍSTICAS INDIVIDUALES
# ══════════════════════════════════════════
with tab_stats:
    # Tabla resumen
    df_stats = pd.DataFrame({
        "Rendimiento anual (%)": (mu_anual * 100).round(2),
        "Volatilidad anual (%)": (sigma_anual * 100).round(2),
        "Sharpe individual":     ((mu_anual - rf) / sigma_anual).round(3),
        "Rend. diario prom. (%)": (ret_simples.mean() * 100).round(4),
        "Vol. diaria (%)":        (ret_simples.std()  * 100).round(4),
        "Máx. caída diaria (%)":  (ret_simples.min()  * 100).round(3),
        "Máx. suba diaria (%)":   (ret_simples.max()  * 100).round(3),
        "Días de datos":           ret_simples.count(),
    })
    st.markdown("### 📋 Resumen estadístico por activo")
    st.dataframe(df_stats, use_container_width=True)

    st.markdown("### 🗺️ Riesgo vs Rendimiento individual")
    fig_rv = go.Figure()
    for i, t in enumerate(tickers_ok):
        fig_rv.add_trace(go.Scatter(
            x=[sigma_anual[t] * 100], y=[mu_anual[t] * 100],
            mode="markers+text",
            marker=dict(size=16, color=colores_linea[i % len(colores_linea)],
                        line=dict(color="white", width=1.2)),
            text=[t], textposition="top right", textfont=dict(color="white", size=11),
            name=t,
            hovertemplate=f"<b>{t}</b><br>Vol: {sigma_anual[t]*100:.1f}%<br>Rend: {mu_anual[t]*100:.1f}%<extra></extra>",
        ))
    fig_rv.add_hline(y=0, line_dash="dash", line_color="#666688", opacity=0.6)
    fig_rv.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420,
        xaxis=dict(title="Volatilidad Anual (%)", ticksuffix="%", gridcolor="#1e2030"),
        yaxis=dict(title="Rendimiento Anual (%)", ticksuffix="%", gridcolor="#1e2030"),
        showlegend=False, margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown("### 🌡️ Matriz de correlación")
    corr = ret_log.corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=tickers_ok, y=tickers_ok,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont={"size": 11}, hoverongaps=False,
    ))
    fig_heat.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=max(320, 65 * n), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickfont=dict(color="white")),
        yaxis=dict(tickfont=dict(color="white")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Verde = correlación alta (se mueven juntos). Rojo = correlación negativa (se mueven en sentido opuesto). Activos con baja correlación entre sí mejoran la diversificación.")


# ══════════════════════════════════════════
#  TAB 1 — OPTIMIZACIÓN  (se ejecuta aquí)
# ══════════════════════════════════════════
with tab_opt:

    # ── Optimización ──────────────────────
    with st.spinner("⚙️ Optimizando carteras..."):
        pesos_mv = optimizar(lambda w: metricas(w, mu, Sigma, rf)[1],  mu, Sigma, rf, n)
        pesos_ms = optimizar(lambda w: -metricas(w, mu, Sigma, rf)[2], mu, Sigma, rf, n)
        ret_mv, vol_mv, sh_mv = metricas(pesos_mv, mu, Sigma, rf)
        ret_ms, vol_ms, sh_ms = metricas(pesos_ms, mu, Sigma, rf)
        ret_eq, vol_eq, sh_eq = metricas(pesos_eq, mu, Sigma, rf)

        pesos_tg = None
        if usar_target and target_ret is not None:
            cons_tg  = [{"type": "ineq",
                         "fun": lambda w, t=target_ret: metricas(w, mu, Sigma, rf)[0] - t}]
            pesos_tg = optimizar(lambda w: metricas(w, mu, Sigma, rf)[1],
                                 mu, Sigma, rf, n, constraints_extra=cons_tg)
            ret_tg, vol_tg, sh_tg = metricas(pesos_tg, mu, Sigma, rf)
            if ret_tg < target_ret - 0.01:
                st.warning(f"⚠️ Con los activos seleccionados el rendimiento máximo alcanzable "
                           f"es ~{ret_tg*100:.1f}%. Se muestra la cartera de mínimo riesgo con ese resultado.")

    # ── Monte Carlo ───────────────────────
    with st.spinner(f"🎲 Simulando {n_sim:,} carteras..."):
        np.random.seed(42)
        sim_ret = np.zeros(n_sim)
        sim_vol = np.zeros(n_sim)
        sim_sh  = np.zeros(n_sim)
        for i in range(n_sim):
            w = np.random.dirichlet(np.ones(n))
            sim_ret[i], sim_vol[i], sim_sh[i] = metricas(w, mu, Sigma, rf)

    # ── Frontera Eficiente ─────────────────
    with st.spinner("📐 Calculando frontera eficiente..."):
        fe_rets, fe_vols = [], []
        for t in np.linspace(ret_mv, sim_ret.max() * 1.02, 80):
            cons_fe = [
                {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
                {"type": "eq",  "fun": lambda w, t=t: metricas(w, mu, Sigma, rf)[0] - t},
            ]
            res_fe = minimize(lambda w: metricas(w, mu, Sigma, rf)[1], pesos_eq,
                              method="SLSQP", bounds=[(0,1)]*n, constraints=cons_fe,
                              options={"ftol": 1e-10, "maxiter": 500})
            if res_fe.success:
                _, v_fe, _ = metricas(res_fe.x, mu, Sigma, rf)
                fe_vols.append(v_fe * 100)
                fe_rets.append(t * 100)

    # ── Tarjetas de métricas ───────────────
    st.markdown(f"#### {len(tickers_ok)} activos · {periodo} año{'s' if periodo>1 else ''} de datos · {len(precios):,} días hábiles")

    carteras_graf = [
        ("🛡️ Mínima Varianza",  "color-minvar", pesos_mv, ret_mv, vol_mv, sh_mv),
        ("🚀 Máximo Sharpe",    "color-sharpe", pesos_ms, ret_ms, vol_ms, sh_ms),
        ("⚖️ Equitativa",       "color-equal",  pesos_eq, ret_eq, vol_eq, sh_eq),
    ]
    if pesos_tg is not None:
        carteras_graf.append((f"🎯 Objetivo ≥{target_ret*100:.0f}%", "color-target",
                               pesos_tg, ret_tg, vol_tg, sh_tg))

    cols = st.columns(len(carteras_graf))
    for col, (nombre, cls, _, ret, vol, sh) in zip(cols, carteras_graf):
        with col:
            st.markdown(tarjeta(nombre, cls, ret, vol, sh), unsafe_allow_html=True)

    st.markdown("---")

    # ── Frontera Eficiente (gráfico) ───────
    st.markdown("### 📈 Frontera Eficiente de Markowitz")

    fig = go.Figure()

    # Nube Monte Carlo
    fig.add_trace(go.Scatter(
        x=sim_vol*100, y=sim_ret*100, mode="markers",
        marker=dict(color=sim_sh, colorscale="Plasma", size=4, opacity=0.35,
                    colorbar=dict(title="Sharpe", tickfont=dict(color="white"),
                                  title_font=dict(color="white")), showscale=True),
        name="Carteras simuladas",
        hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra></extra>",
    ))

    # Línea frontera
    if fe_vols:
        fig.add_trace(go.Scatter(
            x=fe_vols, y=fe_rets, mode="lines",
            line=dict(color="#00ffcc", width=2.5), name="Frontera Eficiente",
            hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra>Frontera</extra>",
        ))

    # CML
    vol_cml = np.linspace(0, max(sim_vol)*110, 200)
    fig.add_trace(go.Scatter(
        x=vol_cml, y=rf*100 + sh_ms*vol_cml, mode="lines",
        line=dict(color="#ffdd44", width=1.5, dash="dash"),
        name="Capital Market Line (CML)",
    ))

    # Rf
    fig.add_trace(go.Scatter(
        x=[0], y=[rf*100], mode="markers+text",
        marker=dict(color="#ffdd44", size=12, symbol="diamond"),
        text=["Rf"], textposition="top right", textfont=dict(color="#ffdd44"),
        name=f"Tasa libre de riesgo ({rf*100:.1f}%)",
    ))

    # Activos individuales
    for i, t in enumerate(tickers_ok):
        fig.add_trace(go.Scatter(
            x=[sigma_anual[t]*100], y=[mu_anual[t]*100],
            mode="markers+text",
            marker=dict(color="white", size=8, symbol="circle"),
            text=[t], textposition="top right", textfont=dict(color="#aaaacc", size=10),
            showlegend=False,
            hovertemplate=f"<b>{t}</b><br>Vol: {sigma_anual[t]*100:.1f}%<br>Rend: {mu_anual[t]*100:.1f}%<extra></extra>",
        ))

    # Puntos de carteras óptimas
    iconos = {"🛡️ Mínima Varianza": ("diamond","#00ccff"),
              "🚀 Máximo Sharpe":   ("star",   "#ff6b35"),
              "⚖️ Equitativa":      ("triangle-up","#aaffaa")}
    for nombre, _, _, ret, vol, sh in carteras_graf:
        sym, col_pt = iconos.get(nombre, ("pentagon","#a855f7"))
        fig.add_trace(go.Scatter(
            x=[vol*100], y=[ret*100], mode="markers+text",
            marker=dict(color=col_pt, size=18, symbol=sym, line=dict(color="white",width=1.5)),
            text=[nombre.split(" ",1)[1]], textposition="top right",
            textfont=dict(color=col_pt, size=11), name=nombre,
            hovertemplate=f"<b>{nombre}</b><br>Vol: {vol*100:.1f}%<br>Rend: {ret*100:.1f}%<br>Sharpe: {sh:.3f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=560,
        xaxis=dict(title="Volatilidad Anual — Riesgo (%)", ticksuffix="%", gridcolor="#1e2030"),
        yaxis=dict(title="Rendimiento Esperado Anual (%)", ticksuffix="%", gridcolor="#1e2030"),
        legend=dict(bgcolor="rgba(20,20,40,0.8)", bordercolor="#2d2d5e",
                    font=dict(color="white", size=11)),
        margin=dict(l=60, r=20, t=20, b=60),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tortas de composición ──────────────
    st.markdown("---")
    st.markdown("### 🥧 Composición de carteras")
    colores_pie = px.colors.qualitative.Set3
    cols_pie    = st.columns(len(carteras_graf))

    for col, (nombre, _, pesos, ret, vol, sh) in zip(cols_pie, carteras_graf):
        with col:
            _, color_pt = iconos.get(nombre, ("pentagon","#a855f7"))
            fig_pie = go.Figure(go.Pie(
                labels=tickers_ok,
                values=[round(p*100, 2) for p in pesos],
                hole=0.45,
                marker=dict(colors=colores_pie[:n], line=dict(color="#0e1117", width=2)),
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>",
            ))
            fig_pie.update_layout(
                title=dict(
                    text=f"<b style='color:{color_pt}'>{nombre}</b><br>"
                         f"<span style='font-size:11px;color:#888'>{ret*100:.1f}% rend · "
                         f"{vol*100:.1f}% vol · Sharpe {sh:.3f}</span>",
                    x=0.5),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                showlegend=True,
                legend=dict(font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0)"),
                height=360, margin=dict(l=10, r=10, t=70, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Tabla de pesos ─────────────────────
    st.markdown("---")
    st.markdown("### 📋 Pesos por activo")

    tabla_pesos = {"Ticker": tickers_ok}
    for nombre, _, pesos, _, _, _ in carteras_graf:
        tabla_pesos[nombre.split(" ",1)[1]] = [f"{p*100:.2f}%" for p in pesos]
    tabla_pesos["Rend. anual"] = [f"{mu_anual[t]*100:.1f}%" for t in tickers_ok]
    tabla_pesos["Vol. anual"]  = [f"{sigma_anual[t]*100:.1f}%" for t in tickers_ok]

    df_tabla = pd.DataFrame(tabla_pesos).set_index("Ticker")
    st.dataframe(df_tabla, use_container_width=True, height=min(60 + 36*n, 480))

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555577; font-size:13px; padding:16px 0;">
    ⚠️ <strong>Advertencia:</strong> Este modelo usa datos históricos en USD.
    Los rendimientos pasados no garantizan rendimientos futuros.<br>
    El modelo de Markowitz asume retornos normalmente distribuidos y covarianzas estables en el tiempo.<br><br>
    Construido con <strong>Streamlit</strong> · Datos de <strong>Yahoo Finance</strong>
</div>
""", unsafe_allow_html=True)
