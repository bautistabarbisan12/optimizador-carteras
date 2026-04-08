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
    /* Fondo general */
    .stApp { background-color: #0e1117; }

    /* Tarjetas de métricas */
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

    /* Colores por cartera */
    .color-minvar  { color: #00ccff; }
    .color-sharpe  { color: #ff6b35; }
    .color-target  { color: #a855f7; }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0f0f2e 0%, #1a0a3e 100%);
        border-bottom: 2px solid #2d2d5e;
        padding: 24px 32px;
        margin: -1rem -1rem 2rem -1rem;
    }
    .main-header h1 { color: white; font-size: 2rem; margin: 0; }
    .main-header p  { color: #8888aa; margin: 6px 0 0 0; font-size: 0.95rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #2d2d5e; }

    /* Tablas */
    .dataframe thead th { background-color: #1a1a2e !important; color: white !important; }
    .dataframe tbody td { background-color: #111122 !important; color: #ccccdd !important; }

    /* Botón principal */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.85; }

    /* Separador */
    hr { border-color: #2d2d5e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Optimizador de Cartera · CEDEARs</h1>
    <p>Frontera Eficiente de Markowitz · Precios en USD · Datos de Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR — PARÁMETROS
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    st.markdown("### 📌 Tickers")
    tickers_input = st.text_area(
        "Ingresá los tickers separados por comas",
        value="AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, KO",
        height=100,
        help="Usá los tickers de Yahoo Finance (acciones en USD). Ej: AAPL, GOOGL, MSFT"
    )

    st.markdown("### 📅 Período histórico")
    periodo = st.select_slider(
        "Años de datos históricos",
        options=[1, 2, 3, 5, 7, 10],
        value=3,
        help="Más años = más historia estadística. Menos años = captura mejor el contexto actual."
    )

    st.markdown("### 🎯 Rendimiento mínimo objetivo")
    usar_target = st.checkbox("Fijar rendimiento mínimo", value=False,
                               help="Si lo activás, el sistema busca la cartera de menor riesgo que alcance ese rendimiento mínimo.")
    target_ret = None
    if usar_target:
        target_ret = st.slider(
            "Rendimiento anual mínimo (%)",
            min_value=0.0, max_value=60.0, value=15.0, step=0.5,
            format="%.1f%%"
        ) / 100.0

    st.markdown("### 🏦 Tasa libre de riesgo")
    rf = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0, max_value=20.0, value=4.3, step=0.1,
        format="%.1f",
        help="Referencia: T-Bills USA a 3 meses. Actualmente ~4.3% anual."
    ) / 100.0

    st.markdown("### 🎲 Simulaciones")
    n_sim = st.select_slider(
        "Carteras a simular",
        options=[3000, 5000, 10000, 20000],
        value=10000,
        help="Más simulaciones = frontera más precisa, pero tarda un poco más."
    )

    st.markdown("---")
    correr = st.button("🚀 Optimizar cartera", type="primary")

# ─────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────
DIAS = 252

def metricas(pesos, mu, Sigma, rf):
    pesos = np.array(pesos)
    ret = np.dot(pesos, mu)
    vol = np.sqrt(np.dot(pesos.T, np.dot(Sigma, pesos)))
    sharpe = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sharpe

def optimizar(objetivo_fn, mu, Sigma, rf, n, constraints_extra=None):
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if constraints_extra:
        cons += constraints_extra
    bounds = [(0.0, 1.0)] * n
    x0 = np.array([1/n] * n)
    for _ in range(5):
        res = minimize(objetivo_fn, x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
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
        <div style="margin-top:10px; display:flex; justify-content:space-around;">
            <div>
                <div class="metric-sub">Volatilidad</div>
                <div style="color:#ccccdd; font-size:18px; font-weight:600;">{vol*100:.1f}%</div>
            </div>
            <div>
                <div class="metric-sub">Sharpe</div>
                <div style="color:#ccccdd; font-size:18px; font-weight:600;">{sharpe:.3f}</div>
            </div>
        </div>
    </div>"""

# ─────────────────────────────────────────
#  PANTALLA INICIAL (sin ejecutar)
# ─────────────────────────────────────────
if not correr:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #666688;">
        <div style="font-size: 64px; margin-bottom: 20px;">📈</div>
        <h2 style="color: #8888aa;">Configurá tu cartera en el panel izquierdo</h2>
        <p style="font-size: 16px; max-width: 500px; margin: 12px auto;">
            Ingresá los tickers de las acciones subyacentes de tus CEDEARs,
            ajustá los parámetros y presioná <strong style="color:#a855f7;">Optimizar cartera</strong>.
        </p>
        <div style="margin-top: 32px; color: #555577; font-size: 14px;">
            Ejemplo de tickers: AAPL · GOOGL · MSFT · AMZN · TSLA · META · NVDA · KO · DIS · JPM
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────
#  PROCESAMIENTO
# ─────────────────────────────────────────
# Parsear tickers
tickers = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # Eliminar duplicados

if len(tickers) < 2:
    st.error("⚠️ Necesitás al menos 2 tickers para optimizar una cartera.")
    st.stop()

# Descarga de datos
fecha_fin   = datetime.today()
fecha_ini   = fecha_fin - timedelta(days=periodo * 365)

with st.spinner(f"📥 Descargando precios de {len(tickers)} activos ({periodo} años)..."):
    try:
        raw = yf.download(
            tickers, start=fecha_ini.strftime("%Y-%m-%d"),
            end=fecha_fin.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False
        )
        if len(tickers) == 1:
            precios = raw[["Close"]]
            precios.columns = tickers
        else:
            precios = raw["Close"]

        precios = precios.ffill().dropna()
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        st.stop()

# Validar que todos los tickers tienen datos
tickers_ok  = [t for t in tickers if t in precios.columns and precios[t].notna().sum() > 50]
tickers_err = [t for t in tickers if t not in tickers_ok]

if tickers_err:
    st.warning(f"⚠️ No se encontraron datos para: **{', '.join(tickers_err)}**. Verificá que el ticker sea correcto en Yahoo Finance.")

if len(tickers_ok) < 2:
    st.error("No hay suficientes activos con datos válidos.")
    st.stop()

precios = precios[tickers_ok]

# Rendimientos y parámetros
rendimientos    = np.log(precios / precios.shift(1)).dropna()
mu              = (rendimientos.mean() * DIAS).values
Sigma           = (rendimientos.cov()  * DIAS).values
vol_ind         = rendimientos.std() * np.sqrt(DIAS)
n               = len(tickers_ok)
pesos_eq        = np.array([1/n] * n)

# ─────────────────────────────────────────
#  OPTIMIZACIÓN
# ─────────────────────────────────────────
with st.spinner("⚙️ Optimizando carteras..."):

    # Mínima Varianza
    pesos_mv = optimizar(
        lambda w: metricas(w, mu, Sigma, rf)[1],
        mu, Sigma, rf, n
    )

    # Máximo Sharpe
    pesos_ms = optimizar(
        lambda w: -metricas(w, mu, Sigma, rf)[2],
        mu, Sigma, rf, n
    )

    # Cartera con rendimiento mínimo objetivo
    pesos_tg = None
    if usar_target and target_ret is not None:
        cons_tg = [{"type": "ineq", "fun": lambda w, t=target_ret: metricas(w, mu, Sigma, rf)[0] - t}]
        pesos_tg = optimizar(
            lambda w: metricas(w, mu, Sigma, rf)[1],
            mu, Sigma, rf, n,
            constraints_extra=cons_tg
        )
        ret_tg, vol_tg, sh_tg = metricas(pesos_tg, mu, Sigma, rf)
        if ret_tg < target_ret - 0.005:
            st.warning(f"⚠️ Con los activos seleccionados no es posible alcanzar un rendimiento de {target_ret*100:.1f}% sin incrementar el riesgo significativamente. Se muestra la cartera de máximo rendimiento alcanzable.")

    # Métricas de las carteras
    ret_mv, vol_mv, sh_mv = metricas(pesos_mv, mu, Sigma, rf)
    ret_ms, vol_ms, sh_ms = metricas(pesos_ms, mu, Sigma, rf)
    ret_eq, vol_eq, sh_eq = metricas(pesos_eq, mu, Sigma, rf)

# ─────────────────────────────────────────
#  SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────
with st.spinner(f"🎲 Simulando {n_sim:,} carteras..."):
    np.random.seed(42)
    sim_ret = np.zeros(n_sim)
    sim_vol = np.zeros(n_sim)
    sim_sh  = np.zeros(n_sim)

    for i in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = metricas(w, mu, Sigma, rf)
        sim_ret[i] = r
        sim_vol[i] = v
        sim_sh[i]  = s

# ─────────────────────────────────────────
#  FRONTERA EFICIENTE (matemática)
# ─────────────────────────────────────────
with st.spinner("📐 Calculando frontera eficiente..."):
    fe_rets = []
    fe_vols = []
    targets_fe = np.linspace(ret_mv, sim_ret.max() * 1.02, 80)

    for t in targets_fe:
        cons_fe = [
            {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
            {"type": "eq",  "fun": lambda w, t=t: metricas(w, mu, Sigma, rf)[0] - t},
        ]
        bounds_fe = [(0, 1)] * n
        res_fe = minimize(
            lambda w: metricas(w, mu, Sigma, rf)[1],
            pesos_eq, method="SLSQP",
            bounds=bounds_fe, constraints=cons_fe,
            options={"ftol": 1e-10, "maxiter": 500}
        )
        if res_fe.success:
            _, v_fe, _ = metricas(res_fe.x, mu, Sigma, rf)
            fe_vols.append(v_fe * 100)
            fe_rets.append(t * 100)

# ─────────────────────────────────────────
#  RESULTADOS — TARJETAS DE MÉTRICAS
# ─────────────────────────────────────────
st.markdown("## 🎯 Resultados de optimización")
st.markdown(f"**{len(tickers_ok)} activos** · **{periodo} años** de historia · **{n_sim:,} simulaciones**")

if pesos_tg is not None:
    c1, c2, c3, c4 = st.columns(4)
else:
    c1, c2, c3 = st.columns(3)
    c4 = None

with c1:
    st.markdown(tarjeta("🛡️ Mínima Varianza", "color-minvar", ret_mv, vol_mv, sh_mv), unsafe_allow_html=True)
with c2:
    st.markdown(tarjeta("🚀 Máximo Sharpe", "color-sharpe", ret_ms, vol_ms, sh_ms), unsafe_allow_html=True)
with c3:
    st.markdown(tarjeta("⚖️ Equitativa", "color-target", ret_eq, vol_eq, sh_eq), unsafe_allow_html=True)
if c4 and pesos_tg is not None:
    with c4:
        st.markdown(tarjeta(f"🎯 Objetivo ≥{target_ret*100:.0f}%", "color-target", ret_tg, vol_tg, sh_tg), unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
#  GRÁFICO: FRONTERA EFICIENTE
# ─────────────────────────────────────────
st.markdown("## 📈 Frontera Eficiente de Markowitz")

fig = go.Figure()

# Nubes de puntos simulados
fig.add_trace(go.Scatter(
    x=sim_vol * 100, y=sim_ret * 100,
    mode="markers",
    marker=dict(
        color=sim_sh, colorscale="Plasma",
        size=4, opacity=0.4,
        colorbar=dict(title="Sharpe Ratio", tickfont=dict(color="white"), title_font=dict(color="white")),
        showscale=True,
    ),
    name="Carteras simuladas",
    hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra></extra>",
))

# Línea frontera eficiente
if fe_vols:
    fig.add_trace(go.Scatter(
        x=fe_vols, y=fe_rets,
        mode="lines",
        line=dict(color="#00ffcc", width=2.5),
        name="Frontera Eficiente",
        hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra>Frontera Eficiente</extra>",
    ))

# CML (Capital Market Line)
vol_cml = np.linspace(0, max(sim_vol)*100 * 1.1, 200)
ret_cml = rf * 100 + sh_ms * vol_cml
fig.add_trace(go.Scatter(
    x=vol_cml, y=ret_cml,
    mode="lines",
    line=dict(color="#ffdd44", width=1.5, dash="dash"),
    name="Línea de Mercado de Capitales (CML)",
    hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra>CML</extra>",
))

# Tasa libre de riesgo
fig.add_trace(go.Scatter(
    x=[0], y=[rf * 100],
    mode="markers+text",
    marker=dict(color="#ffdd44", size=12, symbol="diamond"),
    text=["Rf"], textposition="top right", textfont=dict(color="#ffdd44"),
    name=f"Tasa libre de riesgo ({rf*100:.1f}%)",
    hovertemplate=f"Tasa libre de riesgo: {rf*100:.1f}%<extra></extra>",
))

# Activos individuales
for i, t in enumerate(tickers_ok):
    fig.add_trace(go.Scatter(
        x=[vol_ind[t] * 100], y=[rendimientos.mean()[t] * DIAS * 100],
        mode="markers+text",
        marker=dict(color="white", size=9, symbol="circle"),
        text=[t], textposition="top right", textfont=dict(color="#aaaacc", size=10),
        name=t, showlegend=False,
        hovertemplate=f"<b>{t}</b><br>Vol: {vol_ind[t]*100:.1f}%<br>Rend: {rendimientos.mean()[t]*DIAS*100:.1f}%<extra></extra>",
    ))

# Carteras óptimas
puntos = [
    (vol_mv*100, ret_mv*100, "🛡️ Mín. Varianza", "#00ccff", "diamond"),
    (vol_ms*100, ret_ms*100, "🚀 Máx. Sharpe",   "#ff6b35", "star"),
    (vol_eq*100, ret_eq*100, "⚖️ Equitativa",    "#aaffaa", "triangle-up"),
]
if pesos_tg is not None:
    puntos.append((vol_tg*100, ret_tg*100, f"🎯 Objetivo", "#a855f7", "pentagon"))

for vp, rp, nombre, color, symbol in puntos:
    fig.add_trace(go.Scatter(
        x=[vp], y=[rp],
        mode="markers+text",
        marker=dict(color=color, size=16, symbol=symbol, line=dict(color="white", width=1.5)),
        text=[nombre.split(" ", 1)[1]], textposition="top right",
        textfont=dict(color=color, size=11),
        name=nombre,
        hovertemplate=f"<b>{nombre}</b><br>Vol: {vp:.1f}%<br>Rend: {rp:.1f}%<extra></extra>",
    ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    height=560,
    xaxis=dict(title="Volatilidad Anual — Riesgo (%)", ticksuffix="%", gridcolor="#1e2030"),
    yaxis=dict(title="Rendimiento Esperado Anual (%)", ticksuffix="%", gridcolor="#1e2030"),
    legend=dict(
        bgcolor="rgba(20,20,40,0.8)", bordercolor="#2d2d5e",
        font=dict(color="white", size=11),
    ),
    margin=dict(l=60, r=20, t=20, b=60),
    hoverlabel=dict(bgcolor="#1a1a2e", font_size=12),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
#  GRÁFICO: COMPOSICIÓN DE CARTERAS
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 🥧 Composición de carteras")

carteras_graf = [
    ("🛡️ Mínima Varianza",  pesos_mv, "#00ccff"),
    ("🚀 Máximo Sharpe",    pesos_ms, "#ff6b35"),
    ("⚖️ Equitativa",       pesos_eq, "#aaffaa"),
]
if pesos_tg is not None:
    carteras_graf.append((f"🎯 Objetivo ≥{target_ret*100:.0f}%", pesos_tg, "#a855f7"))

n_cols = len(carteras_graf)
cols_pie = st.columns(n_cols)

colores_pie = px.colors.qualitative.Set3

for col, (nombre, pesos, color) in zip(cols_pie, carteras_graf):
    with col:
        fig_pie = go.Figure(go.Pie(
            labels=tickers_ok,
            values=[round(p * 100, 2) for p in pesos],
            hole=0.45,
            marker=dict(colors=colores_pie[:n], line=dict(color="#0e1117", width=2)),
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>",
        ))
        ret_c, vol_c, sh_c = metricas(pesos, mu, Sigma, rf)
        fig_pie.update_layout(
            title=dict(text=f"<b style='color:{color}'>{nombre}</b><br><span style='font-size:11px;color:#888'>{ret_c*100:.1f}% rend · {vol_c*100:.1f}% vol · Sharpe {sh_c:.3f}</span>", x=0.5),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            showlegend=True,
            legend=dict(font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0)"),
            height=350,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────
#  TABLA DE PESOS DETALLADA
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📋 Pesos por activo")

tabla_pesos = {"Ticker": tickers_ok}
for nombre, pesos, _ in carteras_graf:
    nombre_limpio = nombre.split(" ", 1)[1]
    tabla_pesos[nombre_limpio] = [f"{p*100:.2f}%" for p in pesos]

tabla_pesos["Rend. Individual"] = [f"{rendimientos.mean()[t]*DIAS*100:.1f}%" for t in tickers_ok]
tabla_pesos["Vol. Individual"]  = [f"{vol_ind[t]*100:.1f}%" for t in tickers_ok]

df_tabla = pd.DataFrame(tabla_pesos).set_index("Ticker")
st.dataframe(df_tabla, use_container_width=True, height=min(40 + 36*n, 450))

# ─────────────────────────────────────────
#  ESTADÍSTICAS INDIVIDUALES
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Estadísticas individuales")

st.markdown("### Matriz de correlación")
corr = rendimientos.corr()

fig_heat = go.Figure(go.Heatmap(
    z=corr.values,
    x=tickers_ok, y=tickers_ok,
    colorscale="RdYlGn",
    zmin=-1, zmax=1,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    textfont={"size": 11},
    hoverongaps=False,
))
fig_heat.update_layout(
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    height=max(300, 60 * n),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(tickfont=dict(color="white")),
    yaxis=dict(tickfont=dict(color="white")),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555577; font-size:13px; padding: 16px 0;">
    ⚠️ <strong>Advertencia:</strong> Este modelo usa datos históricos. Los rendimientos pasados no garantizan rendimientos futuros.<br>
    El modelo de Markowitz asume retornos normalmente distribuidos y covarianzas estables en el tiempo.<br><br>
    Construido con <strong>Streamlit</strong> · Datos de <strong>Yahoo Finance</strong>
</div>
""", unsafe_allow_html=True)
