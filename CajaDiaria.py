import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Configuración ---
st.set_page_config(page_title="Análisis de Caja", layout="wide")

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_excel("BaseCaja.xlsx")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    # Total en USD Eq.
    df["Total_USD_Eq"] = (df["Caja ML"] + df["Inversión ML"]) / df["T.C."] + (df["Caja USD"] + df["Inversión USD"])
    # % de USD vs ML
    df["USD_pct"] = (df["Caja USD"] + df["Inversión USD"]) / df["Total_USD_Eq"]
    df["ML_pct"] = 1 - df["USD_pct"]
    return df

df = load_data()

# --- Filtros ---
st.sidebar.header("Filtros")
modo = st.sidebar.radio("¿Qué quieres ver?", ["Consolidado", "Por país"])

paises = df["País"].unique()
pais_sel = st.sidebar.multiselect("Selecciona país(es):", paises, default=paises)

fecha_min, fecha_max = df["Fecha"].min(), df["Fecha"].max()
rango_fechas = st.sidebar.date_input("Rango de fechas:", [fecha_min, fecha_max])

df_filtrado = df[
    (df["País"].isin(pais_sel)) &
    (df["Fecha"].between(pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])))
]

# --- Consolidado vs País ---
if modo == "Consolidado":
    serie = df_filtrado.groupby("Fecha")["Total_USD_Eq"].sum().reset_index()
else:
    serie = df_filtrado.groupby(["Fecha", "País"])["Total_USD_Eq"].sum().reset_index()

# --- Serie de tiempo Total USD Eq. ---
st.subheader("📈 Serie de tiempo de Caja Total (USD Eq.)")
fig, ax = plt.subplots(figsize=(12, 6))

if modo == "Consolidado":
    ax.plot(serie["Fecha"], serie["Total_USD_Eq"], label="Consolidado")
else:
    for pais, subdf in serie.groupby("País"):
        ax.plot(subdf["Fecha"], subdf["Total_USD_Eq"], label=pais)

ax.legend()
ax.set_ylabel("USD Equivalente")
ax.set_xlabel("Fecha")
st.pyplot(fig)

# --- Proporción USD vs ML ---
st.subheader("💱 Proporción USD vs Moneda Local")
if modo == "Consolidado":
    prop = df_filtrado.groupby("Fecha")[["USD_pct", "ML_pct"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(prop["Fecha"], prop["USD_pct"], prop["ML_pct"], labels=["USD", "ML"])
    ax.legend(loc="upper left")
    st.pyplot(fig)
else:
    st.info("Selecciona 'Consolidado' para ver proporción global USD vs ML.")

# --- Patrones de entradas/salidas ---
st.subheader("🔄 Variaciones diarias")
if modo == "Consolidado":
    serie["Var_Diaria"] = serie["Total_USD_Eq"].diff()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(serie["Fecha"], serie["Var_Diaria"])
    ax.axhline(0, color="black", linewidth=1)
    st.pyplot(fig)

    # Alertas automáticas
    std = serie["Var_Diaria"].std()
    mean = serie["Var_Diaria"].mean()
    umbral = 2 * std
    outliers = serie[abs(serie["Var_Diaria"] - mean) > umbral]

    if not outliers.empty:
        st.warning("⚠️ Movimientos inusuales detectados:")
        for _, row in outliers.iterrows():
            if row["Var_Diaria"] > 0:
                st.write(f"📈 {row['Fecha'].date()} - Entrada fuerte de {row['Var_Diaria']:,.2f} USD")
            else:
                st.write(f"📉 {row['Fecha'].date()} - Salida fuerte de {row['Var_Diaria']:,.2f} USD")
    else:
        st.info("No se detectaron movimientos fuera de lo normal.")
else:
    st.info("Selecciona 'Consolidado' para ver patrones globales de entradas/salidas.")

# --- Proyección de cajas (regresión lineal simple) ---
st.subheader("📊 Proyección de Caja (Regresión Lineal)")
horizonte = st.slider("Selecciona días a proyectar:", 5, 60, 15)

if modo == "Consolidado":
    serie = serie.dropna()
    X = np.arange(len(serie)).reshape(-1, 1)
    y = serie["Total_USD_Eq"].values
    model = LinearRegression().fit(X, y)

    # Proyección
    future_X = np.arange(len(serie), len(serie)+horizonte).reshape(-1, 1)
    future_pred = model.predict(future_X)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie["Fecha"], y, label="Histórico")
    future_dates = pd.date_range(start=serie["Fecha"].iloc[-1] + pd.Timedelta(days=1), periods=horizonte)
    ax.plot(future_dates, future_pred, label="Proyección", linestyle="--")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Selecciona 'Consolidado' para ver proyección global de caja.")
