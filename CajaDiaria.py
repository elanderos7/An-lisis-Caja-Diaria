import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Configuración ---
st.set_page_config(page_title="Análisis de Caja", layout="wide")

st.title("📊 Análisis de Caja Diaria")
st.write("Sube tu archivo **BaseCaja.xlsx** para comenzar el análisis.")

# --- Subir archivo ---
uploaded_file = st.file_uploader("📂 Sube tu archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # --- Cargar datos ---
    df = pd.read_excel(uploaded_file)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Total_USD_Eq"] = (df["Caja ML"] + df["Inversión ML"]) / df["T.C."] + (df["Caja USD"] + df["Inversión USD"])
    df["USD_pct"] = (df["Caja USD"] + df["Inversión USD"]) / df["Total_USD_Eq"]
    df["ML_pct"] = 1 - df["USD_pct"]

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
    if modo == "Consolidado":
        fig = px.line(serie, x="Fecha", y="Total_USD_Eq", title="Caja Total Consolidada (USD Eq.)")
    else:
        fig = px.line(serie, x="Fecha", y="Total_USD_Eq", color="País", title="Caja Total por País (USD Eq.)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Proporción USD vs ML ---
    st.subheader("💱 Proporción USD vs Moneda Local")
    if modo == "Consolidado":
        prop = df_filtrado.groupby("Fecha")[["USD_pct", "ML_pct"]].mean().reset_index()
        prop = prop.melt(id_vars="Fecha", value_vars=["USD_pct", "ML_pct"], var_name="Moneda", value_name="Proporción")
        fig = px.area(prop, x="Fecha", y="Proporción", color="Moneda", title="Proporción Consolidada USD vs ML")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona 'Consolidado' para ver proporción global USD vs ML.")

    # --- Patrones de entradas/salidas ---
    st.subheader("🔄 Variaciones diarias")
    if modo == "Consolidado":
        serie["Var_Diaria"] = serie["Total_USD_Eq"].diff()
        fig = px.bar(serie, x="Fecha", y="Var_Diaria", title="Variaciones Diarias de Caja (USD Eq.)")
        st.plotly_chart(fig, use_container_width=True)

        # Detectar outliers
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

    # --- Proyección de cajas ---
    st.subheader("📊 Proyección de Caja (Regresión Lineal)")
    horizonte = st.slider("Selecciona días a proyectar:", 5, 60, 15)

    if modo == "Consolidado":
        serie = serie.dropna()
        X = np.arange(len(serie)).reshape(-1, 1)
        y = serie["Total_USD_Eq"].values
        model = LinearRegression().fit(X, y)

        future_X = np.arange(len(serie), len(serie)+horizonte).reshape(-1, 1)
        future_pred = model.predict(future_X)

        future_dates = pd.date_range(start=serie["Fecha"].iloc[-1] + pd.Timedelta(days=1), periods=horizonte)
        df_pred = pd.DataFrame({"Fecha": future_dates, "Proyección": future_pred})

        fig = px.line(title="Proyección de Caja (Regresión Lineal)")
        fig.add_scatter(x=serie["Fecha"], y=serie["Total_USD_Eq"], mode="lines", name="Histórico")
        fig.add_scatter(x=df_pred["Fecha"], y=df_pred["Proyección"], mode="lines", name="Proyección")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona 'Consolidado' para ver proyección global de caja.")

else:
    st.warning("⚠️ Sube el archivo BaseCaja.xlsx para continuar.")
