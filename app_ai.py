import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_model, preprocess, predict
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Verzuimdashboard Plus", layout="wide")
st.title("📊 Verzuimdashboard Plus")

uploaded_file = st.file_uploader("📤 Upload een Excelbestand", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        model = load_model("model.pkl")
        df_clean = preprocess(df)
        risicoscore = predict(model, df_clean)
        df_pred = pd.concat([df.reset_index(drop=True), risicoscore.reset_index(drop=True)], axis=1)
        df_pred['Risicoklasse'] = pd.cut(df_pred['Risicoscore'], bins=[0, 0.33, 0.66, 1], labels=['Laag', 'Midden', 'Hoog'])

        # 🔍 HR-inzichten
        st.subheader("📌 HR-inzichten")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Aantal medewerkers", len(df_pred))
        col2.metric("Gemiddelde Risicoscore", f"{df_pred['Risicoscore'].mean():.2f}")
        col3.metric("% Hoog risico", f"{(df_pred['Risicoklasse'] == 'Hoog').mean() * 100:.1f}%")
        hoogste_afdeling = df_pred.groupby("Afdeling")["Risicoscore"].mean().idxmax()
        col4.metric("Hoogste risico afdeling", hoogste_afdeling)

        # 🔢 Filters
        st.subheader("🎯 Filter op Afdeling")
        afdelingen = df_pred["Afdeling"].unique().tolist()
        geselecteerde_afdelingen = st.multiselect("Afdelingen", options=afdelingen, default=afdelingen)
        df_filtered = df_pred[df_pred["Afdeling"].isin(geselecteerde_afdelingen)].copy()

        if "Maand" not in df_filtered.columns:
            df_filtered["Maand"] = pd.date_range(start="2023-01-01", periods=len(df_filtered), freq="M")
            df_filtered["Maand"] = df_filtered["Maand"].dt.to_period("M").astype(str)

        st.subheader("📈 Gemiddelde Risicoscore per Maand")
        trenddata = df_filtered.groupby(["Maand", "Afdeling"])["Risicoscore"].mean().reset_index()
        fig_trend = px.line(trenddata, x="Maand", y="Risicoscore", color="Afdeling", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("📊 Verdeling Risicoscores")
        fig_hist = px.histogram(df_filtered, x="Risicoscore", nbins=10)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("🏢 Gemiddeld Risico per Afdeling")
        afdeling_risico = df_filtered.groupby("Afdeling")["Risicoscore"].mean().reset_index()
        fig_bar = px.bar(afdeling_risico, x="Afdeling", y="Risicoscore", color="Afdeling")
        st.plotly_chart(fig_bar, use_container_width=True)

        # 💡 AI-Adviesmodule
        st.subheader("🤖 AI Advies per Medewerker")

        def genereer_advies(score):
            if score < 0.33:
                return "Geen direct risico"
            elif score < 0.66:
                return "Let op werkdruk & balans"
            else:
                return "Plan coachinggesprek / mentale check-in"

        df_filtered["Advies"] = df_filtered["Risicoscore"].apply(genereer_advies)
        st.dataframe(df_filtered[["Werknemer_ID", "Afdeling", "Risicoscore", "Risicoklasse", "Advies"]])

        # 📥 Downloadbare adviezen
        advies_csv = df_filtered[["Werknemer_ID", "Afdeling", "Risicoscore", "Risicoklasse", "Advies"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Adviezen (CSV)", data=advies_csv, file_name="ai_adviezen.csv", mime="text/csv")

        # 📤 Download alle resultaten (optioneel)
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Alle Resultaten (CSV)", data=csv, file_name="verzuimresultaten.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Fout bij verwerken van bestand: {e}")
else:
    st.info("📎 Upload een Excel- of CSV-bestand om te beginnen.")
