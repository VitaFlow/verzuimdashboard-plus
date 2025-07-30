
import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_model, preprocess, predict
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Verzuimdashboard Plus", layout="wide")
st.title("📊 Verzuimdashboard Plus")

uploaded_file = st.file_uploader("📤 Upload een Excelbestand", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Ingevoerde Data")
        st.dataframe(df)

        model = load_model("model.pkl")
        df_clean = preprocess(df)
        df_pred = predict(model, df_clean)

        st.subheader("📊 Samenvatting")
        st.write(df_pred.describe())

        # Filter op afdeling
        afdelingen = df_pred["Afdeling"].unique().tolist()
        geselecteerde_afdelingen = st.multiselect("Filter op Afdeling", options=afdelingen, default=afdelingen)
        df_filtered = df_pred[df_pred["Afdeling"].isin(geselecteerde_afdelingen)]

        # Simuleer maandoverdracht (voor trendanalyse)
        if "Maand" not in df_filtered.columns:
            df_filtered["Maand"] = pd.to_datetime("2023-01-01") + pd.to_timedelta((df_filtered.index % 12) * 30, unit="D")
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

        # Download CSV
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Resultaten (CSV)", data=csv, file_name="verzuimresultaten.csv", mime="text/csv")

        # Rapportage PDF
        def genereer_rapport(data):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Verzuimrapportage", ln=True, align="C")
            pdf.ln(10)
            afdelingen = data["Afdeling"].unique()
            for afd in afdelingen:
                subset = data[data["Afdeling"] == afd]
                pdf.set_font("Arial", "B", size=12)
                pdf.cell(200, 10, txt=f"Afdeling: {afd}", ln=True)
                pdf.set_font("Arial", size=11)
                for col in ["Risicoscore", "ZiekteverzuimScore", "MentaleBelastingScore", "FysiekeBelastingScore"]:
                    if col in subset.columns:
                        waarde = subset[col].mean()
                        pdf.cell(200, 10, txt=f"Gemiddelde {col}: {waarde:.2f}", ln=True)
                pdf.ln(5)
            buffer = BytesIO()
            pdf.output(buffer)
            return buffer.getvalue()

        if st.button("📄 Genereer PDF-Rapport"):
            try:
                pdf_bytes = genereer_rapport(df_filtered)
                st.download_button("📥 Download Rapport (PDF)", data=pdf_bytes, file_name="verzuimrapport.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Fout bij genereren rapport: {e}")

    except Exception as e:
        st.error(f"Fout bij verwerken van bestand: {e}")
else:
    st.info("📎 Upload een Excelbestand om te beginnen.")
