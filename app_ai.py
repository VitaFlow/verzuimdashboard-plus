import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_model, predict

st.set_page_config(layout='wide')
st.title("HR Verzuimvoorspeller Dashboard + AI Advies")

model = load_model("model.pkl")

uploaded_file = st.file_uploader("Upload Excelbestand met medewerkergegevens", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df_pred = predict(model, df)

        st.subheader("📊 Samenvatting in HR-inzichten")
        col1, col2, col3 = st.columns(3)

        with col1:
            gem_risico = df_pred["Risicoscore"].mean()
            st.metric("Gemiddelde Risicoscore", f"{gem_risico:.2f}")

        with col2:
            hoog_risico_pct = (df_pred["Risicoscore"] > 0.5).mean() * 100
            st.metric("Hoog risico medewerkers (%)", f"{hoog_risico_pct:.1f}%")

        with col3:
            hoogste_afdeling = df_pred.groupby("Afdeling")["Risicoscore"].mean().sort_values(ascending=False).idxmax()
            st.metric("Afdeling met hoogste risico", hoogste_afdeling)

        afdelingen = st.multiselect("Filter op Afdeling", options=df_pred['Afdeling'].unique(), default=df_pred['Afdeling'].unique())
        risico_filter = st.multiselect("Filter op Risicoklasse", options=['Laag', 'Midden', 'Hoog'], default=['Laag', 'Midden', 'Hoog'])

        df_filtered = df_pred[df_pred['Afdeling'].isin(afdelingen) & df_pred['Risicoklasse'].isin(risico_filter)]

        st.subheader("🧠 AI Adviesmodule")
        col1, col2 = st.columns(2)

        with col1:
            top_afdelingen = df_filtered.groupby("Afdeling")["Risicoscore"].mean().sort_values(ascending=False).head(3)
            st.markdown("**Top 3 afdelingen met hoogste gemiddelde risico:**")
            for afd, score in top_afdelingen.items():
                st.write(f"🔺 {afd}: {score:.2f}")

        with col2:
            risicotelling = df_filtered['Risicoklasse'].value_counts().reindex(['Laag', 'Midden', 'Hoog']).fillna(0).astype(int)
            st.markdown("**Aantal medewerkers per risicoklasse:**")
            for klasse, aantal in risicotelling.items():
                st.write(f"👤 {klasse}: {aantal} medewerkers")

        st.subheader("📈 Verzuim Risicoverdeling")
        fig = px.histogram(df_filtered, x="Risicoscore", nbins=20, title="Histogram Risicoscore")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏢 Gemiddeld risico per Afdeling")
        avg_risico = df_filtered.groupby("Afdeling")["Risicoscore"].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(avg_risico, x="Afdeling", y="Risicoscore", title="Gemiddeld Risico per Afdeling")
        st.plotly_chart(fig2, use_container_width=True)


        st.subheader("📉 Trendlijn: Gemiddelde Risicoscore per Maand")
        if 'Maand' in df_filtered.columns:
            afdelingen_trend = df_filtered['Afdeling'].unique().tolist()
            selected_afdelingen = st.multiselect("Selecteer afdelingen voor trendlijn", options=afdelingen_trend, default=afdelingen_trend)
            trend_df = df_filtered[df_filtered["Afdeling"].isin(selected_afdelingen)]
            maand_risico = trend_df.groupby(["Maand", "Afdeling"])["Risicoscore"].mean().reset_index()
            fig3 = px.line(maand_risico.sort_values("Maand"), x="Maand", y="Risicoscore",
                           color="Afdeling", markers=True,
                           title="Gemiddelde Risicoscore per Maand per Afdeling")
            fig3.update_layout(xaxis_title="Maand", yaxis_title="Gem. Risicoscore", yaxis_range=[0, 1])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📅 Geen 'Maand'-kolom gevonden in dataset. Upload een bestand met een 'Maand'-kolom (bijv. 2024-05).")

        st.subheader("📥 Download Resultaten")
        csv = df_pred.to_csv(index=False).encode("utf-8")



        # PDF Rapportage met afdelingsfilter + gekleurde grafieken
        from fpdf import FPDF
        import tempfile
        import datetime
        import plotly.io as pio

        st.subheader("📄 HR Rapportage Export")
        afdelingen_pdf = df_filtered['Afdeling'].unique().tolist()
        selected_afdelingen_pdf = st.multiselect("Selecteer afdelingen voor rapportage", options=afdelingen_pdf, default=afdelingen_pdf)

        if st.button("Genereer HR Rapportage (PDF)"):
            try:
                report_df = df_filtered[df_filtered["Afdeling"].isin(selected_afdelingen_pdf)]
                avg_score = report_df["Risicoscore"].mean()
                high_risk_pct = (report_df["Risicoscore"] > 0.5).mean() * 100
                top_afdeling = report_df.groupby("Afdeling")["Risicoscore"].mean().idxmax()

                # Maak grafieken
                hist_fig = px.histogram(report_df, x="Risicoscore", nbins=20, title="Histogram Risicoscore")

                trend_fig = None
                if 'Maand' in report_df.columns:
                    maand_risico = report_df.groupby(["Maand", "Afdeling"])["Risicoscore"].mean().reset_index()
                    trend_fig = px.line(
                        maand_risico.sort_values("Maand"),
                        x="Maand", y="Risicoscore",
                        color="Afdeling", markers=True,
                        title="Gemiddelde Risicoscore per Maand per Afdeling",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )

                with tempfile.TemporaryDirectory() as tmpdir:
                    hist_path = f"{tmpdir}/histogram.png"
                    hist_fig.write_image(hist_path, format="png")

                    trend_path = None
                    if trend_fig:
                        trend_path = f"{tmpdir}/trendlijn.png"
                        trend_fig.write_image(trend_path, format="png")

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)

                    pdf.set_title("HR Verzuimanalyse Rapport")
                    pdf.cell(200, 10, txt="HR Verzuimanalyse Rapport", ln=True)
                    pdf.cell(200, 10, txt=f"Datum: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)

                    pdf.ln(10)
                    pdf.cell(200, 10, txt=f"Gemiddelde Risicoscore: {avg_score:.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"% medewerkers met hoog risico: {high_risk_pct:.1f}%", ln=True)
                    pdf.cell(200, 10, txt=f"Afdeling met hoogste risico: {top_afdeling}", ln=True)

                    # Voeg grafieken toe
                    pdf.ln(10)
                    pdf.cell(200, 10, txt="Histogram Risicoscore", ln=True)
                    pdf.image(hist_path, x=10, w=180)

                    if trend_path:
                        pdf.add_page()
                        pdf.cell(200, 10, txt="Trendlijn Risicoscore per Maand per Afdeling", ln=True)
                        pdf.image(trend_path, x=10, w=180)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                        pdf.output(tmpfile.name)
                        with open(tmpfile.name, "rb") as f:
                            st.download_button("Download HR Rapportage", f.read(), file_name="hr_rapportage.pdf")

            except Exception as e:
                st.error(f"Fout bij genereren rapport: {e}")

                st.error(f"Fout bij genereren rapport: {e}")

                st.error(f"Fout bij genereren rapport: {e}")

        st.download_button("Download CSV met Risicoscores", csv, "verzuimresultaten.csv", "text/csv")
    except Exception as e:
        st.error(f"Fout bij verwerken bestand: {e}")
else:
    st.info("Upload een bestand om te beginnen.")



# === AI Aanbevelingen ===
if 'df_pred' in locals():
    st.subheader("🧠 AI Aanbevelingen per Werknemer")

    def genereer_aanbeveling(rij):
        aanbeveling = []
        if rij['Risicoscore'] > 0.6:
            aanbeveling.append("Zeer hoog verzuimrisico – plan een preventief gesprek.")
        elif rij['Risicoscore'] > 0.4:
            aanbeveling.append("Verhoogd risico – houd actief vinger aan de pols.")
        if rij['MentaleBelastingScore'] > 0.7:
            aanbeveling.append("Let op mentale belasting – overweeg coachingsgesprek.")
        if rij['FysiekeBelastingScore'] > 0.7:
            aanbeveling.append("Fysieke belasting is hoog – check werkplek en ergonomie.")
        if rij['WerktevredenheidScore'] < 0.4:
            aanbeveling.append("Lage tevredenheid – plan HR check-in.")
        if rij['ZiekteverzuimScore'] > 0.6:
            aanbeveling.append("Aanhoudend ziekteverzuim – monitor herstelbegeleiding.")
        return " ".join(aanbeveling) if aanbeveling else "Geen directe actie nodig."

    df_pred['AI_Aanbeveling'] = df_pred.apply(genereer_aanbeveling, axis=1)

    st.dataframe(df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']])

    csv_ai = df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download AI Aanbevelingen (CSV)", data=csv_ai, file_name="ai_aanbevelingen.csv", mime="text/csv")


st.subheader("🧠 AI Aanbevelingen per Werknemer")

def genereer_aanbeveling(rij):
    aanbeveling = []
    if rij['Risicoscore'] > 0.6:
        aanbeveling.append("Zeer hoog verzuimrisico – plan een preventief gesprek.")
    elif rij['Risicoscore'] > 0.4:
        aanbeveling.append("Verhoogd risico – houd actief vinger aan de pols.")
    if rij['MentaleBelastingScore'] > 0.7:
        aanbeveling.append("Let op mentale belasting – overweeg coachingsgesprek.")
    if rij['FysiekeBelastingScore'] > 0.7:
        aanbeveling.append("Fysieke belasting is hoog – check werkplek en ergonomie.")
    if rij['WerktevredenheidScore'] < 0.4:
        aanbeveling.append("Lage tevredenheid – plan HR check-in.")
    if rij['ZiekteverzuimScore'] > 0.6:
        aanbeveling.append("Aanhoudend ziekteverzuim – monitor herstelbegeleiding.")
    return " ".join(aanbeveling) if aanbeveling else "Geen directe actie nodig."

df_pred['AI_Aanbeveling'] = df_pred.apply(genereer_aanbeveling, axis=1)

st.dataframe(df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']])

csv_ai = df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']].to_csv(index=False).encode('utf-8')
st.download_button("📥 Download AI Aanbevelingen (CSV)", data=csv_ai, file_name="ai_aanbevelingen.csv", mime="text/csv")
