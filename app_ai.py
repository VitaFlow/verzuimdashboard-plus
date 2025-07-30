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

        st.subheader("📊 Samenvatting Data")
        st.write(df_pred.describe())

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
            maand_risico = df_filtered.groupby("Maand")["Risicoscore"].mean().reset_index()
            fig3 = px.line(maand_risico.sort_values("Maand"), x="Maand", y="Risicoscore",
                           title="Gemiddelde Risicoscore per Maand", markers=True)
            fig3.update_layout(xaxis_title="Maand", yaxis_title="Gem. Risicoscore", yaxis_range=[0, 1])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📅 Geen 'Maand'-kolom gevonden in dataset. Upload een bestand met een 'Maand'-kolom (bijv. 2024-05).")

        st.subheader("📥 Download Resultaten")
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV met Risicoscores", csv, "verzuimresultaten.csv", "text/csv")
    except Exception as e:
        st.error(f"Fout bij verwerken bestand: {e}")
else:
    st.info("Upload een bestand om te beginnen.")
