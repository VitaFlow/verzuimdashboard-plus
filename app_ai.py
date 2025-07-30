import streamlit as st
import pandas as pd
import plotly.express as px
from model import load_model, preprocess, predict

st.set_page_config(page_title="Verzuimdashboard AI", layout="wide")
st.title("📊 Verzuimdashboard Plus – met AI-aanbevelingen")

uploaded_file = st.file_uploader("📤 Upload een Excelbestand", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Ingevoerde Data")
        st.dataframe(df)

        model = load_model("model.pkl")
        df_clean = preprocess(df)
        df_pred = predict(model, df_clean)

        # === AI Aanbevelingen per medewerker (rule-based) ===
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

        st.subheader("🧠 AI Aanbevelingen per Werknemer")
        st.dataframe(df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']])

        csv_ai = df_pred[['Werknemer_ID', 'Afdeling', 'Risicoscore', 'AI_Aanbeveling']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download AI Aanbevelingen (CSV)", data=csv_ai, file_name="ai_aanbevelingen.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Fout bij verwerken van bestand: {e}")
else:
    st.info("📎 Upload een Excelbestand om te beginnen.")
