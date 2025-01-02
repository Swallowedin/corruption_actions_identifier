import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import io

# Configuration de la page
st.set_page_config(page_title="Générateur de Mesures de Remédiation", layout="wide")

# Configuration OpenAI - À mettre dans les secrets Streamlit
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Données des processus intégrées directement
PROCESSES = {
    "PILOTAGE": {
        "DIRECTION": "FP-PIL-DIR",
        "INTERNATIONAL": "FP-PIL-INT",
        "PERFORMANCE": "FP-PIL-PERF",
        "DEVELOPPEMENT NATIONAL": "FP-PIL-DEVNAT",
        "DEVELOPPEMENT INTERNATIONAL": "FP-PIL-DEVINT",
        "RSE": "FP-PIL-RSE",
        "GESTION DES RISQUES": "FP-PIL-RISK",
        "FUSIONS ET ACQUISITIONS": "FP-PIL-FUSAC",
        "INNOVATION ET TRANSFO": "FP-PIL-TRA"
    },
    "OPERATIONNELS": {
        "VENTE MAGASIN": "FP-OPE-VMAG",
        "LOGISTIQUE": "FP-OPE-LOG",
        "APPROVISONNEMENT": "FP-OPE-APPRO",
        "ACHATS": "FP-OPE-ACH",
        "SAV": "FP-OPE-SAV",
        "IMPORT": "FP-OPE-IMP",
        "FINANCEMENT": "FP-OPE-FIN",
        "AUTRES MODES DE VENTE": "FP-OPE-AUTVEN",
        "VALORISATION DES DECHETS": "FP-OPE-VALO",
        "QUALITE": "FP-OP-QUALPRO",
        "VENTE WEB": "FP-OPE-VWEB",
        "FRANCHISE": "FP-OPE-FRA"
    },
    "SUPPORT": {
        "COMPTABILITE": "FP-SUP-COMPTA",
        "DSI": "FP-SUP-DSI",
        "GESTION RH": "FP-SUP-DRH",
        "MARKETING ET COM": "FP-SUP-COM",
        "ORGANISATION": "FP-SUP-ORG",
        "TECHNIQUE": "FP-SUP-TEC",
        "JURIDIQUE": "FP-SUP-JUR",
        "SECURITE": "FP-SUP-SEC"
    }
}

# Liste des risques intégrée directement
RISKS = [
    "A.1 - CAD - Cadeaux et invitations dérogeant à la politique",
    "A.2 - CAD - Cadeaux manifestement corruptifs",
    "A.3 - CAD - Prise en charge de prestations anormales par le partenaire",
    "A.4 - CAD - Salons professionnels",
    "A.5 - CAD - Démarque pour ne pas affecter le compte d'exploitation",
    "B.1 - CON - Conflit d'intérêts entre un collaborateur et une société",
    "B.2 - CON - Collaborateur favorisant sa propre entreprise / son patrimoine",
    "B.3 - CON - Boulanger / Electro Dépôt",
    "B.4 - CON - United.B & AFM / Electro Dépôt",
    "B.5 - CON - Prestataires également partenaires",
    "B.6 - CON - Gestion du personnel"
    # ... ajoutez tous les autres risques ici
]

def generate_measures(risk, process):
    """Génère des mesures via GPT pour un risque donné"""
    prompt = f"""Pour le processus {process} et le risque {risk}, proposer UNIQUEMENT des mesures concrètes et spécifiques de remédiation selon les catégories suivantes. 
    Pour chaque catégorie, donner EXACTEMENT UNE mesure concrète et applicable :

    D = Mesure de détection du risque
    R = Mesure de réduction du risque
    A = Mesure d'acceptation du risque
    F = Mesure de refus / fin de non-recevoir
    T = Mesure de transfert du risque
    
    Format de réponse attendu (EXACTEMENT ce format) :
    D: [votre mesure de détection]
    R: [votre mesure de réduction]
    A: [votre mesure d'acceptation]
    F: [votre mesure de refus]
    T: [votre mesure de transfert]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de génération: {str(e)}"

def main():
    st.title("🛡️ Générateur de Mesures de Remédiation des Risques")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Sélection de la famille de processus
        famille = st.selectbox(
            "Famille de processus",
            options=list(PROCESSES.keys())
        )
        
        # Sélection du processus
        if famille:
            processus = st.selectbox(
                "Processus",
                options=list(PROCESSES[famille].keys()),
                format_func=lambda x: f"{x} ({PROCESSES[famille][x]})"
            )

        # Sélection multiple des risques
        selected_risks = st.multiselect(
            "Risques à analyser",
            options=RISKS
        )

    with col2:
        if selected_risks and st.button("Générer les mesures de remédiation"):
            results = []
            
            for risk in selected_risks:
                st.subheader(f"📊 {risk}")
                
                with st.spinner("Génération des mesures..."):
                    measures = generate_measures(risk, processus)
                    st.text_area("Mesures proposées", measures, height=200)
                    
                    # Stockage pour export
                    results.append({
                        "Processus": processus,
                        "Référence": PROCESSES[famille][processus],
                        "Risque": risk,
                        "Mesures": measures
                    })
            
            # Export Excel
            if results:
                df = pd.DataFrame(results)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📥 Télécharger le rapport Excel",
                    data=buffer.getvalue(),
                    file_name="mesures_remediation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
