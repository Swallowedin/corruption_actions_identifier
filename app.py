import streamlit as st
import pandas as pd
import openai
from pathlib import Path

# Configuration de la page
st.set_page_config(page_title="Générateur de Mesures de Remédiation", layout="wide")

# Configuration OpenAI - À mettre dans les secrets Streamlit
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Lecture du fichier des processus (format CSV attendu)
def load_processes():
    df = pd.read_csv("processus.csv", delimiter='\t')
    processes = {}
    for _, row in df.iterrows():
        famille = row['FAMILLE DE PROCESSUS']
        processus = row['PROCESSUS']
        reference = row['REFERENCE']
        if famille not in processes:
            processes[famille] = {}
        processes[famille][processus] = reference
    return processes

# Génération des mesures via GPT
def generate_measures(risk, process):
    prompt = f"""
    Pour le processus {process} et le risque {risk}, proposer des mesures concrètes de remédiation selon les catégories suivantes :
    
    D = Détection du risque
    R = Réduction du risque
    A = Acceptation du risque
    F = Refus / fin de non-recevoir
    T = Transfert du risque
    
    Répondre uniquement avec les mesures, une par ligne, préfixées par la lettre correspondante.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de génération: {str(e)}"

# Interface principale
def main():
    st.title("🛡️ Générateur de Mesures de Remédiation des Risques")
    
    # Chargement des processus
    processes = load_processes()
    
    # Sélection de la famille de processus
    famille = st.selectbox(
        "Sélectionnez la famille de processus",
        options=list(processes.keys())
    )
    
    # Sélection du processus
    if famille:
        processus = st.selectbox(
            "Sélectionnez le processus",
            options=list(processes[famille].keys()),
            format_func=lambda x: f"{x} ({processes[famille][x]})"
        )
        
        # Liste des risques (à remplacer par votre liste complète)
        risks = [
            "A.1 - CAD - Cadeaux et invitations dérogeant à la politique",
            "A.2 - CAD - Cadeaux manifestement corruptifs",
            "B.1 - CON - Conflit d'intérêts entre un collaborateur et une société",
            # Ajoutez tous vos risques ici
        ]
        
        # Sélection multiple des risques
        selected_risks = st.multiselect(
            "Sélectionnez les risques à analyser",
            options=risks
        )
        
        if selected_risks:
            if st.button("Générer les mesures de remédiation"):
                results = []
                
                for risk in selected_risks:
                    st.subheader(f"📊 {risk}")
                    
                    with st.spinner("Génération des mesures..."):
                        measures = generate_measures(risk, processus)
                        st.text_area("Mesures proposées", measures, height=200)
                        
                        # Stockage pour export
                        results.append({
                            "Processus": processus,
                            "Référence": processes[famille][processus],
                            "Risque": risk,
                            "Mesures": measures
                        })
                
                # Export Excel
                if results:
                    df = pd.DataFrame(results)
                    excel_buffer = df.to_excel(index=False)
                    
                    st.download_button(
                        label="📥 Télécharger le rapport Excel",
                        data=excel_buffer,
                        file_name="mesures_remediation.xlsx",
                        mime="application/vnd.ms-excel"
                    )

if __name__ == "__main__":
    main()
