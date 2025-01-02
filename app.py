import streamlit as st
import pandas as pd
from openai import OpenAI
import io
from collections import defaultdict

# Configuration de la page
st.set_page_config(page_title="Générateur de Mesures de Remédiation", layout="wide")

# Configuration OpenAI
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
    prompt = f"""Pour le processus {process} et le risque {risk}, proposer des mesures concrètes et spécifiques de remédiation selon les catégories suivantes.
    Donner AU MOINS 3 mesures par catégorie. Les mesures doivent être précises, actionnables et adaptées au contexte.

    D = Mesures de détection du risque (comment identifier/détecter)
    R = Mesures de réduction du risque (comment réduire la probabilité ou l'impact)
    A = Mesures d'acceptation du risque (comment gérer si on accepte le risque)
    F = Mesures de refus / fin de non-recevoir (quelles limites fixer)
    T = Mesures de transfert du risque (comment transférer à des tiers)

    Format de réponse attendu (EXACTEMENT ce format) :
    D1: [première mesure de détection]
    D2: [deuxième mesure de détection]
    D3: [troisième mesure de détection]
    R1: [première mesure de réduction]
    R2: [deuxième mesure de réduction]
    R3: [troisième mesure de réduction]
    A1: [première mesure d'acceptation]
    A2: [deuxième mesure d'acceptation]
    A3: [troisième mesure d'acceptation]
    F1: [première mesure de refus]
    F2: [deuxième mesure de refus]
    F3: [troisième mesure de refus]
    T1: [première mesure de transfert]
    T2: [deuxième mesure de transfert]
    T3: [troisième mesure de transfert]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de génération: {str(e)}"

def parse_measures(measures_text):
    """Parse les mesures générées en dictionnaire"""
    measures_dict = defaultdict(list)
    for line in measures_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            category = key[0]  # D, R, A, F, ou T
            measures_dict[category].append(value.strip())
    return measures_dict

def find_common_measures(all_measures):
    """Identifie les mesures communes entre les risques"""
    common_measures = defaultdict(lambda: defaultdict(int))
    
    for risk, measures in all_measures.items():
        for category, measure_list in measures.items():
            for measure in measure_list:
                common_measures[category][measure] += 1
                
    return common_measures

def main():
    st.title("🛡️ Générateur de Mesures de Remédiation des Risques")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        famille = st.selectbox(
            "Famille de processus",
            options=list(PROCESSES.keys())
        )
        
        if famille:
            processus = st.selectbox(
                "Processus",
                options=list(PROCESSES[famille].keys()),
                format_func=lambda x: f"{x} ({PROCESSES[famille][x]})"
            )

        selected_risks = st.multiselect(
            "Risques à analyser",
            options=RISKS
        )

    with col2:
        if selected_risks and st.button("Générer les mesures de remédiation"):
            all_measures = {}
            results = []
            
            for risk in selected_risks:
                st.subheader(f"📊 {risk}")
                
                with st.spinner("Génération des mesures..."):
                    measures_text = generate_measures(risk, processus)
                    measures_dict = parse_measures(measures_text)
                    all_measures[risk] = measures_dict
                    
                    # Affichage des mesures par catégorie
                    col_measures1, col_measures2 = st.columns(2)
                    with col_measures1:
                        st.write("**Mesures de Détection (D)**")
                        for m in measures_dict['D']:
                            st.write(f"• {m}")
                        st.write("**Mesures de Réduction (R)**")
                        for m in measures_dict['R']:
                            st.write(f"• {m}")
                        st.write("**Mesures d'Acceptation (A)**")
                        for m in measures_dict['A']:
                            st.write(f"• {m}")
                            
                    with col_measures2:
                        st.write("**Mesures de Refus (F)**")
                        for m in measures_dict['F']:
                            st.write(f"• {m}")
                        st.write("**Mesures de Transfert (T)**")
                        for m in measures_dict['T']:
                            st.write(f"• {m}")
                    
                    # Stockage pour export
                    results.append({
                        "Processus": processus,
                        "Référence": PROCESSES[famille][processus],
                        "Risque": risk,
                        "Mesures": measures_text
                    })
                    
                st.divider()
            
            # Analyse des mesures communes
            if len(selected_risks) > 1:
                st.subheader("🔄 Mesures communes identifiées")
                common_measures = find_common_measures(all_measures)
                
                for category, measures in common_measures.items():
                    category_names = {
                        'D': 'Détection',
                        'R': 'Réduction',
                        'A': 'Acceptation',
                        'F': 'Refus',
                        'T': 'Transfert'
                    }
                    
                    common = {m: count for m, count in measures.items() if count > 1}
                    if common:
                        st.write(f"**Mesures de {category_names[category]} communes:**")
                        for measure, count in common.items():
                            st.write(f"• {measure} _(présente dans {count} risques)_")
                        st.write("")
            
            # Export Excel
            if results:
                df = pd.DataFrame(results)
                df_common = pd.DataFrame([
                    {
                        "Catégorie": cat,
                        "Mesure": measure,
                        "Nombre de risques": count
                    }
                    for cat, measures in common_measures.items()
                    for measure, count in measures.items()
                    if count > 1
                ])
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Mesures détaillées', index=False)
                    if not df_common.empty:
                        df_common.to_excel(writer, sheet_name='Mesures communes', index=False)
                
                st.download_button(
                    label="📥 Télécharger le rapport Excel",
                    data=buffer.getvalue(),
                    file_name="mesures_remediation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
