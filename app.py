import streamlit as st
import pandas as pd
from openai import OpenAI
import io
from collections import defaultdict

# Configuration de la page
st.set_page_config(page_title="Générateur de Mesures de Remédiation", layout="wide")

# Configuration OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Données des processus
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

# Liste des risques
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
    "B.6 - CON - Gestion du personnel",
    "C.1 - COR - Corruption active d'agent public ou d'élu",
    "C.2 - COR - Corruption passive d'agent public ou d'élu",
    "C.3 - COR - Corruption active à l'international",
    "C.4 - COR - Corruption passive à l'international",
    "C.5 - COR - Trafic d'influence",
    "C.6 - COR - Sélection d'un prestataire présentant un risque de corruption important ou avéré",
    "C.7 - COR - Maintien d'une relation d'affaire avec un prestataire en situation de non conformité règlementaire",
    "C.8 - COR - TPE / PME à risque",
    "C.9 - COR - Rétrocommission",
    "C.10 - COR - Chantage interne",
    "C.11 - COR - Coutume locale contraire au droit",
    "C.12 - COR - Non-respect des procédures et des normes de qualité et de conformité",
    "C.13 - COR - Complicité de corruption",
    "C.14 - COR - Prestations fictives",
    "C.15 - COR - Sous-traitance",
    "D.1 - FAV - Défaut de mise en concurrence",
    "D.2 - FAV - Détournement de la procédure de mise en concurrence",
    "D.3 - FAV - Rupture d'égalité entre les candidats",
    "D.4 - FAV - Eviction injustifiée d'une candidature",
    "D.5 - FAV - Sélection sur pression",
    "D.6 - FAV - Sélection d'un prestataire non professionnel",
    "D.7 - FAV - Négociation insuffisante",
    "E.1 - FRAUD - Falsification de factures",
    "E.2 - FRAUD - fausse écriture comptable / Omission d'écritures comptables",
    "E.3 - FRAUD - Manipulation de caisse",
    "E.4 - FRAUD - Abus de biens sociaux et assimilés",
    "E.5 - FRAUD - Fraudes fiscales et douanières",
    "E.6 - FRAUD - Diffusion non-autorisée de données",
    "E.7 - FRAUD - Associations fictives",
    "F.1 - MGMT - Carences dans le contrôle",
    "F.2 - MGMT - Défaut de compétence du manager",
    "F.3 - MGMT - Conflit d'intérêts interne",
    "F.4 - MGMT - Défaut de culture éthique",
    "G.1 - LOB - Lobbying"
]

# Standards et référentiels
STANDARDS = {
    "ISO 37001": {
        "4.4": "Évaluation des risques de corruption",
        "4.5": "Mise en œuvre des contrôles",
        "5.2": "Politique anti-corruption",
        "7.3": "Sensibilisation et formation",
        "8.2": "Due diligence",
        "8.3": "Contrôles financiers",
        "8.4": "Contrôles non-financiers",
        "8.5": "Contrôles anti-corruption",
        "8.7": "Cadeaux, invitations, dons",
        "9.2": "Audit interne"
    },
    "ISO 37301": {
        "4.6": "Identification des obligations de conformité",
        "5.1": "Leadership et engagement",
        "6.1": "Actions face aux risques et opportunités",
        "7.2": "Compétence",
        "7.3": "Sensibilisation",
        "8.1": "Planification et contrôle opérationnels",
        "9.1": "Surveillance et mesure",
        "9.2": "Audit de conformité",
        "10.1": "Amélioration continue"
    },
    "COSO ERM": {
        "Gouvernance": ["Culture", "Supervision des risques"],
        "Stratégie": ["Contexte", "Appétence au risque"],
        "Performance": ["Identification", "Évaluation", "Priorisation", "Réponses"],
        "Revue": ["Révision", "Amélioration"],
        "Information": ["Communication", "Reporting"]
    },
    "COSO CI": {
        "Environnement de contrôle": ["Intégrité", "Structure", "Autorité"],
        "Évaluation des risques": ["Objectifs", "Identification", "Analyse"],
        "Activités de contrôle": ["Politiques", "Procédures"],
        "Information et communication": ["Qualité", "Communication interne/externe"],
        "Pilotage": ["Évaluations", "Suivi"]
    }
}

def generate_measures(risk, process):
    """Génère des mesures via GPT avec références aux standards"""
    prompt = f"""Pour le processus {process} et le risque {risk}, proposer des mesures concrètes et spécifiques en faisant référence aux standards ISO 37001, ISO 37301, COSO ERM et COSO CI.
    Pour chaque catégorie, donner AU MOINS 3 mesures concrètes, chacune avec une référence au standard le plus pertinent.

    Catégories de mesures :
    D = Mesures de détection du risque (comment identifier/détecter)
    R = Mesures de réduction du risque (comment réduire la probabilité ou l'impact)
    A = Mesures d'acceptation du risque (comment gérer si on accepte le risque)
    F = Mesures de refus / fin de non-recevoir (quelles limites fixer)
    T = Mesures de transfert du risque (comment transférer à des tiers)

    Format de réponse attendu :
    [Catégorie][Numéro]: [Description détaillée de la mesure] (Référence standard)

    Exemple de format :
    D1: Mise en place d'audits mensuels des transactions suspectes (ISO 37001 9.2)
    D2: Programme de contrôle continu des déclarations d'intérêts (COSO CI - Activités de contrôle)
    D3: Système d'alerte automatisé sur les dépassements de seuils (ISO 37301 9.1)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de génération: {str(e)}"

def parse_measures_with_refs(measures_text):
    """Parse les mesures générées avec leurs références"""
    measures_dict = defaultdict(list)
    refs_dict = defaultdict(list)
    
    for line in measures_text.split('\n'):
        if ':' in line and line[0] in ['D', 'R', 'A', 'F', 'T']:
            key, content = line.split(':', 1)
            category = key[0]
            
            # Extraire la mesure et la référence
            content = content.strip()
            if '(' in content and ')' in content:
                measure = content[:content.rfind('(')].strip()
                ref = content[content.rfind('(')+1:content.rfind(')')].strip()
            else:
                measure = content
                ref = "Pas de référence"
            
            measures_dict[category].append(measure)
            refs_dict[f"{category}-{len(measures_dict[category])}"] = ref
                
    return measures_dict, refs_dict
def generate_best_practices_and_kpis(risk, process):
    """Génère des bonnes pratiques et KPIs associés"""
    prompt = f"""Pour le processus {process} et le risque {risk}, proposer :
    1. 3-5 bonnes pratiques concrètes basées sur les standards du secteur
    2. Pour chaque bonne pratique, 1-2 KPIs mesurables et pertinents

    Format de réponse attendu :
    BP1: [Description de la bonne pratique]
    - KPI1: [Description du KPI] (Fréquence: [fréquence], Cible: [cible])
    - KPI2: [Description du KPI] (Fréquence: [fréquence], Cible: [cible])

    BP2: [Description de la bonne pratique]
    - KPI1: [Description du KPI] (Fréquence: [fréquence], Cible: [cible])
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

def parse_best_practices_and_kpis(text):
    """Parse les bonnes pratiques et KPIs générés"""
    practices = {}
    current_bp = None
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('BP'):
            current_bp = line.split(':', 1)[1].strip()
            practices[current_bp] = []
        elif line.startswith('-') and current_bp:
            kpi = line[1:].strip()
            practices[current_bp].append(kpi)
            
    return practices

def main():
    st.title("🛡️ Générateur de Mesures de Remédiation des Risques")
    st.markdown("""
    Cette application génère des mesures de remédiation, bonnes pratiques et KPIs 
    en se basant sur les standards ISO 37001, ISO 37301, COSO ERM et COSO CI.
    """)
    
    # Layout en colonnes
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Sélection de la famille de processus
        famille = st.selectbox(
            "Famille de processus",
            options=list(PROCESSES.keys())
        )
        
        # Sélection du processus
        processus = None
        if famille:
            processus = st.selectbox(
                "Processus",
                options=list(PROCESSES[famille].keys()),
                format_func=lambda x: f"{x} ({PROCESSES[famille][x]})"
            )

            # Sélection des risques
            if processus:
                selected_risks = st.multiselect(
                    "Risques à analyser",
                    options=RISKS
                )

    with col2:
        if processus and selected_risks and st.button("Générer l'analyse complète", type="primary"):
            all_measures = {}
            all_refs = {}
            all_practices = {}
            results = []
            
            progress_text = "Génération en cours..."
            progress_bar = st.progress(0)
            
            for idx, risk in enumerate(selected_risks):
                st.subheader(f"📊 {risk}")
                
                with st.spinner(f"Analyse du risque {idx + 1}/{len(selected_risks)}"):
                    # Génération des mesures
                    measures_text = generate_measures(risk, processus)
                    measures_dict, refs_dict = parse_measures_with_refs(measures_text)
                    
                    # Génération des bonnes pratiques et KPIs
                    practices_text = generate_best_practices_and_kpis(risk, processus)
                    practices_dict = parse_best_practices_and_kpis(practices_text)
                    
                    all_measures[risk] = measures_dict
                    all_refs[risk] = refs_dict
                    all_practices[risk] = practices_dict
                    
                    # Création des onglets principaux
                    tab_measures, tab_practices = st.tabs(["Mesures de remédiation", "Bonnes pratiques & KPIs"])
                    
                    with tab_measures:
                        # Sous-onglets pour les différentes catégories de mesures
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("### Mesures de détection (D)")
                            for i, measure in enumerate(measures_dict.get('D', []), 1):
                                ref = refs_dict.get(f"D-{i}", "")
                                st.markdown(f"""
                                📌 **Mesure {i}:**  
                                {measure}  
                                *Ref: {ref}*
                                """)
                            
                            st.markdown("### Mesures de réduction (R)")
                            for i, measure in enumerate(measures_dict.get('R', []), 1):
                                ref = refs_dict.get(f"R-{i}", "")
                                st.markdown(f"""
                                📌 **Mesure {i}:**  
                                {measure}  
                                *Ref: {ref}*
                                """)
                        
                        with cols[1]:
                            st.markdown("### Mesures d'acceptation (A)")
                            for i, measure in enumerate(measures_dict.get('A', []), 1):
                                ref = refs_dict.get(f"A-{i}", "")
                                st.markdown(f"""
                                📌 **Mesure {i}:**  
                                {measure}  
                                *Ref: {ref}*
                                """)
                            
                            st.markdown("### Mesures de refus (F)")
                            for i, measure in enumerate(measures_dict.get('F', []), 1):
                                ref = refs_dict.get(f"F-{i}", "")
                                st.markdown(f"""
                                📌 **Mesure {i}:**  
                                {measure}  
                                *Ref: {ref}*
                                """)
                        
                        with cols[2]:
                            st.markdown("### Mesures de transfert (T)")
                            for i, measure in enumerate(measures_dict.get('T', []), 1):
                                ref = refs_dict.get(f"T-{i}", "")
                                st.markdown(f"""
                                📌 **Mesure {i}:**  
                                {measure}  
                                *Ref: {ref}*
                                """)
                    
                    with tab_practices:
                        st.markdown("### 📋 Bonnes pratiques et indicateurs")
                        for practice, kpis in practices_dict.items():
                            with st.expander(practice):
                                for kpi in kpis:
                                    st.markdown(f"📊 {kpi}")
                    
                    # Stockage pour export
                    results.append({
                        "Processus": processus,
                        "Référence": PROCESSES[famille][processus],
                        "Risque": risk,
                        "Mesures": measures_text,
                        "Bonnes_Pratiques_KPIs": practices_text
                    })
                
                progress_bar.progress((idx + 1) / len(selected_risks))
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
                        for measure, count in sorted(common.items(), key=lambda x: x[1], reverse=True):
                            st.markdown(f"• {measure} _(utilisée dans {count} risques)_")
                        st.write("")
            
            # Export Excel
            if results:
                # Création des DataFrames
                df_main = pd.DataFrame(results)
                
                df_measures = pd.DataFrame([
                    {
                        "Risque": risk,
                        "Catégorie": cat,
                        "Mesure": measure,
                        "Référence": refs_dict[f"{cat}-{i+1}"]
                    }
                    for risk, (measures_dict, refs_dict) in zip(all_measures.keys(), zip(all_measures.values(), all_refs.values()))
                    for cat, measures in measures_dict.items()
                    for i, measure in enumerate(measures)
                ])
                
                df_practices = pd.DataFrame([
                    {
                        "Risque": risk,
                        "Bonne Pratique": practice,
                        "KPI": kpi
                    }
                    for risk, practices in all_practices.items()
                    for practice, kpis in practices.items()
                    for kpi in kpis
                ])
                
                df_standards = pd.DataFrame([
                    {
                        "Standard": standard,
                        "Section": section,
                        "Description": desc if isinstance(desc, str) else ", ".join(desc)
                    }
                    for standard, sections in STANDARDS.items()
                    for section, desc in sections.items()
                ])
                
                # Export avec toutes les feuilles
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_main.to_excel(writer, sheet_name='Vue générale', index=False)
                    df_measures.to_excel(writer, sheet_name='Mesures détaillées', index=False)
                    df_practices.to_excel(writer, sheet_name='Bonnes Pratiques & KPIs', index=False)
                    df_standards.to_excel(writer, sheet_name='Référentiel', index=False)
                    
                    if len(selected_risks) > 1:
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
                        df_common.to_excel(writer, sheet_name='Mesures communes', index=False)
                
                st.download_button(
                    label="📥 Télécharger le rapport complet",
                    data=buffer.getvalue(),
                    file_name=f"mesures_remediation_{processus}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Génération terminée avec succès!")

if __name__ == "__main__":
    main()
