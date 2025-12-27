import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import io
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURATION DE LA PAGE =========================
st.set_page_config(
    page_title="Prédiction Souscription Assurance Automobile",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs professionnelles
PRIMARY_COLOR = "#003366"
SECONDARY_COLOR = "#005588"
ACCENT_COLOR = "#00A0DF"
LIGHT_GRAY = "#F5F7FA"
DARK_GRAY = "#333333"

# ========================= CHARGEMENT DES DONNÉES ET MODÈLE =========================
@st.cache_resource
def load_data_and_model():
    df_raw = pd.read_csv('carInsurance_2024 (3).csv')
    
    # Calcul durée appel pour dashboard analytique
    def time_to_seconds(t):
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except:
            return 0
    
    df_raw['CallDuration_min'] = (df_raw['CallEnd'].apply(time_to_seconds) - 
                                  df_raw['CallStart'].apply(time_to_seconds)) / 60.0
    
    # Dashboard : avec CallDuration_min
    df_dashboard = df_raw.drop(['Id', 'CallStart', 'CallEnd'], axis=1)
    df_dashboard.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)
    
    # Prédiction : exactement comme l'entraînement (sans CallDuration_min, Default, HHInsurance, CarLoan)
    columns_to_drop_pred = ['Id', 'CallStart', 'CallEnd', 'CallDuration_min', 'Default', 'HHInsurance', 'CarLoan']
    df_pred_template = df_raw.drop(columns_to_drop_pred, axis=1, errors='ignore')
    df_pred_template.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)
    
    # Chargement du modèle
    model_path = 'modele_assurance_auto.h5'
    if not os.path.exists(model_path):
        st.error(f"Modèle non trouvé : {os.path.abspath(model_path)}")
        return None, None, None, None
    
    model = load_model(model_path)
    
    # Préprocesseur (38 features)
    categorical_cols = ['Job', 'Marital', 'Education', 'Communication', 'LastContactMonth', 'Outcome']
    
    ct = ColumnTransformer(
        [('one_hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    
    X = df_pred_template.drop('CarInsurance', axis=1)
    X_encoded = ct.fit_transform(X)
    
    sc = StandardScaler()
    sc.fit(X_encoded)
    
    return df_dashboard, model, ct, sc

df_dashboard, model, ct, sc = load_data_and_model()

if model is None:
    st.stop()

# ========================= SIDEBAR =========================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    
    st.markdown(f"""
    <h2 style='color: {PRIMARY_COLOR}; font-size: 24px;'>
        Assurance Automobile
    </h2>
    <p style='color: #666; font-size: 14px;'>
        Prédiction de Souscription
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Contexte du Projet")
    st.write("""
    Optimisation des campagnes téléphoniques d'une banque américaine 
    pour la vente d'assurance automobile via un modèle de deep learning.
    """)
    
    st.markdown("### Navigation")
    page = st.radio("Section", ["Dashboard Analytique", "Prédiction Individuelle", "Prédiction en Batch"])
    
    if page == "Dashboard Analytique":
        st.markdown("### Filtres")
        selected_month = st.selectbox("Mois", ['Tous'] + sorted(df_dashboard['LastContactMonth'].unique().tolist()))
        selected_job = st.selectbox("Profession", ['Tous'] + sorted(df_dashboard['Job'].unique().tolist()))
        selected_comm = st.selectbox("Communication", ['Tous'] + sorted(df_dashboard['Communication'].unique().tolist()))
    
    st.markdown("---")
    st.caption("Projet Réseaux de Neurones Profonds\nOusmane Faye — Décembre 2025")

# ========================= FILTRES DASHBOARD =========================
filtered_df = df_dashboard.copy()
if page == "Dashboard Analytique":
    if selected_month != 'Tous':
        filtered_df = filtered_df[filtered_df['LastContactMonth'] == selected_month]
    if selected_job != 'Tous':
        filtered_df = filtered_df[filtered_df['Job'] == selected_job]
    if selected_comm != 'Tous':
        filtered_df = filtered_df[filtered_df['Communication'] == selected_comm]

# ========================= PAGE DASHBOARD ANALYTIQUE =========================
if page == "Dashboard Analytique":
    st.markdown(f"""
    <div style='background-color: {PRIMARY_COLOR}; padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>Dashboard Analytique</h1>
        <p style='color: #ddd; font-size: 18px; margin: 10px 0 0 0;'>
            Facteurs clés de souscription à l'assurance automobile
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_dashboard)
    taux_global = df_dashboard['CarInsurance'].mean() * 100
    taux_filtre = filtered_df['CarInsurance'].mean() * 100
    duree_moy_sub = df_dashboard[df_dashboard['CarInsurance'] == 1]['CallDuration_min'].mean()
    success_prev = df_dashboard[df_dashboard['Outcome'] == 'success']['CarInsurance'].mean() * 100

    with col1:
        st.markdown(f"<div style='background:{LIGHT_GRAY}; padding:20px; border-radius:10px; text-align:center; height:160px; display:flex; flex-direction:column; justify-content:center;'><p style='color:#666; margin:0;'>Clients totaux</p><h3 style='color:{PRIMARY_COLOR}; margin:10px 0 0 0;'>{total:,}</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background:{LIGHT_GRAY}; padding:20px; border-radius:10px; text-align:center; height:160px; display:flex; flex-direction:column; justify-content:center;'><p style='color:#666; margin:0;'>Taux de souscription</p><h3 style='color:{PRIMARY_COLOR}; margin:10px 0 0 0;'>{taux_global:.1f}%</h3><p style='color:#888; font-size:12px; margin:5px 0 0 0;'>{taux_filtre - taux_global:+.1f}% vs actuel</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background:{LIGHT_GRAY}; padding:20px; border-radius:10px; text-align:center; height:160px; display:flex; flex-direction:column; justify-content:center;'><p style='color:#666; margin:0;'>Durée appel moyenne<br>(souscripteurs)</p><h3 style='color:{PRIMARY_COLOR}; margin:10px 0 0 0;'>{duree_moy_sub:.1f} min</h3></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='background:{LIGHT_GRAY}; padding:20px; border-radius:10px; text-align:center; height:160px; display:flex; flex-direction:column; justify-content:center;'><p style='color:#666; margin:0;'>Conversion si succès précédent</p><h3 style='color:{PRIMARY_COLOR}; margin:10px 0 0 0;'>{success_prev:.1f}%</h3></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {DARK_GRAY};'>Analyses Clés</h2>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Taux de Souscription par Profession")
        job_rate = df_dashboard.groupby('Job')['CarInsurance'].mean().sort_values(ascending=True) * 100
        fig_job = px.bar(x=job_rate.values, y=job_rate.index, orientation='h',
                         color=job_rate.values, color_continuous_scale=['#e6f2ff', SECONDARY_COLOR])
        fig_job.update_layout(height=500, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_job, use_container_width=True)

        st.markdown("#### Saisonnalité")
        month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        month_rate = df_dashboard.groupby('LastContactMonth')['CarInsurance'].mean().reindex(month_order) * 100
        fig_month = px.line(x=[m.capitalize() for m in month_rate.index], y=month_rate.values,
                            markers=True, line_shape='spline')
        fig_month.update_traces(line=dict(color=ACCENT_COLOR, width=4))
        st.plotly_chart(fig_month, use_container_width=True)

    with col_right:
        st.markdown("#### Résultat Campagne Précédente")
        outcome_rate = df_dashboard.groupby('Outcome')['CarInsurance'].mean().sort_values(ascending=False) * 100
        fig_outcome = px.bar(x=outcome_rate.index, y=outcome_rate.values,
                             color=outcome_rate.values, color_continuous_scale=['#ffcccc','#ccffcc'],
                             text=outcome_rate.round(1).astype(str)+'%')
        fig_outcome.update_traces(textposition='outside')
        st.plotly_chart(fig_outcome, use_container_width=True)

        st.markdown("#### Canal de Communication")
        comm_rate = df_dashboard.groupby('Communication')['CarInsurance'].mean().sort_values(ascending=False) * 100
        fig_comm = px.bar(x=comm_rate.index, y=comm_rate.values,
                          color=comm_rate.values, color_continuous_scale=['#e6f2ff', SECONDARY_COLOR])
        st.plotly_chart(fig_comm, use_container_width=True)

    st.markdown(f"<h2 style='color: {DARK_GRAY};'>Corrélations Numériques</h2>", unsafe_allow_html=True)
    num_cols = ['Age','Balance','CallDuration_min','NoOfContacts','DaysPassed','PrevAttempts','CarInsurance']
    corr = df_dashboard[num_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale='Blues', text=corr.round(2).values, texttemplate="%{text}"))
    st.plotly_chart(fig_corr, use_container_width=True)

# ========================= PAGE PRÉDICTION INDIVIDUELLE =========================
elif page == "Prédiction Individuelle":
    st.markdown(f"""
    <div style='background-color: {PRIMARY_COLOR}; padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>Prédiction pour un Nouveau Client</h1>
        <p style='color: #ddd; font-size: 18px; margin: 10px 0 0 0;'>
            Estimez la probabilité de souscription avant l'appel
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge", 18, 100, 35)
            job = st.selectbox("Profession", sorted(df_dashboard['Job'].unique()))
            marital = st.selectbox("Statut marital", sorted(df_dashboard['Marital'].unique()))
            education = st.selectbox("Éducation", sorted(df_dashboard['Education'].unique()))
            balance = st.number_input("Solde moyen ($)", -10000.0, 100000.0, 1000.0)
        with c2:
            comm = st.selectbox("Communication", sorted(df_dashboard['Communication'].unique()))
            day = st.number_input("Jour contact", 1, 31, 15)
            month = st.selectbox("Mois contact", sorted(df_dashboard['LastContactMonth'].unique()))
            contacts = st.number_input("Nb contacts campagne", 0, 50, 1)
            days_passed = st.number_input("Jours depuis précédent (-1 si aucun)", -1, 365, -1)
            prev_att = st.number_input("Tentatives précédentes", 0, 50, 0)
            outcome = st.selectbox("Résultat précédent", sorted(df_dashboard['Outcome'].unique()))

        submitted = st.form_submit_button("Prédire la Probabilité")

    if submitted:
        input_data = pd.DataFrame({
            'Age': [age],
            'Job': [job],
            'Marital': [marital],
            'Education': [education],
            'Balance': [balance],
            'Communication': [comm],
            'LastContactDay': [day],
            'LastContactMonth': [month],
            'NoOfContacts': [contacts],
            'DaysPassed': [days_passed],
            'PrevAttempts': [prev_att],
            'Outcome': [outcome]
        })

        X_new = ct.transform(input_data)
        X_scaled = sc.transform(X_new)

        prob = float(model.predict(X_scaled)[0][0])
        decision = "Oui — Prioriser l'appel" if prob > 0.5 else "Non — Faible potentiel"

        st.success(f"**Probabilité de souscription : {prob:.1%}**")
        st.info(f"**Recommandation :** {decision}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de Souscription (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': ACCENT_COLOR},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ========================= PAGE PRÉDICTION EN BATCH =========================
else:
    st.markdown(f"""
    <div style='background-color: {PRIMARY_COLOR}; padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>Prédiction en Batch</h1>
        <p style='color: #ddd; font-size: 18px; margin: 10px 0 0 0;'>
            Importez un fichier CSV ou Excel pour prédire la probabilité de souscription sur plusieurs clients
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **Format requis du fichier** :
    - Colonnes obligatoires (exactement ces noms) :
      Age, Job, Marital, Education, Balance, Communication, LastContactDay, LastContactMonth,
      NoOfContacts, DaysPassed, PrevAttempts, Outcome
    - Aucune autre colonne (pas d'Id, CallStart, CallEnd, Default, HHInsurance, CarLoan)
    """)

    uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.success(f"Fichier chargé : {uploaded_file.name} — {len(batch_df)} clients")

            # Colonnes requises
            required_cols = ['Age', 'Job', 'Marital', 'Education', 'Balance', 'Communication',
                             'LastContactDay', 'LastContactMonth', 'NoOfContacts',
                             'DaysPassed', 'PrevAttempts', 'Outcome']

            missing = [col for col in required_cols if col not in batch_df.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
                st.stop()

            # Préprocessing
            batch_input = batch_df[required_cols].copy()
            batch_input.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)

            X_batch = ct.transform(batch_input)
            X_batch_scaled = sc.transform(X_batch)

            probabilities = model.predict(X_batch_scaled).flatten()

            batch_df['Probability (%)'] = (probabilities * 100).round(2)
            batch_df['Recommendation'] = np.where(probabilities > 0.5, "Prioriser l’appel", "Faible potentiel")

            # Résumé
            st.markdown("### Résumé des Prédictions")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total clients", len(batch_df))
            col2.metric("À prioriser", len(batch_df[batch_df['Recommendation'] == "Prioriser l’appel"]))
            col3.metric("Probabilité moyenne", f"{probabilities.mean()*100:.1f}%")

            # Tableau trié
            st.dataframe(batch_df.sort_values('Probability (%)', ascending=False))

            # Téléchargement
            output = io.BytesIO()
            batch_df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)

            st.download_button(
                label="Télécharger les résultats (CSV)",
                data=output,
                file_name=f"predictions_batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")

# ========================= PIED DE PAGE =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>Projet académique</strong> — Introduction aux Réseaux de Neurones Profonds<br>
    Réalisé par <strong>Ousmane Faye</strong> | Décembre 2025
</div>
""", unsafe_allow_html=True)