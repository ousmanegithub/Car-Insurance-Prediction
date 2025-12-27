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

st.set_page_config(
    page_title="Prédiction Souscription Assurance Automobile",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Fond clair */
    .main, .stApp {
        background-color: #f8f9fc;
        color: #2d3748;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: white;
        color: #2d3748;
        border: 1px solid #cbd5e0;
        border-radius: 8px;
    }
    
    /* Labels */
    label {
        color: #4a5568 !important;
        font-weight: 600;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(90deg, #003366, #005588);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 83, 136, 0.2);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 83, 136, 0.3);
    }
    
    /* Cartes KPI */
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }
    
    /* Header principal */
    .header {
        background: linear-gradient(135deg, #003366, #005588, #0077b6);
        padding: 3.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    .title {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 1.5rem;
        color: #e0eaff;
        margin: 1rem 0 0 0;
        font-weight: 400;
    }
    
    /* Graphiques */
    .plot-container {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header">
    <h1 class="title">Prédiction de Souscription</h1>
    <p class="subtitle">Outil d’intelligence artificielle pour l’optimisation des campagnes d’assurance automobile</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_and_model():
    df_raw = pd.read_csv('carInsurance_2024 (3).csv')
    
    def time_to_seconds(t):
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except:
            return 0
    
    df_raw['CallDuration_min'] = (df_raw['CallEnd'].apply(time_to_seconds) - 
                                  df_raw['CallStart'].apply(time_to_seconds)) / 60.0
    
    df_dashboard = df_raw.drop(['Id', 'CallStart', 'CallEnd'], axis=1)
    df_dashboard.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)
    
    columns_to_drop_pred = ['Id', 'CallStart', 'CallEnd', 'CallDuration_min', 'Default', 'HHInsurance', 'CarLoan']
    df_pred = df_raw.drop(columns_to_drop_pred, axis=1, errors='ignore')
    df_pred.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)
    
    model_path = 'modele_assurance_auto.h5'
    if not os.path.exists(model_path):
        st.error(f"Modèle non trouvé : {os.path.abspath(model_path)}")
        return None, None, None, None
    
    model = load_model(model_path)
    
    categorical_cols = ['Job', 'Marital', 'Education', 'Communication', 'LastContactMonth', 'Outcome']
    
    ct = ColumnTransformer(
        [('one_hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    
    X = df_pred.drop('CarInsurance', axis=1)
    X_encoded = ct.fit_transform(X)
    
    sc = StandardScaler()
    sc.fit(X_encoded)
    
    return df_dashboard, model, ct, sc

df_dashboard, model, ct, sc = load_data_and_model()

if model is None:
    st.stop()


with st.sidebar:
    st.markdown("<div style='text-align:center; padding:1.5rem;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/bank-building.png", width=100)
    st.markdown("<h2 style='color:#003366; text-align:center; margin-top:1rem;'>Banque & Assurance</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio("", ["Dashboard Analytique", "Prédiction Individuelle", "Prédiction en Batch"], label_visibility="collapsed")
    
    st.markdown("### À propos")
    st.caption("""
    Application développée pour l'optimisation des campagnes téléphoniques d'une banque  
    pour la vente d'assurance automobile via un modèle de deep learning. 
    **Réseaux de Neurones Profonds**  
    **Ousmane Faye** — Décembre 2025
    """)


filtered_df = df_dashboard.copy()
if page == "Dashboard Analytique":
    st.markdown("### Filtres d'analyse")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_month = st.selectbox("Mois du contact", ['Tous'] + sorted(df_dashboard['LastContactMonth'].unique().tolist()))
    with col2:
        selected_job = st.selectbox("Profession", ['Tous'] + sorted(df_dashboard['Job'].unique().tolist()))
    with col3:
        selected_comm = st.selectbox("Canal", ['Tous'] + sorted(df_dashboard['Communication'].unique().tolist()))
    
    if selected_month != 'Tous':
        filtered_df = filtered_df[filtered_df['LastContactMonth'] == selected_month]
    if selected_job != 'Tous':
        filtered_df = filtered_df[filtered_df['Job'] == selected_job]
    if selected_comm != 'Tous':
        filtered_df = filtered_df[filtered_df['Communication'] == selected_comm]


if page == "Dashboard Analytique":
    st.markdown("### Indicateurs Clés de Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df_dashboard)
    taux_global = df_dashboard['CarInsurance'].mean() * 100
    taux_filtre = filtered_df['CarInsurance'].mean() * 100
    duree_moy_sub = df_dashboard[df_dashboard['CarInsurance'] == 1]['CallDuration_min'].mean()
    success_prev = df_dashboard[df_dashboard['Outcome'] == 'success']['CarInsurance'].mean() * 100

    with col1:
        st.markdown(f"""
        <div class="card">
            <p style='color:#718096; margin:0; font-size:0.95rem;'>Clients analysés</p>
            <h2 style='color:#003366; margin:0.5rem 0 0 0; font-size:2.5rem;'>{total:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <p style='color:#718096; margin:0; font-size:0.95rem;'>Taux de souscription global</p>
            <h2 style='color:#003366; margin:0.5rem 0 0 0; font-size:2.5rem;'>{taux_global:.1f}%</h2>
            <p style='color:#4c51bf; font-size:0.9rem; margin:0.5rem 0 0 0;'>
                {taux_filtre - taux_global:+.1f}% vs filtre actuel
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
            <p style='color:#718096; margin:0; font-size:0.95rem;'>Durée moyenne appel<br>(souscripteurs)</p>
            <h2 style='color:#003366; margin:0.5rem 0 0 0; font-size:2.5rem;'>{duree_moy_sub:.1f} min</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card">
            <p style='color:#718096; margin:0; font-size:0.95rem;'>Conversion si succès précédent</p>
            <h2 style='color:#003366; margin:0.5rem 0 0 0; font-size:2.5rem;'>{success_prev:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Analyses Stratégiques")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Taux de Souscription par Profession")
        job_rate = df_dashboard.groupby('Job')['CarInsurance'].mean().sort_values(ascending=True) * 100
        fig_job = px.bar(x=job_rate.values, y=job_rate.index, orientation='h',
                         color=job_rate.values, color_continuous_scale=['#e2e8f0', '#003366'])
        fig_job.update_layout(height=500, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_job, use_container_width=True)

        st.markdown("#### Saisonnalité des Campagnes")
        month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        month_rate = df_dashboard.groupby('LastContactMonth')['CarInsurance'].mean().reindex(month_order) * 100
        fig_month = px.line(x=[m.capitalize() for m in month_rate.index], y=month_rate.values,
                            markers=True, line_shape='spline')
        fig_month.update_traces(line=dict(color='#003366', width=5))
        st.plotly_chart(fig_month, use_container_width=True)

    with col_right:
        st.markdown("#### Impact du Résultat Précédent")
        outcome_rate = df_dashboard.groupby('Outcome')['CarInsurance'].mean().sort_values(ascending=False) * 100
        fig_outcome = px.bar(x=outcome_rate.index, y=outcome_rate.values,
                             color=outcome_rate.values, color_continuous_scale=['#fed7d7', '#9ae6b4'],
                             text=outcome_rate.round(1).astype(str)+'%')
        fig_outcome.update_traces(textposition='outside')
        st.plotly_chart(fig_outcome, use_container_width=True)

        st.markdown("#### Efficacité par Canal")
        comm_rate = df_dashboard.groupby('Communication')['CarInsurance'].mean().sort_values(ascending=False) * 100
        fig_comm = px.bar(x=comm_rate.index, y=comm_rate.values,
                          color=comm_rate.values, color_continuous_scale=['#e2e8f0', '#003366'])
        st.plotly_chart(fig_comm, use_container_width=True)

    st.markdown("### Corrélations entre Variables Numériques")
    num_cols = ['Age','Balance','CallDuration_min','NoOfContacts','DaysPassed','PrevAttempts','CarInsurance']
    corr = df_dashboard[num_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale='Blues', text=corr.round(2).values, texttemplate="%{text}"))
    st.plotly_chart(fig_corr, use_container_width=True)


elif page == "Prédiction Individuelle":
    st.markdown("### Saisie des Informations Client")
    
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
                'bar': {'color': "#003366"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}
            }
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("### Prédiction en Batch")
    st.info("""
    **Format requis** :
    - Colonnes : Age, Job, Marital, Education, Balance, Communication, LastContactDay, LastContactMonth, NoOfContacts, DaysPassed, PrevAttempts, Outcome
    - Pas de Default, HHInsurance, CarLoan, Id, CallStart, CallEnd
    """)

    uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            required_cols = ['Age', 'Job', 'Marital', 'Education', 'Balance', 'Communication',
                             'LastContactDay', 'LastContactMonth', 'NoOfContacts',
                             'DaysPassed', 'PrevAttempts', 'Outcome']

            missing = [col for col in required_cols if col not in batch_df.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
                st.stop()

            batch_input = batch_df[required_cols].copy()
            batch_input.fillna({'Job': 'unknown', 'Education': 'unknown', 'Communication': 'unknown', 'Outcome': 'unknown'}, inplace=True)

            X_batch = ct.transform(batch_input)
            X_batch_scaled = sc.transform(X_batch)

            probabilities = model.predict(X_batch_scaled).flatten()

            batch_df['Probability (%)'] = (probabilities * 100).round(2)
            batch_df['Recommendation'] = np.where(probabilities > 0.5, "Prioriser l’appel", "Faible potentiel")

            st.markdown("### Résumé des Prédictions")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total clients", len(batch_df))
            col2.metric("À prioriser", len(batch_df[batch_df['Recommendation'] == "Prioriser l’appel"]))
            col3.metric("Probabilité moyenne", f"{probabilities.mean()*100:.1f}%")

            st.dataframe(batch_df.sort_values('Probability (%)', ascending=False))

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

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4a5568; padding: 3rem;'>
    <p style='margin: 0.5rem; font-size: 1.2rem;'>
        <strong>Projet IA</strong> —  Réseaux de Neurones Profonds
    </p>
    <p style='margin: 0.5rem;'>
        Réalisé par <strong>Ousmane Faye</strong> | Décembre 2025
    </p>
    
</div>
""", unsafe_allow_html=True)