import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import branca
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Dashboard Optimasi K-Medoids", page_icon="🗃️")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Government Aid Priorities Dashboard',
                           ['About', 'Learn about Data', 'Methodology',
                            'K-Medoids Results', 'Simulation of K-Medoids'],
                           menu_icon='cast',
                           icons=['house', 'gear', 'list-task', 'app'],
                           default_index=0)

# About Page
if selected == 'About':
    st.title('Welcome to the Government Aid Priorities Dashboard!')
    
    st.write('**Optimasi K-Medoids dengan Particle Swarm Optimization** bertujuan untuk menentukan prioritas bantuan pemerintah di Desa Kalipuro. Proyek ini menggunakan metode K-Medoids untuk mengelompokkan data dan Particle Swarm Optimization untuk meningkatkan efisiensi pengelompokan.')
    
    st.write('Dashboard ini dirancang untuk membantu pemangku kepentingan dalam memahami data dan hasil analisis yang dilakukan. Dengan menggunakan visualisasi interaktif, pengguna dapat mengeksplorasi data dan hasil pengelompokan dengan lebih baik.')

    st.title("What's Inside?")
    st.write('Tersedia beberapa menu utama dalam dashboard ini, yaitu: ')
    st.write('**1. About** - Informasi mengenai dashboard dan tujuan proyek.')
    st.write('**2. Learn about Data** - Deskripsi data yang digunakan dalam analisis.')
    st.write('**3. Methodology** - Metode yang digunakan dalam analisis dan pengelompokan data.')
    st.write('**4. K-Medoids Results** - Hasil pengelompokan menggunakan K-Medoids.')
    st.write('**5. Simulation of K-Medoids** - Simulasi untuk melihat bagaimana perubahan data mempengaruhi hasil pengelompokan.')

# Learn about Data
if selected == 'Learn about Data':
    st.title('Descriptive Analysis of Data Used')

    st.write('Data yang digunakan dalam proyek ini mencakup berbagai variabel yang relevan untuk menentukan prioritas bantuan pemerintah. Data ini diambil dari sumber resmi dan mencakup informasi demografis, ekonomi, dan sosial dari Desa Kalipuro.')

    # Import data
    df_data = pd.read_csv('path_to_your_data.csv')  # Update with your data source
    st.write(df_data.head())  # Display the first few rows of the data

    st.write('Menu berikut dapat digunakan untuk memperoleh analisis deskriptif sederhana dari data yang digunakan.')
    
    # Add more analysis or visualizations as needed

# Methodology
if selected == 'Methodology':
    st.title('Methodology of K-Medoids Optimization')

    st.write('Proyek ini menggunakan metode K-Medoids untuk mengelompokkan data berdasarkan kesamaan karakteristik. K-Medoids adalah metode clustering yang mirip dengan K-Means, tetapi lebih robust terhadap outlier.')
    
    st.write('Particle Swarm Optimization (PSO) digunakan untuk mengoptimalkan pemilihan medoid dalam proses clustering. Metode ini terinspirasi oleh perilaku sosial dari burung dan ikan, dan digunakan untuk menemukan solusi optimal dalam ruang pencarian yang kompleks.')

    st.write('Ilustrasi dari proses K-Medoids dan PSO ditunjukkan pada gambar berikut.')
    
    # Add images or illustrations related to K-Medoids and PSO
    # Example:
    # st.image('path_to_your_image.png', caption='Ilustrasi K-Medoids dan PSO', use_column_width=True)

# K-Medoids Results
if selected == 'K-Medoids Results':
    st.title('K-Medoids Clustering Results')

    st.write('Hasil pengelompokan menggunakan K-Medoids ditampilkan di bawah ini. Setiap cluster merepresentasikan kelompok yang memiliki karakteristik serupa.')
    
    # Load the clustering results
    df_results = pd.read_csv('path_to_your_results.csv')  # Update with your results source
    st.write(df_results.head())  # Display the first few rows of the results

    # Visualization of clusters
    # Example: Create a map or chart to visualize the clusters
    # map = folium.Map(location=[-7.576882, 111.819939], zoom_start=7)
    # Add cluster visualization to the map
    # st_folium(map)

# Simulation of K-Medoids
if selected == 'Simulation of K-Medoids':
    st.title('Simulation of K-Medoids Clustering')

    st.write('Simulasi ini memungkinkan pengguna untuk menginput data baru dan melihat bagaimana perubahan tersebut mempengaruhi hasil pengelompokan K-Medoids.')

    # Define columns for user input
    columns = ['variable1', 'variable2', 'variable3']  # Update with your actual variables
    
    # Initialize the DataFrame in session state if it doesn't exist
    if 'df_input' not in st.session_state:
        st.session_state.df_input = pd.DataFrame(columns=columns)
    
    # Streamlit form for user input
    with st.form(key='form-simulation', clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            variable1 = st.text_input('Variable 1', value='', key='variable1')
        with col2:
            variable2 = st.text_input('Variable 2', value='', key='variable2')
        with col3:
            variable3 = st.text_input('Variable 3', value='', key='variable3')
        
        submitted = st.form_submit_button("Store to Data")
        clear_all = st.form_submit_button("Clear all existing data")
    
    # Process the form data
    if submitted:
        user_input = {
            'variable1': validate_number(variable1),
            'variable2': validate_number(variable2),
            'variable3': validate_number(variable3)
        }
    
        # Convert the user input to a DataFrame with a single row
        input_df = pd.DataFrame([user_input], columns=st.session_state.df_input.columns)
        
        # Check for None values
        if input_df.isnull().values.any():
            st.error("Please provide valid numeric input")
        else:
            # Append to the existing DataFrame in session state
            st.session_state.df_input = pd.concat([st.session_state.df_input, input_df], ignore_index=True)
            st.write("Data stored successfully!")
            st.write(st.session_state.df_input)
    
    # Button to clear all existing data
    if clear_all:
        st.session_state.df_input = pd.DataFrame(columns=columns)
        st.write("All existing data has been cleared.")
    
    st.write("")
    if st.button('Predict K-Medoids Clustering!'):
        if len(st.session_state.df_input) < 2:
            st.error("Please provide more data input (minimum: 2)")
        else:
            data_used = np.array(st.session_state.df_input)
            scaler = MinMaxScaler()
            input_scaled = scaler.fit_transform(data_used)      
            
            # Load your K-Medoids model and predict
            # Example:
            # model = load_model('path_to_your_model.pkl')
            # predictions = model.predict(input_scaled)
            # st.success('Predicted clusters are:')
            # st.write(predictions)
 
