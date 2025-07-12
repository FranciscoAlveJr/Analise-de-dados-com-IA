import os
from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Carregar a chave da API do Groq do arquivo .env
# api_key = os.getenv('api_key')
api_key = st.secrets['api_key']

st.title('Assistente de Análise de Dados com IA')

up_file = st.file_uploader('Carregue um arquivo CSV', type=['csv', 'xlsx', 'xls'])

if up_file is not None:
    if up_file.name.endswith('.csv'):
        df = pd.read_csv(up_file)

    elif up_file.name.endswith('.xlsx') or up_file.name.endswith('.xls'):
        df = pd.read_excel(up_file)

    st.write('Primeiras 5 linhas do arquivo:')
    st.write(df.head())
    st.write('Faça uma pergunta sobre os dados:')

    q = st.text_input('Pergunta:')

    if q:
        def criar_agente(df):
            if 'GROQ_API_KEY' not in os.environ:
                os.environ['GROQ_API_KEY'] = api_key
            
            llm = ChatGroq(model='llama-3.3-70b-versatile', max_tokens=200)
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

            return agent
        
        agent = criar_agente(df)

        with st.spinner('Processando...'):
            try:
                resposta = agent.invoke(q)
                st.write(f'Resposta:')
                st.write(resposta['output'])
            except Exception as e:
                st.error(f'Ocorreu um erro: {e}')