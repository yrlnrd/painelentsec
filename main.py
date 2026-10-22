
## Sempre iniciar no terminal com  
##       'streamlit run main.py'
## para a criar a aplicação em localhost.
streamlit run my_app.py --server.port=80
##### Importações
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

##### Mensagem de boas-vindas
st.set_page_config(page_title="Painel de entrevistas 2025", page_icon="🎤")

"""
# Visualização de Entrevistas
"""

##### Widget de upload dos dados
file_upload = st.file_uploader(label="Faça upload da tabela a seguir:", type=["xlsx"])

#### Verificação do upload
if file_upload:

    # Leitura dos dados
    df = pd.read_excel(file_upload)
    df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y").dt.date

    #-------------------------------------------------------------
# PANORAMA INICIAL ############################################
    # -------------------------------------------------------------
    qtd_codsec_unicos = df['codsec'].nunique()
    qtd_registros = len(df)
    qtd_veiculosmidia = df['veiculo'].nunique()
    qtd_entrevistados = df['fonte'].nunique()

    # Exibindo em destaque usando st.metric
    """
    ## Panorama Geral
    """
     #Colunagem dupla
    col1, col2 = st.columns([1, 2])
    with col1:
          st.metric(label="Secretarias/Órgãos mobilizados:", value=qtd_codsec_unicos)
          st.metric(label="Total de Entrevistas:", value=qtd_registros)
    with col2:
          st.metric(label="Veículos de Mídia mobilizados:", value=qtd_veiculosmidia)
          st.metric(label="Estimativa de entrevistados:", value=qtd_entrevistados)

    """
    ## Contadores 
    """

    #-------------------------------------------------------------
# SECRETARIAS E VINCULADAS ###################################
    # ------------------------------------------------------------
    exp1 = st.expander("🏦 Secretarias e Vinculadas")

    exp1.markdown('Global = Secretarias e Vinculadas são unificadas (SEC + CODSEC)')
    exp1.markdown('Absoluta = Secretarias e Vinculadas são contadas independentes (SEC e CODSEC)')
    tab_sec, tab_vinc = exp1.tabs(['Global', 'Absoluta'])
    with tab_sec:
        #Contagem de ocorrências por GLOBAL
            df_sec_qtd = df["sec"].value_counts().reset_index()
            df_sec_qtd.columns =['Secretaria', 'Quantidade']
            #Top 10 secretarias
            df_top10_sec = df_sec_qtd.head(10)
            #Ranking - Gráfico de barras horizontal
            fig_top10_sec = px.bar(
                      df_top10_sec, 
                      x='Quantidade', 
                      y='Secretaria', 
                      orientation='h', 
                      title='🏆 Top 10 Secretarias', 
                      labels={'Quantidade': 'Total de Entrevistas', 'Secretaria': 'Secretaria'}, 
                      text='Quantidade')
                 #Inverte o eixo Y para que a secretaria maior fique no topo
            fig_top10_sec.update_layout(yaxis={'categoryorder': 'total ascending'},height=max(400, len(df_top10_sec)*25))
            fig_top10_sec.update_traces(textposition='outside')
            st.plotly_chart(fig_top10_sec)
        #Exibe a lista completa
            st.markdown('#### Listagem geral Global')
            geral_sec_display = df_sec_qtd.copy()
            geral_sec_display.index = range(1, len(geral_sec_display) + 1)
            st.dataframe(
             geral_sec_display,
             column_config={
                  "Secretaria": st.column_config.TextColumn('Secretaria'),
                  "Quantidade": st.column_config.NumberColumn('Entrevistas')})
    with tab_vinc:
        #Contagem de ocorrências por vinculadas
            df_vinc_qtd = df["codsec"].value_counts().reset_index()
            df_vinc_qtd.columns =['Órgão', 'Quantidade']
            #Top 10 Absolutos
            df_top10_vinc = df_vinc_qtd.head(10)
            #Ranking - Gráfico de barras horizontal
            fig_top10_vinc = px.bar(
                      df_top10_vinc, 
                      x='Quantidade', 
                      y='Órgão', 
                      orientation='h', 
                      title='🏆 Top 10 Absoluto', 
                      labels={'Quantidade': 'Total de Entrevistas', 'Órgão': 'Secretaria ou Vinculada'}, 
                      text='Quantidade')
                 #Inverte o eixo Y para que a secretaria maior fique no topo
            fig_top10_vinc.update_layout(yaxis={'categoryorder': 'total ascending'},height=max(400, len(df_top10_vinc)*25))
            fig_top10_vinc.update_traces(textposition='outside')
            st.plotly_chart(fig_top10_vinc)
        #Exibe a lista completa
            st.markdown('#### Listagem geral Absoluta')
            geral_vinc_display = df_vinc_qtd.copy()
            geral_vinc_display.index = range(1, len(geral_vinc_display) + 1) #Índice iniciando em 1
            st.dataframe(
             geral_vinc_display,
             column_config={
                  "Sec ou Vinc": st.column_config.TextColumn('Secretaria / Vinculada'),
                  "Quantidade": st.column_config.NumberColumn('Entrevistas')})

    #-------------------------------------------------------------
# ENTREVISTADOS ###################################
    # ------------------------------------------------------------
    with st.expander("👥 Entrevistados", expanded=False):
    # Controle de estado para expandir a visualização
        if 'mostrar_top30_fonte' not in st.session_state:
               st.session_state.mostrar_top30_fonte = False
    # Interface de Botão de Alternância e Barra de busca
        col_btn, col_busca = st.columns([1, 2])
        with col_btn:
            st.write("")
            if st.button(
                   'Mostrar Top 30' if not st.session_state.mostrar_top30_fonte else 'Voltar para Top 10',
                   key='btn_toggle_fonte'
            ):
                st.session_state.mostrar_top30_fonte = not st.session_state.mostrar_top30_fonte
        with col_busca:
            termo_busca = st.text_input('🔍 Buscar por termo', value="")
    # Filtragem por busca - se houver termo digitado
        df_fonte = df.copy()
        if termo_busca.strip():
            df_fonte = df_fonte[df_fonte['fonte'].astype(str).str.contains(termo_busca, case=False, na=False)]
    # Contagem das frequências
        df_fonte_qtd = df_fonte['fonte'].value_counts().reset_index()
        df_fonte_qtd.columns = ['Fonte', 'Quantidade']

    #Definição do limite top 10 ou top 30
        limite_fonte = 30 if st.session_state.mostrar_top30_fonte else 10
        df_fonte_top = df_fonte_qtd.head(limite_fonte)

        if not df_fonte_top.empty:
            #Gráfico de barras
            fig_fonte = px.bar(
                   df_fonte_top,
                   x='Quantidade',
                   y='Fonte',
                   orientation='h',
                   title=f"Top {len(df_fonte_top)}" + (f" (Filtrado por: '{termo_busca}')" if termo_busca else""),
                   labels={'Quantidade': 'Total de Citações', 'Fonte':'Fonte'},
                   text='Quantidade'
            )
            fig_fonte.update_layout(yaxis={'categoryorder': 'total ascending'},height=max(400, len(df_fonte_top) * 30))
            fig_fonte.update_traces(textposition='outside')
            st.plotly_chart(fig_fonte, width="stretch")
        else:
            st.warning('Nenhuma fonte encontrada para o termo pesquisado.')
    #-------------------------------------------------------------
# MÍDIA ###################################
    # ------------------------------------------------------------
    with st.expander("📻 Veículos de Mídia", expanded=False):
        tab_midia, tab_veiculo = st.tabs(["📊 Distribuição por Mídia", "🏆 Top 20 Veículos"])
    # --- Aba 1: Distribuição por Tipo de Mídia ---
        with tab_midia:
            df_midia_qtd = df['midia'].value_counts().reset_index()
            df_midia_qtd.columns = ['Mídia', 'Quantidade']

            fig_midia = px.pie(
                df_midia_qtd,
                names='Mídia',
                values='Quantidade',
                title="Distribuição por Tipo de Mídia",
                hole=0.4  # Gráfico em formato de rosca
            )
            fig_midia.update_traces(textinfo='percent+label+value')

            st.plotly_chart(fig_midia, width="stretch")

    # --- Aba 2: Top 20 Veículos ---
        with tab_veiculo:
            dt_veiculo_qtd = df['veiculo'].value_counts().reset_index()
            dt_veiculo_qtd.columns = ['Veículo', 'Quantidade']
        
            # Seleciona os Top 20
            df_top20_veiculos = dt_veiculo_qtd.head(20)
        
            # Reindexa de 1 a 30 para exibir a classificação explicitamente
            top20_veic_display = df_top20_veiculos.copy()
            top20_veic_display.index = range(1, len(top20_veic_display) + 1)

            st.dataframe(
                top20_veic_display,
                column_config={
                    "Veículo": st.column_config.TextColumn("Nome do Veículo"),
                    "Quantidade": st.column_config.NumberColumn("Total de Ocorrências")
                },
                width="stretch"
            )
    #-------------------------------------------------------------
# REGIÕES ###################################
    # ------------------------------------------------------------
    with st.expander("🗺️ Mapeamento por Região de Planejamento", expanded=False):

    # Dicionário de coordenadas centrais das 14 macrorregiões do Ceará
        COORD_REGIOES = {
        'Grande Fortaleza': {'lat': -3.7319, 'lon': -38.5267},
        'Litoral Oeste / Vale do Curu': {'lat': -3.5350, 'lon': -39.2810},
        'Litoral Norte': {'lat': -2.8870, 'lon': -40.1190},
        'Litoral Leste': {'lat': -4.5610, 'lon': -37.7690},
        'Serra da Ibiapaba': {'lat': -3.8120, 'lon': -40.9160},
        'Sertão de Sobral': {'lat': -3.6890, 'lon': -40.3480},
        'Maciço de Baturité': {'lat': -4.3290, 'lon': -38.8820},
        'Sertão de Canindé': {'lat': -4.3580, 'lon': -39.3120},
        'Sertão Central': {'lat': -5.1830, 'lon': -39.2920},
        'Sertão dos Crateús': {'lat': -5.1720, 'lon': -40.6690},
        'Sertão dos Inhamuns': {'lat': -5.9960, 'lon': -40.2910},
        'Vale do Jaguaribe': {'lat': -5.1430, 'lon': -38.0980},
        'Centro Sul': {'lat': -6.3580, 'lon': -39.2980},
        'Cariri': {'lat': -7.2270, 'lon': -39.3150}
        }

    # Tratamento dos dados
        df_regiao_counts = df['regiao'].value_counts().reset_index()
        df_regiao_counts.columns = ['regiao', 'contagem']

    # Mapeamento para lat/lon
        df_regiao_counts['lat'] = df_regiao_counts['regiao'].map(lambda r: COORD_REGIOES.get(r, {}).get('lat'))
        df_regiao_counts['lon'] = df_regiao_counts['regiao'].map(lambda r: COORD_REGIOES.get(r, {}).get('lon'))

    # Remoção de nulos
        df_mapa = df_regiao_counts.dropna(subset=['lat', 'lon']).copy()

        if not df_mapa.empty:
        # Ajuste de escala do raio/tamanho dos pontos
            df_mapa['radius'] = df_mapa['contagem'] * 800

        # Visualização de estado (ViewState)
            view_state = pdk.ViewState(
                latitude=-5.2000,
                longitude=-39.5000,
                zoom=6,
                pitch=0
            )

        # Camada Scatterplot Layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_mapa,
                get_position=["lon", "lat"],
                get_color="[255, 102, 0, 180]",  # Cor Laranja RGBA
                get_radius="radius",
                pickable=True,
                auto_highlight=True
            )

        # Objeto pdk.Deck com map_style=None para sincronizar nativamente com o tema do Streamlit
            deck = pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    "html": "<b>Região:</b> {regiao}<br/><b>Total de Entrevistas:</b> {contagem}",
                    "style": {"backgroundColor": "#1E1E1E", "color": "#FFFFFF", "fontSize": "13px"}
                }
            )

        # Renderização do gráfico
            st.pydeck_chart(deck, width="stretch")
        else:
            st.warning("Nenhuma correspondência de coordenadas encontrada para os valores da variável 'regiao'.")           

        dt_regioes_qtd = df['regiao'].value_counts().reset_index()
        dt_regioes_qtd.columns = ['Região', 'Quantidade']
        
        st.dataframe(
            dt_regioes_qtd,
                column_config={
                    "Região": st.column_config.TextColumn("Região"),
                    "Quantidade": st.column_config.NumberColumn("Total de Ocorrências")
                },
                width="stretch", hide_index=True
            )
    # -------------------------------------------------------------
# Análise de Pareto da Variável 'assunto' (Usando N-Grams)
    # -------------------------------------------------------------
    with st.expander("📈 Assuntos - Principais Temas (N-Grams)", expanded=False):

        # Remove valores nulos
        df_pareto_ngram = df.dropna(subset=['assunto']).copy()

        if not df_pareto_ngram.empty:
            # Stopwords para limpar os termos inúteis
            stopwords_pt = {
                "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", 
                "para", "com", "por", "um", "uma", "uns", "umas", "e", "ou", "o", "a", 
                "os", "as", "que", "se", "sobre", "mais", "como", "mas", "ao", "aos",
                "sua", "seu", "pelo", "pela"
            }

            # Extração de Bigramas
            def get_bigrams(text):
                text = re.sub(r'[^\w\s]', '', str(text).lower())
                words = [w for w in text.split() if w not in stopwords_pt and len(w) > 2]
                if len(words) < 2:
                    return []
                return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

            # Aplica a extração e expande
            df_pareto_ngram['bigrams'] = df_pareto_ngram['assunto'].apply(get_bigrams)
            df_exploded = df_pareto_ngram.explode('bigrams').dropna(subset=['bigrams'])

            if not df_exploded.empty:
                # 1. Contagem de frequências dos bigramas
                df_pareto = df_exploded['bigrams'].value_counts().reset_index()
                df_pareto.columns = ['Tema (Bigrama)', 'Frequência']

                # 2. Cálculos do Pareto (Percentual e Acumulado)
                total_ocorrencias = df_pareto['Frequência'].sum()
                df_pareto['Percentual'] = (df_pareto['Frequência'] / total_ocorrencias) * 100
                df_pareto['Acumulado (%)'] = df_pareto['Percentual'].cumsum()

            # Filtramos os Top 20 para visualização
                top_n = 20
                df_pareto_top = df_pareto.head(top_n)

            # 3. Construção do Gráfico com 2 Eixos
                fig_pareto = go.Figure()

            # Eixo primário (Barras): Frequência
                fig_pareto.add_trace(go.Bar(
                    x=df_pareto_top['Tema (Bigrama)'],
                    y=df_pareto_top['Frequência'],
                    name='Frequência Absoluta',
                    marker_color='royalblue'
                ))

            # Eixo secundário (Linha): Percentual Acumulado
                fig_pareto.add_trace(go.Scatter(
                    x=df_pareto_top['Tema (Bigrama)'],
                    y=df_pareto_top['Acumulado (%)'],
                    name='Acumulado (%)',
                    mode='lines+markers',
                    marker=dict(color='red', size=8),
                    line=dict(width=3),
                    yaxis='y2'
                ))

            # Adiciona a linha de corte dos 80%
                fig_pareto.add_hline(y=80, line_dash="dash", line_color="gray", yref="y2",
                                     annotation_text="Corte de 80%", annotation_position="top left")

                # Configuração do Layout e Eixos
                fig_pareto.update_layout(
                    title=f"Gráfico de Pareto - Top {top_n} Temas Núcleo",
                    xaxis=dict(title='Temas (Bigramas)', tickangle=45),
                    yaxis=dict(title='Frequência Absoluta', side='left'),
                    yaxis2=dict(
                        title='Percentual Acumulado (%)',
                        side='right',
                        overlaying='y',
                        range=[0, 105],
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    margin=dict(b=100) # Margem inferior para os rótulos não cortarem
                )

                st.plotly_chart(fig_pareto, width="stretch")

            # 4. Insight da Regra 80/20
            assuntos_80_porcento = df_pareto[df_pareto['Acumulado (%)'] <= 80]
            total_assuntos_unicos = len(df_pareto)
            qtd_80 = len(assuntos_80_porcento)
            
            st.info(
                f"💡 **Insight de Pareto (Regra 80/20):** "
                f"Apenas **{qtd_80} temas nucleares** (bigramas) representam aproximadamente 80% do volume total de assuntos "
                f"(de um total de {total_assuntos_unicos} combinações únicas). Focar nesses temas garante alinhamento com a maior parte da agenda de mídia."
            )
        else:
            st.warning("Não foi possível extrair pares de palavras da variável 'assunto'.")
    # -------------------------------------------------------------
# Nuvem de Palavras da Variável 'assunto'
    # -------------------------------------------------------------
    with st.expander("☁️ Assuntos - Nuvem de Palavras", expanded=False):

    # Trata os dados da coluna 'assunto', removendo valores nulos
        textos_assunto = df['assunto'].dropna().astype(str)

        if not textos_assunto.empty:
        # Junta todos os textos em uma única string
            texto_completo = " ".join(textos_assunto)

        # Configuração de palavras a ignorar (stopwords em português)
            stopwords_pt = set(STOPWORDS)
            stopwords_pt.update([
                "Ceará", "Fortaleza", "Região", "sobre", "ano", "Programação", "Programa",
                "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
                "para", "com", "por", "um", "uma", "uns", "umas", "e", "ou", "o", "a", "os", "as"
            ])

        # Geração da nuvem de palavras
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=stopwords_pt,
                colormap="viridis",
                max_words=100
            ).generate(texto_completo)

        # Renderização usando Matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")  # Oculta os eixos X e Y

        # Exibição no Streamlit
            st.pyplot(fig)
        else:
            st.warning("Não há dados disponíveis na variável 'assunto' para gerar a nuvem de palavras.")
# Data ###################################
    # ------------------------------------------------------------
    with st.expander("📅 Frequência Temporal", expanded=False):

        df_data = df.copy()
        df_data['data_dt'] = pd.to_datetime(df_data['data'])

    # --- 1. Alerta de Pico de Atividade ---
        contagem_por_dia = df_data['data_dt'].value_counts()
    
        if not contagem_por_dia.empty:
            pico_data = contagem_por_dia.idxmax()
            pico_quantidade = contagem_por_dia.max()
            pico_data_str = pico_data.strftime("%d/%m/%Y")

            st.error(
                f"🔥 **Pico de Atividade:** O dia com maior número de entrevistas/ocorrências foi **{pico_data_str}**, "
                f"com um total de **{pico_quantidade} registros**!"
            )

        st.markdown("---")

        col_mapa, col_semana = st.columns([1.3, 1])

    # --- 2. Mapa de Calor em Formato de Calendário (Semanas x Dias da Semana) ---
        with col_mapa:
            st.markdown("### 🗓️ Mapa de Calor Diário (Calendário por Mês)")

        # Agrupa contagem por dia exato
            df_diario = df_data.groupby(df_data['data_dt'].dt.date).size().reset_index(name='Ocorrências')
            df_diario['data_dt'] = pd.to_datetime(df_diario['data_dt'])

        # Extrai Dia do Mês e Nome/Número do Mês
            df_diario['Dia'] = df_diario['data_dt'].dt.day
            df_diario['AnoMes'] = df_diario['data_dt'].dt.strftime('%Y-%m')

        # Matriz pivot: Dias (1 a 31) nas colunas e Meses nas linhas
            pivot_dias = df_diario.pivot_table(
                index='AnoMes',
                columns='Dia',
                values='Ocorrências',
                aggfunc='sum',
                fill_value=0
            )

        # Garante que todos os dias de 1 a 31 existam nas colunas
            todos_dias = list(range(1, 32))
            pivot_dias = pivot_dias.reindex(columns=todos_dias, fill_value=0)

            fig_cal_dias = px.imshow(
                pivot_dias,
                labels=dict(x="Dia do Mês", y="Mês", color="Ocorrências"),
                x=todos_dias,
                y=pivot_dias.index,
                color_continuous_scale="Reds",
                aspect="auto",
                title="Ocorrências por Dia do Mês"
            )

            fig_cal_dias.update_xaxes(dtick=1)  # Exibe todos os números de dias (1 ao 31)
            fig_cal_dias.update_layout(height=320)

            st.plotly_chart(fig_cal_dias, width="stretch")

    # --- 3. Distribuição por Dia da Semana ---
        with col_semana:
            st.markdown("### 📊 Acumulado por Dia da Semana")

            dias_semana_ordem = [
                'Segunda-feira', 'Terça-feira', 'Quarta-feira',
                'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo'
            ]

            df_data['dia_semana_num'] = df_data['data_dt'].dt.dayofweek
            mapa_dias = {
                0: 'Segunda-feira', 1: 'Terça-feira', 2: 'Quarta-feira',
                3: 'Quinta-feira', 4: 'Sexta-feira', 5: 'Sábado', 6: 'Domingo'
            }
            df_data['dia_semana_nome'] = df_data['dia_semana_num'].map(mapa_dias)

            df_semana_counts = df_data['dia_semana_nome'].value_counts().reindex(dias_semana_ordem, fill_value=0).reset_index()
            df_semana_counts.columns = ['Dia da Semana', 'Quantidade']

            fig_semana = px.bar(
                df_semana_counts,
                x='Dia da Semana',
                y='Quantidade',
                title="Volume por Dia",
                color='Quantidade',
                color_continuous_scale="Blues",
                text='Quantidade'
            )
            fig_semana.update_traces(textposition='outside')
            fig_semana.update_layout(showlegend=False, height=300)

            st.plotly_chart(fig_semana, width="stretch")
    # -------------------------------------------------------------


    """
    ## Análises 
    """
# -------------------------------------------------------------
# Análise de Concentração de Veículos por Secretaria (Índice HHI)
# -------------------------------------------------------------
    with st.expander("📺 Concentração de Veículos por Secretaria (Índice HHI)", expanded=False):

        st.markdown("""
        **O que avalia esta análise?**
        Mede se a presença de cada **Secretaria (`sec`)** está pulverizada entre múltiplos veículos de imprensa ou se há dependência/concentração excessiva em apenas alguns **Veículos (`veiculo`)**.
        * **HHI < 1.500:** Presença diversificada (Baixa concentração em veículos específicos).
        * **1.500 ≤ HHI ≤ 2.500:** Concentração moderada em poucos veículos.
        * **HHI > 2.500:** Alta concentração (Sua comunicação depende fortemente de 1 ou 2 veículos).
        """)

        # Remove valores nulos nas duas colunas
        df_hhi_veic = df.dropna(subset=['sec', 'veiculo']).copy()

        if not df_hhi_veic.empty:
            # --- 1. Cálculo do HHI por Secretaria em relação aos Veículos ---
            # Total de entrevistas por secretaria
            sec_totals = df_hhi_veic.groupby('sec').size().reset_index(name='total_sec')
        
            # Ocorrências de cada veículo dentro de cada secretaria
            sec_veic_counts = df_hhi_veic.groupby(['sec', 'veiculo']).size().reset_index(name='count_veiculo')
        
            # Mescla dados para calcular a "fatia de mercado" (share) do veículo dentro da secretaria
            hhi_df = sec_veic_counts.merge(sec_totals, on='sec')
            hhi_df['share'] = (hhi_df['count_veiculo'] / hhi_df['total_sec']) * 100
            hhi_df['share_sq'] = hhi_df['share'] ** 2
        
            # Soma dos quadrados das porcentagens = HHI
            hhi_por_sec = hhi_df.groupby('sec')['share_sq'].sum().reset_index(name='HHI')
            hhi_por_sec = hhi_por_sec.merge(sec_totals, on='sec')
        
            # Filtro de relevância estatística (mínimo de 3 aparições para não gerar HHI distorcido de 10.000)
            min_insercoes = 3
            hhi_por_sec = hhi_por_sec[hhi_por_sec['total_sec'] >= min_insercoes].sort_values(by='HHI', ascending=False)

            if not hhi_por_sec.empty:
                col_metricas, col_grafico = st.columns([1, 2])

                with col_metricas:
                    # Destaques estatísticos
                    sec_mais_concentrada = hhi_por_sec.iloc[0]
                    sec_mais_pulverizada = hhi_por_sec.iloc[-1]

                    st.metric(
                        label="Secretaria Mais Concentrada",
                        value=sec_mais_concentrada['sec'],
                        delta=f"HHI {sec_mais_concentrada['HHI']:.0f}",
                        delta_color="inverse"
                    )
                
                    st.metric(
                        label="Secretaria Mais Diversificada",
                        value=sec_mais_pulverizada['sec'],
                        delta=f"HHI {sec_mais_pulverizada['HHI']:.0f}",
                        delta_color="normal"
                    )

                    st.caption(f"Considerando apenas órgãos com pelo menos {min_insercoes} inserções mapeadas.")

                with col_grafico:
                    # Gráfico de Barras com Plotly
                    fig_hhi_veic = px.bar(
                        hhi_por_sec,
                        x='HHI',
                        y='sec',
                        orientation='h',
                        color='HHI',
                        color_continuous_scale=[(0, "green"), (0.15, "green"), (0.25, "yellow"), (1, "red")],
                        range_color=[0, 10000],
                        labels={'sec': 'Secretaria / Órgão', 'HHI': 'Índice HHI'},
                        title="Índice HHI de Veículos por Secretaria",
                        hover_data={'total_sec': True}
                    )
                
                    # Linhas de referência dos cortes do HHI
                    fig_hhi_veic.add_vline(x=1500, line_dash="dash", line_color="green", annotation_text="Moderado (1500)")
                    fig_hhi_veic.add_vline(x=2500, line_dash="dash", line_color="red", annotation_text="Alto (2500)")
                
                    fig_hhi_veic.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
                
                    st.plotly_chart(fig_hhi_veic, width="stretch")

                # --- Alerta de Dependência de Mídia ---
                sec_alta_conc = hhi_por_sec[hhi_por_sec['HHI'] > 2500]
                if not sec_alta_conc.empty:
                    nomes_conc = ", ".join(sec_alta_conc['sec'].tolist())
                    st.warning(
                        f"⚠️ **Alerta de Concentração Mídia-Secretaria:** Os seguintes órgãos possuem um HHI superior a 2.500: "
                        f"**{nomes_conc}**. Isso indica forte dependência de um número muito restrito de veículos de imprensa."
                    )
            else:
                st.info(f"Não há secretarias com ao menos {min_insercoes} inserções para gerar o cálculo do HHI.")
        else:
            st.warning("Dados insuficientes nas colunas 'sec' e 'veiculo' para calcular a concentração HHI.")

# -------------------------------------------------------------
# Módulo Alternativo de Concentração de Porta-Vozes (sec × fonte)
# -------------------------------------------------------------
    with st.expander("📊 Índices de Concentração e Diversidade de Porta-Vozes", expanded=False):

        df_conc = df.dropna(subset=['sec', 'fonte']).copy()

        if not df_conc.empty:
            st.markdown("""
            **Métricas de Concentração Disponíveis:**
            * **CR3 (% Top 3):** Porcentagem de espaço ocupada pelos 3 principais porta-vozes da secretaria.
            * **Gini-Simpson (Diversidade %):** Probabilidade de duas entrevistas aleatórias serem de fontes diferentes.
            * **Equitabilidade de Pielou (0 a 1):** Grau de equilíbrio na distribuição da fala entre as fontes cadastradas.
            """)

        # Função para calcular os índices por grupo
            def calcular_indices_concentracao(grupo):
                total_obs = len(grupo)
                contagens = grupo['fonte'].value_counts()
                proporcoes = contagens / total_obs
                s_fontes = len(contagens)

            # 1. CR1 e CR3
                cr1 = proporcoes.iloc[0] * 100 if s_fontes >= 1 else 0
                cr3 = proporcoes.iloc[:3].sum() * 100

            # 2. Gini-Simpson
                gini_simpson = (1 - (proporcoes ** 2).sum()) * 100

            # 3. Equitabilidade de Pielou (Shannon Normalizado)
                if s_fontes > 1:
                    shannon = -np.sum(proporcoes * np.log(proporcoes))
                    pielou = shannon / np.log(s_fontes)
                else:
                    pielou = 0.0

                return pd.Series({
                    'Total Entrevistas': total_obs,
                    'Qtd Fontes Únicas': s_fontes,
                    'CR1 (%)': round(cr1, 1),
                    'CR3 (% Top 3)': round(cr3, 1),
                    'Diversidade Gini-Simpson (%)': round(gini_simpson, 1),
                    'Equidade de Pielou': round(pielou, 2)
                })

            # Aplica o cálculo por secretaria (apenas secretarias com ao menos 3 registros)
            df_metricas = df_conc.groupby('sec').apply(calcular_indices_concentracao).reset_index()
            df_metricas = df_metricas[df_metricas['Total Entrevistas'] >= 3].sort_values(by='CR3 (% Top 3)', ascending=False)

            # Seleção da métrica para visualização no gráfico
            metrica_escolhida = st.selectbox(
                "Selecione a métrica para visualizar o ranking das Secretarias:",
                options=['CR3 (% Top 3)', 'Diversidade Gini-Simpson (%)', 'Equidade de Pielou', 'CR1 (%)']
            )

            fig_conc = px.bar(
                df_metricas,
                x=metrica_escolhida,
                y='sec',
                orientation='h',
                title=f"Ranking por {metrica_escolhida}",
                color=metrica_escolhida,
                color_continuous_scale="Reds" if "CR" in metrica_escolhida else "Greens",
                hover_data=['Total Entrevistas', 'Qtd Fontes Únicas', 'CR3 (% Top 3)', 'Diversidade Gini-Simpson (%)']
            )
            fig_conc.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_conc, width="stretch")

            # Exibição da Tabela Completa
            st.markdown("### 📋 Tabela Comparativa Detalhada")
            st.dataframe(df_metricas.style.format({
                'CR1 (%)': '{:.1f}%',
                'CR3 (% Top 3)': '{:.1f}%',
                'Diversidade Gini-Simpson (%)': '{:.1f}%',
                'Equidade de Pielou': '{:.2f}'
            }), width="stretch")

        else:
            st.warning("Dados insuficientes em 'sec' e 'fonte' para calcular os índices.") 
 
     # -------------------------------------------------------------
    
# Análise de Perfil da Entrevista (Variável 'tipo_ent')

    with st.expander("🗣️ Perfil da Entrevista: Propósito e Porta-Vozes", expanded=False):

        if 'tipo_ent' in df.columns:
            # Dicionário de mapeamento dos valores numéricos para texto
            mapa_tipo = {
                0: 'Promoção do Governo',
                1: 'Divulgação de Serviço',
                2: 'Voz de Especialista',
                3: 'Resposta à Crise'
            }

            # Cria um dataframe de cópia e aplica o mapeamento
            df_tipo = df.dropna(subset=['tipo_ent']).copy()
            df_tipo['tipo_ent_nome'] = df_tipo['tipo_ent'].map(mapa_tipo)

            if not df_tipo.empty:
                # --- 1. Gráfico Geral de Distribuição ---
                st.markdown("### 📊 Distribuição Geral por Tipo de Entrevista")
            
                tipo_counts = df_tipo['tipo_ent_nome'].value_counts().reset_index()
                tipo_counts.columns = ['Tipo de Entrevista', 'Quantidade']

                fig_tipo = px.pie(
                    tipo_counts, 
                    names='Tipo de Entrevista', 
                    values='Quantidade',
                    title="Proporção dos Propósitos das Entrevistas",
                    hole=0.4, # Transforma em gráfico de rosca (donut)
                    color='Tipo de Entrevista',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_tipo.update_traces(textposition='inside', textinfo='percent+label')
            
                st.plotly_chart(fig_tipo, width="stretch")

                st.markdown("---")

                # --- 2. Sub-módulo: Ranking de Fontes e Secretarias por Tipo ---
                st.markdown("### 🏆 Ranking: Top Fontes e Secretarias por Tipo")
                st.write("Selecione o tipo de entrevista nas abas abaixo para ver os destaques:")

                # Cria abas para cada tipo de entrevista
                nomes_tipos = list(mapa_tipo.values())
                abas = st.tabs(nomes_tipos)

                for i, (chave_tipo, nome_tipo) in enumerate(mapa_tipo.items()):
                    with abas[i]:
                        # Filtra os dados apenas para o tipo da aba atual
                        df_filtrado = df_tipo[df_tipo['tipo_ent'] == chave_tipo]

                        if not df_filtrado.empty:
                            col_fonte, col_sec = st.columns(2)

                            with col_fonte:
                                # Ranking Top 10 Fontes
                                top_fontes = df_filtrado['fonte'].value_counts().head(10).reset_index()
                                top_fontes.columns = ['Porta-Voz (Fonte)', 'Volume']
                            
                                fig_fontes = px.bar(
                                    top_fontes, 
                                    x='Volume', 
                                    y='Porta-Voz (Fonte)',
                                    orientation='h',
                                    title=f"Top 10 Fontes em '{nome_tipo}'",
                                    color='Volume',
                                    color_continuous_scale="Teal"
                                )
                                # Ordena para a maior barra ficar no topo
                                fig_fontes.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                                st.plotly_chart(fig_fontes, width="stretch")

                            with col_sec:
                                # Ranking Top 3 Secretarias
                                top_secs = df_filtrado['sec'].value_counts().head(3).reset_index()
                                top_secs.columns = ['Secretaria / Órgão', 'Volume']
                            
                                fig_secs = px.bar(
                                    top_secs, 
                                    x='Volume', 
                                    y='Secretaria / Órgão',
                                    orientation='h',
                                    title=f"Top 3 Secretarias em '{nome_tipo}'",
                                    color='Volume',
                                    color_continuous_scale="Oranges"
                                )
                                fig_secs.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                                st.plotly_chart(fig_secs, width="stretch")
                        else:
                            st.info(f"Nenhum registro encontrado para a categoria '{nome_tipo}'.")
            else:
                st.warning("Não há dados válidos na coluna 'tipo_ent' para plotar os gráficos.")
        else:
            st.error("A coluna 'tipo_ent' não foi encontrada no DataFrame. Certifique-se de que a tabela carregada possui esta variável.")
    # -------------------------------------------------------------
# Capilaridade Territorial e Descentralização (veiculo × regiao)
    # -------------------------------------------------------------
    with st.expander("🗺️ Correlações Veículo × Região", expanded=False):

        if 'veiculo' in df.columns and 'regiao' in df.columns:
            df_cap = df.dropna(subset=['veiculo', 'regiao']).copy()
            df_cap['regiao'] = df_cap['regiao'].astype(str).str.strip()

            if not df_cap.empty:
                st.markdown("""
                **O que esta análise revela?**
                Mede o alcance da comunicação distinguindo **Veículos Estaduais** (abrangência 'Ceará') 
                de **Veículos Regionais/Locais** (específicos de macrorregiões de planejamento do estado).
                """)

                # --- 1. Métrica de Descentralização ---
                total_insercoes = len(df_cap)
                qtd_estadual = len(df_cap[df_cap['regiao'].str.lower() == 'ceará'])
                qtd_regional = total_insercoes - qtd_estadual

                pct_estadual = (qtd_estadual / total_insercoes) * 100 if total_insercoes > 0 else 0
                pct_regional = (qtd_regional / total_insercoes) * 100 if total_insercoes > 0 else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Total de Inserções", f"{total_insercoes:,}".replace(",", "."))
                c2.metric("Alcance Estadual ('Ceará')", f"{pct_estadual:.1f}%", help="Veículos de cobertura para todo o Estado")
                c3.metric("Alcance Regionalizado (Interior)", f"{pct_regional:.1f}%", help="Veículos com foco em macrorregiões específicas")

                st.markdown("---")

                # --- 2. Distribuição por Região e Top Veículos Locais ---
                col_reg, col_veic_locais = st.columns([1.2, 1])

                with col_reg:
                    st.markdown("### 📍 Inserções por Região de Abrangência")
                    reg_counts = df_cap['regiao'].value_counts().reset_index()
                    reg_counts.columns = ['Região', 'Quantidade']

                    fig_reg = px.bar(
                        reg_counts,
                        x='Quantidade',
                        y='Região',
                        orientation='h',
                        title="Volume por Macrorregião / Estado",
                        color='Região',
                        # Destaque diferenciado para o escopo 'Ceará'
                        color_discrete_map={'Ceará': '#1f77b4'}
                    )
                    fig_reg.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
                    st.plotly_chart(fig_reg, width="stretch")

                with col_veic_locais:
                    st.markdown("### 📻 Top Veículos Regionais (Excluindo 'Ceará')")
                    df_interior = df_cap[df_cap['regiao'].str.lower() != 'ceará']

                    if not df_interior.empty:
                        top_locais = df_interior.groupby(['veiculo', 'regiao']).size().reset_index(name='Volume')
                        top_locais = top_locais.sort_values(by='Volume', ascending=False).head(10)

                        fig_locais = px.bar(
                            top_locais,
                            x='Volume',
                            y='veiculo',
                            color='regiao',
                            orientation='h',
                            title="Top 10 Veículos Hiperlocais / Regionais",
                            labels={'veiculo': 'Veículo', 'regiao': 'Macrorregião'}
                        )
                        fig_locais.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                        st.plotly_chart(fig_locais, width="stretch")
                    else:
                        st.info("Não há dados de veículos cadastrados exclusivamente no interior/macrorregiões.")

                st.markdown("---")

                # --- 3. Matriz Cruzada (Veículos × Região) ---
                st.markdown("### 🧩 Matriz de Presença Territorial: Top Veículos × Região")

                top_n = st.slider("Selecione a quantidade de veículos na matriz:", 5, 25, 15, key="slider_capilaridade")
                top_veiculos = df_cap['veiculo'].value_counts().head(top_n).index

                df_pivot_cap = df_cap[df_cap['veiculo'].isin(top_veiculos)].pivot_table(
                    index='veiculo', 
                    columns='regiao', 
                    aggfunc='size', 
                    fill_value=0
                )

                fig_matriz_cap = px.imshow(
                    df_pivot_cap,
                    labels=dict(x="Região de Abrangência", y="Veículo", color="Inserções"),
                    x=df_pivot_cap.columns,
                    y=df_pivot_cap.index,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    title=f"Distribuição Territorial dos Top {top_n} Veículos"
                )
                fig_matriz_cap.update_xaxes(tickangle=-45)
                fig_matriz_cap.update_layout(height=450, margin=dict(b=100))

                st.plotly_chart(fig_matriz_cap, width="stretch")

                # --- Insight Automático ---
                if not df_interior.empty:
                    reg_mais_ativa = df_interior['regiao'].value_counts().idxmax()
                    reg_menos_ativa = df_interior['regiao'].value_counts().idxmin()
                    st.info(
                        f"💡 **Insight de Capilaridade:** Excluindo veículos de cobertura estadual ('Ceará'), a macrorregião "
                        f"com maior volume de mídia local mapeada é **{reg_mais_ativa}**. A região com menor presença de imprensa regional é **{reg_menos_ativa}**."
                    )
            else:
                st.warning("Não há registros válidos cruzando as variáveis 'veiculo' e 'regiao'.")
        else:
            st.error("As colunas 'veiculo' e/ou 'regiao' não foram encontradas no DataFrame.")
    #-------------------------------------------------------------
# Análise de N-Grams (Bigramas) - Veículo x Assunto (Top 20% Pareto)
    # -------------------------------------------------------------
    with st.expander("🔗 Correlação N-Grams: Veículo x Assunto (Top 20% Pareto)", expanded=False):

    # Remove nulos
        df_ngram = df.dropna(subset=['veiculo', 'assunto']).copy()

        if not df_ngram.empty:
        # Lista básica de stopwords para limpar os assuntos
            stopwords_pt = {
                "Ceará", "Fortaleza", "Região", "sobre", "ano", "Programação", "Programa",
                "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", 
                "para", "com", "por", "um", "uma", "uns", "umas", "e", "ou", "o", "a", 
                "os", "as", "que", "se", "sobre", "mais", "como", "mas", "ao", "aos"
                }

        # Função para limpar o texto e extrair bigramas (n-grams de tamanho 2)
            def get_bigrams(text):
            # Remove pontuação e converte para minúsculas
                text = re.sub(r'[^\w\s]', '', str(text).lower())
                words = [w for w in text.split() if w not in stopwords_pt and len(w) > 2]
            
            # Cria pares de palavras (bigramas)
                if len(words) < 2:
                    return []
                return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

            # Aplica a extração e explode a lista para que cada bigrama seja uma linha
            df_ngram['bigrams'] = df_ngram['assunto'].apply(get_bigrams)
            df_exploded = df_ngram.explode('bigrams').dropna(subset=['bigrams'])

            # Conta a frequência da combinação (Veículo + Bigrama)
            df_freq = df_exploded.groupby(['veiculo', 'bigrams']).size().reset_index(name='Frequencia')
        
            # Ordena da maior frequência para a menor
            df_freq = df_freq.sort_values(by='Frequencia', ascending=False)

            # ---------------------------------------------------------
            # Aplicação da Regra de Pareto (Top 20% das combinações)
            # ---------------------------------------------------------
            top_20_percent_idx = max(1, int(len(df_freq) * 0.20))
            df_pareto_ngrams = df_freq.head(top_20_percent_idx)

            if not df_pareto_ngrams.empty:
                # Visualização usando Treemap
                fig_treemap = px.treemap(
                    df_pareto_ngrams,
                    path=['veiculo', 'bigrams'],  # Hierarquia: Veículo -> N-gram
                    values='Frequencia',
                    title=f"Top 20% N-Grams (Veículo x Assunto) - {len(df_pareto_ngrams)} combinações chave",
                    color='Frequencia',
                    color_continuous_scale='Blues'
                )
            
                fig_treemap.update_traces(root_color="lightgrey")
                fig_treemap.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=600)

                st.plotly_chart(fig_treemap, width="stretch")

                st.info(
                    f"💡 **Filtro de Pareto Aplicado:** O gráfico exibe apenas as **{len(df_pareto_ngrams)} combinações mais frequentes** "
                    f"(correspondentes aos 20% do topo). Pelo Princípio de Pareto, esse pequeno grupo de veículos e assuntos "
                    "geralmente é responsável por quase todo o volume de mídia analisado."
                )
            else:
                st.warning("Não há combinações suficientes de N-Grams para gerar o gráfico.")
        else:
            st.warning("Dados insuficientes nas colunas 'veiculo' e 'assunto'.")

    """
    ## Matrizes e modelagens 
    """
    # -------------------------------------------------------------
# Matriz de Perfil do Veículo (veiculo × tipo_ent)
    # -------------------------------------------------------------
    with st.expander("📰 Matriz de Perfil do Veículo (Veículo × Tipo de Entrevista)", expanded=False):

        if 'tipo_ent' in df.columns and 'veiculo' in df.columns:
            # Mapeamento numérico para nomes legíveis
            mapa_tipo = {
                0: 'Promoção do Governo',
                1: 'Divulgação de Serviço',
                2: 'Voz de Especialista',
                3: 'Resposta à Crise'
            }

            # Tratamento dos dados
            df_perfil = df.dropna(subset=['veiculo', 'tipo_ent']).copy()
            df_perfil['tipo_ent_nome'] = df_perfil['tipo_ent'].map(mapa_tipo)

            if not df_perfil.empty:
                st.markdown("""
                Mapeia os tipos de entrevista da cobertura de cada veículo de imprensa. 
                Permite identificar quais veículos dão mais espaço para pautas de **Serviço e Promoção** 
                e quais são acionados prioritariamente em cenários de **Crise**.
                """)

                # Slider para limitar aos Top N veículos e manter a visualização limpa
                top_n = st.slider(
                    "Selecione o número de veículos exibidos:", 
                    min_value=5, max_value=30, value=15, 
                    key="slider_veiculos_perfil"
                )
            
                top_veiculos = df_perfil['veiculo'].value_counts().head(top_n).index
                df_filtrado = df_perfil[df_perfil['veiculo'].isin(top_veiculos)]

                # Agrupamento e cálculo de frequência
                df_cross = df_filtrado.groupby(['veiculo', 'tipo_ent_nome']).size().reset_index(name='Quantidade')

                # Alternância entre Volume Absoluto e Porcentagem Relativa
                modo_visualizacao = st.radio(
                    "Modo de Visualização:",
                    ["Perfil Proporcional (100%)", "Volume Absoluto (Quantidade)"],
                    horizontal=True
                )

                # Cores personalizadas padronizadas para cada tipo de pauta
                cores_tipo = {
                    'Promoção do Governo': '#1f77b4', # Azul
                    'Divulgação de Serviço': '#2ca02c', # Verde
                    'Voz de Especialista': '#9467bd', # Roxo
                    'Resposta à Crise': '#d62728'     # Vermelho
                }

                if modo_visualizacao == "Perfil Proporcional (100%)":
                    # Calcula a porcentagem relativa de cada tipo dentro do próprio veículo
                    df_cross['Total_Veiculo'] = df_cross.groupby('veiculo')['Quantidade'].transform('sum')
                    df_cross['Porcentagem'] = (df_cross['Quantidade'] / df_cross['Total_Veiculo']) * 100

                    fig_perfil = px.bar(
                        df_cross,
                        x='Porcentagem',
                        y='veiculo',
                        color='tipo_ent_nome',
                        orientation='h',
                        title=f"Perfil Relativo de Cobertura - Top {top_n} Veículos (%)",
                        labels={'veiculo': 'Veículo de Imprensa', 'Porcentagem': 'Proporção (%)', 'tipo_ent_nome': 'Tipo de Entrevista'},
                        color_discrete_map=cores_tipo
                    )
                    fig_perfil.update_layout(barmode='stack', xaxis=dict(range=[0, 100]))
                else:
                    fig_perfil = px.bar(
                        df_cross,
                        x='Quantidade',
                        y='veiculo',
                        color='tipo_ent_nome',
                        orientation='h',
                        title=f"Volume Absoluto de Entrevistas - Top {top_n} Veículos",
                        labels={'veiculo': 'Veículo de Imprensa', 'Quantidade': 'Quantidade de Entrevistas', 'tipo_ent_nome': 'Tipo de Entrevista'},
                        color_discrete_map=cores_tipo
                    )
                    fig_perfil.update_layout(barmode='stack')

                fig_perfil.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, margin=dict(b=50))
                st.plotly_chart(fig_perfil, width="stretch")

                # --- Alerta e Insights Automáticos ---
                # Recalcula percentual para localizar destaques de crise
                df_cross['Total_Veiculo'] = df_cross.groupby('veiculo')['Quantidade'].transform('sum')
                df_cross['Porcentagem'] = (df_cross['Quantidade'] / df_cross['Total_Veiculo']) * 100
            
                df_crise = df_cross[df_cross['tipo_ent_nome'] == 'Resposta à Crise']
                if not df_crise.empty:
                    top_crise_veiculo = df_crise.sort_values(by='Porcentagem', ascending=False).iloc[0]
                    st.warning(
                        f"⚠️ **Destaque de Crise:** O veículo **'{top_crise_veiculo['veiculo']}'** é o que possui a maior "
                        f"concentração proporcional de pautas de **Resposta à Crise** ({top_crise_veiculo['Porcentagem']:.1f}% de sua cobertura total)."
                    )
            else:
                st.warning("Não há registros válidos cruzando as variáveis 'veiculo' e 'tipo_ent'.")
        else:
            st.error("As colunas 'veiculo' e/ou 'tipo_ent' não foram encontradas no DataFrame.")
    # -------------------------------------------------------------
# Modelagem de Tópicos (LDA) por Secretaria
    # -------------------------------------------------------------
    with st.expander("🧠 Modelagem de Tópicos (Algoritmo LDA) por Secretaria", expanded=False):

        st.markdown("""
        **Como funciona a Modelagem LDA por Secretaria?**
        O algoritmo analisa a co-ocorrência de palavras nos registros de assuntos e identifica os **Tópicos Latentes** (temas centrais). 
        Utilize o seletor para isolar as pautas de uma secretaria específica ou analisar o panorama geral.
        """)

        # Base de dados limpa
        df_lda_base = df.dropna(subset=['assunto']).copy()

        if not df_lda_base.empty:
            col_filtro, col_slider = st.columns([1, 1])

            with col_filtro:
                # Lista de secretarias com opção de selecionar 'Todas'
                secretarias_disponiveis = ["Todas as Secretarias"] + sorted(
                    [str(s) for s in df_lda_base['sec'].dropna().unique()]
                )
                sec_selecionada = st.selectbox(
                    "Filtrar por Secretaria/Órgão ('sec'):",
                    options=secretarias_disponiveis
                )

            with col_slider:
                num_topicos = st.slider(
                    "Quantidade de tópicos a identificar:", 
                    min_value=2, max_value=10, value=4, step=1
                )

            # Filtra o DataFrame conforme a seleção
            if sec_selecionada != "Todas as Secretarias":
                df_lda = df_lda_base[df_lda_base['sec'].astype(str) == sec_selecionada].copy()
            else:
                df_lda = df_lda_base.copy()

            st.markdown("---")

            # Verifica se há volume suficiente para rodar a modelagem
            if len(df_lda) >= 3:
                stopwords_pt = [
                    "Ceará", "Fortaleza", "ceará", "fortaleza", "Região", "sobre", "ano", 
                    "Programação", "Programa",
                    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", 
                    "para", "com", "por", "um", "uma", "uns", "umas", "e", "ou", "o", "a", 
                    "os", "as", "que", "se", "sobre", "mais", "como", "mas", "ao", "aos",
                    "sua", "seu", "pelo", "pela", "tem", "dos", "das", "foi", "são", "ser"
                ]

            # Ajusta min_df caso o filtro da secretaria tenha poucos registros
                min_df_val = 2 if len(df_lda) >= 10 else 1
                vectorizer = CountVectorizer(stop_words=stopwords_pt, min_df=min_df_val, max_df=0.95)
            
                try:
                    matriz_documento_termo = vectorizer.fit_transform(df_lda['assunto'].astype(str))
                
                    # Ajusta dinamicamente o número de tópicos se o vocabulário for menor que o desejado
                    num_features = matriz_documento_termo.shape[1]
                    num_topicos_efetivo = min(num_topicos, num_features)

                    if num_topicos_efetivo < num_topicos and num_topicos_efetivo > 0:
                        st.info(
                            f"ℹ️ A secretaria **'{sec_selecionada}'** possui um vocabulário menor. "
                            f"Exibindo **{num_topicos_efetivo} tópico(s)**."
                        )

                    # Aplicação do LDA
                    lda_model = LatentDirichletAllocation(n_components=num_topicos_efetivo, random_state=42)
                    lda_model.fit(matriz_documento_termo)
                
                    palavras = vectorizer.get_feature_names_out()
                    n_top_words = 10 
                
                    # Renderização dos gráficos em 2 colunas
                    cols_por_linha = 2
                
                    for i in range(0, num_topicos_efetivo, cols_por_linha):
                        cols = st.columns(cols_por_linha)
                    
                        for j in range(cols_por_linha):
                            topico_idx = i + j
                        
                            if topico_idx < num_topicos_efetivo:
                                with cols[j]:
                                    topico = lda_model.components_[topico_idx]
                                    top_palavras_idx = topico.argsort()[:-n_top_words - 1:-1]

                                    df_topico = pd.DataFrame({
                                        'Palavra': [palavras[idx] for idx in top_palavras_idx],
                                        'Importância': [topico[idx] for idx in top_palavras_idx]
                                    }).iloc[::-1]
                                
                                    fig_lda = px.bar(
                                        df_topico,
                                        x='Importância',
                                        y='Palavra',
                                        orientation='h',
                                        title=f"Tópico {topico_idx + 1} ({sec_selecionada})",
                                        color='Importância',
                                        color_continuous_scale="Teal"
                                    )
                                
                                    fig_lda.update_layout(
                                        xaxis_title=None,
                                        yaxis_title=None,
                                        height=350,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        coloraxis_showscale=False
                                    )
                                
                                    st.plotly_chart(fig_lda, width="stretch")
                                
                except Exception as e:
                    st.warning("Não há vocabulário ou variabilidade de texto suficiente na secretaria selecionada para estruturar o modelo LDA.")
            else:
                st.warning(f"A secretaria **'{sec_selecionada}'** possui apenas {len(df_lda)} registro(s) em 'assunto'. Selecione um órgão com mais dados para rodar a análise.")
        else:
            st.warning("Não há dados na coluna 'assunto' para executar o LDA.")
    # -------------------------------------------------------------
# Análise de Sombreamento e Sinergia via N-Grams (Sec × Assunto)
    # -------------------------------------------------------------
    with st.expander("🤝 Matriz de Sombreamento Intersecretarial via N-Grams (sec x assunto)", expanded=False):

        st.markdown("""
        **Como interpretar este gráfico?**
        Utilizamos **Pares de Palavras (Bigramas)** extraídos dos assuntos para encontrar o núcleo dos temas.
        * 🔍 **Sombreamento (Competição/Sobreposição):** Olhe para as colunas. Se um tema (ex: "saude publica") tiver quadrados escuros em várias Secretarias, múltiplos órgãos estão pautando a mesma agenda.
        """)

        # Remove valores nulos
        df_sinergia_ngram = df.dropna(subset=['sec', 'assunto']).copy()

        if not df_sinergia_ngram.empty:
            # Stopwords para limpar os termos inúteis
            stopwords_pt = {
                "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", 
                "para", "com", "por", "um", "uma", "uns", "umas", "e", "ou", "o", "a", 
                "os", "as", "que", "se", "sobre", "mais", "como", "mas", "ao", "aos",
                "sua", "seu", "pelo", "pela", "sobre", "dos", "das"
            }

            # Extração de Bigramas (N-grams de tamanho 2)
            def get_bigrams(text):
                # Limpa pontuação e joga para minúsculas
                text = re.sub(r'[^\w\s]', '', str(text).lower())
                words = [w for w in text.split() if w not in stopwords_pt and len(w) > 2]
                if len(words) < 2:
                    return []
                return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

            # Aplica a extração e expande o dataframe
            df_sinergia_ngram['bigrams'] = df_sinergia_ngram['assunto'].apply(get_bigrams)
            df_exploded = df_sinergia_ngram.explode('bigrams').dropna(subset=['bigrams'])

            if not df_exploded.empty:
                # Seleciona os Top 20 Bigramas e Top 15 Secretarias globais
                top_bigrams = df_exploded['bigrams'].value_counts().head(20).index
                top_secs = df_exploded['sec'].value_counts().head(15).index

                # Conta a frequência da combinação e filtra os Tops
                df_grouped = df_exploded.groupby(['sec', 'bigrams']).size().reset_index(name='contagem')
                df_filtered = df_grouped[
                    (df_grouped['bigrams'].isin(top_bigrams)) & 
                    (df_grouped['sec'].isin(top_secs))
                ]

                # Cria a matriz (Pivot Table) cruzando sec e bigramas
                pivot_sinergia = df_filtered.pivot(
                    index='sec', 
                    columns='bigrams', 
                    values='contagem'
                ).fillna(0)

                if not pivot_sinergia.empty:
                    # Plota o Mapa de Calor
                    fig_sinergia = px.imshow(
                        pivot_sinergia,
                        labels=dict(x="Temas (Bigramas)", y="Secretaria (Top 15)", color="Volume"),
                        x=pivot_sinergia.columns,
                        y=pivot_sinergia.index,
                        color_continuous_scale="Purples",
                        aspect="auto",
                        title="Matriz de Pautas: Secretarias vs. Núcleos Temáticos"
                    )
                
                    # Inclina os rótulos do eixo X e ajusta margens
                    fig_sinergia.update_xaxes(tickangle=-45)
                    fig_sinergia.update_layout(height=600, margin=dict(b=120))

                    st.plotly_chart(fig_sinergia, width="stretch")
                
                    # Alerta dinâmico: Bigramas presentes em 3 ou mais secretarias
                    assuntos_sombreados = (pivot_sinergia > 0).sum(axis=0)
                    assuntos_alerta = assuntos_sombreados[assuntos_sombreados >= 2].index.tolist()
                
                    if assuntos_alerta:
                        st.warning(
                            f"⚠️ **Alerta de Sombreamento de Pautas:** Os temas núcleo **'{', '.join(assuntos_alerta)}'** "
                            f"estão presentes no discurso de 2 ou mais secretarias diferentes. Ideal verificar se há sinergia "
                            f"ou sobreposição não intencional de mensagens."
                        )
                else:
                    st.info("Não há cruzamentos suficientes entre as principais Secretarias e Temas.")
            else:
                st.warning("Não foi possível extrair pares de palavras da variável 'assunto'.")
        else:
            st.warning("Dados insuficientes nas variáveis 'sec' e 'assunto'.")

    """
    ## Tabela completa
    """
    #-------------------------------------------------------------
# DATAFRAME ORIGINAL #########################################
    # ------------------------------------------------------------
    #Exibição dos dados no App
    st.dataframe(df, hide_index=True)
