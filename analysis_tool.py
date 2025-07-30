import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from streamlit_option_menu import option_menu
import statsmodels.api as sm

# --------------------------------#
# 데이터 전처리 및 분석 함수 #
# --------------------------------#

def preprocess_product_name(name):
    """'REPORTED PRODUCT NAME'을 정제하는 함수"""
    if not isinstance(name, str): return ''
    name = re.sub(r'\[.*?\]', '', name)
    name = name.split('_')[0]
    name = re.sub(r'(\(?\s*\d+\.?\d*\s*(kg|g|l|ml)\s*\)?)', '', name, flags=re.I)
    name = re.sub(r'[^A-Za-z0-9가-힣]', '', name)
    return name.strip()

def get_cluster_name(cluster_labels, preprocessed_names):
    """각 클러스터의 이름을 생성하는 함수"""
    cluster_name_map = {}
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label != -1:
            names_in_cluster = preprocessed_names[cluster_labels == label]
            if len(names_in_cluster) > 0:
                most_common_name = Counter(names_in_cluster).most_common(1)[0][0]
                cluster_name_map[label] = most_common_name
            else:
                cluster_name_map[label] = f'Cluster {label}'
    final_cluster_names = {}
    name_counts = Counter(cluster_name_map.values())
    used_names = {}
    for label, name in cluster_name_map.items():
        if name_counts[name] > 1:
            if name not in used_names: used_names[name] = 1
            final_cluster_names[label] = f"{name}_{used_names[name]}"
            used_names[name] += 1
        else:
            final_cluster_names[label] = name
    final_cluster_names[-1] = 'Noise'
    return final_cluster_names

def remove_outliers_iqr(df, column_name):
    """IQR 방식을 사용하여 이상치를 제거하는 함수"""
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_rows = len(df)
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    removed_rows = initial_rows - len(df_filtered)
    if removed_rows > 0:
        st.warning(f"분석의 정확도를 위해 시장 데이터의 단가(Unit Price) 이상치 {removed_rows}건을 제거했습니다.")
    return df_filtered

def reset_analysis_states():
    """모든 분석 상태를 초기화하는 함수"""
    st.session_state.analysis_done = False
    st.session_state.market_analysis_done = False
    keys_to_reset = ['customer_name', 'plot_df', 'customer_df', 'contract_date', 
                     'tfidf_matrix', 'savings_df', 'total_savings', 'market_df',
                     'analyzed_product_name']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def reset_market_analysis_states():
    """목표 2 분석 상태만 초기화하는 함수"""
    st.session_state.market_analysis_done = False
    keys_to_reset = ['market_df', 'analyzed_product_name', 'selected_customer', 
                     'market_contract_date', 'top_competitors_list',
                     'all_competitors_ranked']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# --------------------------#
# 메인 애플리케이션 UI 및 로직 #
# --------------------------#

st.set_page_config(layout="wide")

# --- 세션 상태 초기화 ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.customer_name = None
    st.session_state.plot_df = None
    st.session_state.customer_df = None
    st.session_state.contract_date = None
    st.session_state.tfidf_matrix = None
    st.session_state.savings_df = None
    st.session_state.total_savings = None

if 'market_analysis_done' not in st.session_state:
    st.session_state.market_analysis_done = False
    st.session_state.market_df = None
    st.session_state.analyzed_product_name = None
    st.session_state.selected_customer = None
    st.session_state.market_contract_date = None
    st.session_state.top_competitors_list = []
    st.session_state.all_competitors_ranked = None

# --- 사이드바 메뉴 ---
with st.sidebar:
    selected = option_menu(
        menu_title="메뉴",
        options=["고객사 효율 분석", "시장 경쟁력 분석"],
        icons=["person-bounding-box", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-size: 0.8rem;">
            © Made by Seungha Lee
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 페이지 1: 고객사 효율 분석
# ==============================================================================
if selected == "고객사 효율 분석":
    st.title('💲 고객사 효율 분석 (Overview)')
    
    if st.session_state.analysis_done:
        st.button("새로운 분석 시작 (다시하기)", on_click=reset_analysis_states)
    
    if not st.session_state.analysis_done:
        with st.form(key='analysis_form'):
            st.header("⚙️ 분석 설정")
            uploaded_file = st.file_uploader("고객사 데이터 파일을 업로드하세요", type=['csv', 'xlsx'])
            contract_date_input = st.date_input("계약 시작일 (Contract Date)을 선택하세요")
            submitted = st.form_submit_button("분석 실행")

        if submitted and uploaded_file is not None:
            with st.spinner('고객사 데이터를 분석 중입니다...'):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                required_columns = ['Date', 'Raw Importer Name', 'Reported Product Name', 'Volume', 'Unit Price']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"업로드한 파일에 필수 컬럼이 없습니다. ({', '.join(required_columns)} 컬럼이 필요합니다.)")
                    st.stop()
                
                df.rename(columns={'Date': 'date', 'Raw Importer Name': 'importer_name', 'Reported Product Name': 'product_name', 'Volume': 'volume', 'Unit Price': 'unit_price'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df['year_month'] = df['date'].dt.to_period('M')
                df['year'] = df['date'].dt.year
                df = df.dropna(subset=['importer_name', 'product_name', 'volume', 'unit_price'])
                
                customer_name = df['importer_name'].mode()[0]
                customer_df = df[df['importer_name'] == customer_name].copy()
                customer_df['product_preprocessed'] = customer_df['product_name'].apply(preprocess_product_name)
                vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                tfidf_matrix = vectorizer.fit_transform(customer_df['product_preprocessed'])
                dbscan = DBSCAN(eps=0.9, min_samples=3, metric='cosine')
                cluster_labels = dbscan.fit_predict(tfidf_matrix)
                cluster_name_map = get_cluster_name(cluster_labels, customer_df['product_preprocessed'])
                customer_df['cluster'] = cluster_labels
                customer_df['cluster_name'] = customer_df['cluster'].map(cluster_name_map)
                plot_df = customer_df[customer_df['cluster'] != -1].copy()

                contract_date = pd.to_datetime(contract_date_input)
                before_contract_df = plot_df[plot_df['date'] < contract_date]
                after_contract_df = plot_df[plot_df['date'] >= contract_date]
                avg_price_before = before_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_before')
                avg_price_after = after_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_after')
                volume_after = after_contract_df.groupby('cluster_name')['volume'].sum().rename('volume_after')
                savings_df = pd.concat([avg_price_before, avg_price_after, volume_after], axis=1).dropna()
                savings_df['savings'] = (savings_df['avg_price_before'] - savings_df['avg_price_after']) * savings_df['volume_after']
                savings_df = savings_df.sort_values('savings', ascending=False)
                total_savings = savings_df['savings'].sum()

                st.session_state.customer_name = customer_name
                st.session_state.plot_df = plot_df
                st.session_state.customer_df = customer_df
                st.session_state.contract_date = contract_date
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.savings_df = savings_df
                st.session_state.total_savings = total_savings
                st.session_state.analysis_done = True
                
            st.success(f"'{customer_name}' 고객사 분석 완료!")
            st.rerun()

    if st.session_state.analysis_done:
        with st.expander("1. 계약 전후 예상 절감액 분석", expanded=True):
            st.subheader("총 예상 절감액")
            st.markdown(f"## <span style='color:blue;'>${st.session_state.total_savings:,.2f}</span>", unsafe_allow_html=True)
            st.caption(f"※ 계약일({st.session_state.contract_date.date()}) 이후, 고객사의 자체 구매 단가 변화에 따른 총 예상 절감액입니다.")
            st.subheader("품목군별 상세 절감 내역")
            cols = st.columns(4)
            for i, row in enumerate(st.session_state.savings_df.itertuples()):
                col = cols[i % 4]
                color, arrow, val = ("blue", "▼", row.savings) if row.savings >= 0 else ("red", "▲", -row.savings)
                col.markdown(f"""<div style="border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem; text-align: center; height: 120px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;"><strong>{row.Index}</strong><p style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-top: 8px; margin-bottom: 0;">{arrow} ${val:,.0f}</p></div>""", unsafe_allow_html=True)
            
            st.dataframe(st.session_state.savings_df.style.format({
                'avg_price_before': '${:,.2f}',
                'avg_price_after': '${:,.2f}',
                'volume_after': '{:,.0f} KG',
                'savings': '${:,.2f}'
            }))

        with st.expander("2. 수입 품목군 정제 및 군집화 (DBSCAN & PCA)"):
            if st.session_state.tfidf_matrix is not None and st.session_state.tfidf_matrix.shape[0] > 0:
                pca = PCA(n_components=2, random_state=42)
                components = pca.fit_transform(st.session_state.tfidf_matrix.toarray())
                vis_df = pd.DataFrame(components, columns=['x', 'y'])
                vis_df['cluster_name'] = st.session_state.customer_df['cluster_name'].values
                vis_df['product_name'] = st.session_state.customer_df['product_name'].values
                cluster_volume_sorted = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False)
                top_clusters_for_viz = cluster_volume_sorted.head(15).index.tolist()
                vis_df_filtered = vis_df[vis_df['cluster_name'].isin(top_clusters_for_viz)]
                st.info(f"클러스터가 너무 많아, 수입량 기준 상위 {len(top_clusters_for_viz)}개 품목군만 그리드에 시각화합니다.")
                fig1 = px.scatter(vis_df_filtered[vis_df_filtered['cluster_name'] != 'Noise'], x='x', y='y', color='cluster_name', facet_col='cluster_name', facet_col_wrap=5, height=800, 
                                  title=f"<b>[{st.session_state.customer_name}] 품목 유사도 기반 군집화 (상위 품목군 Grid)</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 {len(top_clusters_for_viz)}개 품목군</span>", 
                                  labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}, hover_data=['product_name'])
                fig1.update_traces(marker=dict(size=8, opacity=0.8))
                st.plotly_chart(fig1, use_container_width=True)
                st.subheader("클러스터 리스트 (수입 중량순)")
                plot_df_sorted = st.session_state.plot_df.copy()
                plot_df_sorted['cluster_name'] = pd.Categorical(plot_df_sorted['cluster_name'], categories=cluster_volume_sorted.index.tolist(), ordered=True)
                st.dataframe(plot_df_sorted[['product_name', 'product_preprocessed', 'cluster_name']].drop_duplicates().sort_values('cluster_name'))

        with st.expander("3. 주요 수입 품목군 분석 (월별 수입량)"):
            plot_df_chart = st.session_state.plot_df.copy()
            plot_df_chart['year_month_str'] = plot_df_chart['year_month'].astype(str)
            cluster_volume = plot_df_chart.groupby(['year_month_str', 'cluster_name'])['volume'].sum().reset_index()
            sorted_clusters = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False).index.tolist()
            fig2 = px.bar(cluster_volume, x='year_month_str', y='volume', color='cluster_name', 
                          title=f"<b>[{st.session_state.customer_name}] 주요 수입 품목군 월별 수입량(KG)</b>", 
                          labels={'year_month_str': '연-월', 'volume': '수입량(KG)', 'cluster_name': '품목 클러스터'}, 
                          category_orders={'cluster_name': sorted_clusters})
            st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# 페이지 2: 시장 경쟁력 분석
# ==============================================================================
if selected == "시장 경쟁력 분석":
    st.title('🏆 시장 경쟁력 상세 분석 (Drill-down)')
    
    if st.session_state.get('market_analysis_done', False):
        st.button("새로운 시장 분석 시작 (다시하기)", on_click=reset_market_analysis_states)

    if not st.session_state.get('market_analysis_done', False):
        st.write("특정 품목에 대한 전체 시장 데이터를 업로드하여, 고객사의 시장 내 경쟁력을 심층 분석합니다.")
        market_file = st.file_uploader(f"분석할 품목의 전체 시장 데이터를 업로드하세요.", type=['csv', 'xlsx'], key="market_uploader")
        
        if market_file:
            with st.form("market_analysis_form"):
                try:
                    market_df_for_importers = pd.read_csv(market_file) if market_file.name.endswith('.csv') else pd.read_excel(market_file)
                    importer_list = sorted(market_df_for_importers['Raw Importer Name'].unique())
                    customer_name_selection = st.selectbox("분석할 고객사를 선택해주세요.", options=importer_list)
                except Exception as e:
                    st.error("파일을 읽는 중 오류가 발생했습니다. 컬럼명을 확인해주세요.")
                    customer_name_selection = None
                
                analyzed_product_name_input = st.text_input("분석할 품목명을 입력하세요 (예: 건면)")
                contract_date_input = st.date_input("분석 기준이 될 계약 시작일을 선택하세요.")
                market_submitted = st.form_submit_button("시장 경쟁력 분석 시작")

            if market_submitted and customer_name_selection and analyzed_product_name_input:
                with st.spinner('시장 데이터를 분석 중입니다. 파일 크기에 따라 시간이 걸릴 수 있습니다...'):
                    market_df = market_df_for_importers.copy()
                    market_df.rename(columns={'Date': 'date', 'Raw Importer Name': 'importer_name', 'Reported Product Name': 'product_name', 'Volume': 'volume', 'Unit Price': 'unit_price', 'Origin Country': 'origin_country'}, inplace=True)
                    market_df['date'] = pd.to_datetime(market_df['date'])
                    market_df['year_month'] = market_df['date'].dt.to_period('M')
                    market_df['year'] = market_df['date'].dt.year
                    market_df['quarter'] = market_df['date'].dt.quarter
                    
                    required_market_cols = ['importer_name', 'product_name', 'volume', 'unit_price']
                    if 'Exporter' in market_df.columns: required_market_cols.append('Exporter')
                    if 'origin_country' in market_df.columns: required_market_cols.append('origin_country')
                    market_df = market_df.dropna(subset=required_market_cols)
                    market_df = remove_outliers_iqr(market_df, 'unit_price')
                    
                    lowess_results = sm.nonparametric.lowess(market_df['unit_price'], market_df['volume'], frac=0.5)
                    market_df['expected_price'] = np.interp(market_df['volume'], lowess_results[:, 0], lowess_results[:, 1])
                    market_df['competitiveness_index'] = market_df['expected_price'] - market_df['unit_price']
                    
                    all_competitors_ranked = market_df.groupby('importer_name')['competitiveness_index'].mean().sort_values(ascending=False).reset_index()
                    
                    customer_rank_info = all_competitors_ranked[all_competitors_ranked['importer_name'] == customer_name_selection]
                    customer_rank = customer_rank_info.index[0] if not customer_rank_info.empty else len(all_competitors_ranked)
                    top_competitors_list = all_competitors_ranked.iloc[:customer_rank]['importer_name'].tolist()
                    if customer_name_selection in top_competitors_list:
                        top_competitors_list.remove(customer_name_selection)
                    
                    st.session_state.market_df = market_df
                    st.session_state.analyzed_product_name = analyzed_product_name_input
                    st.session_state.selected_customer = customer_name_selection
                    st.session_state.market_contract_date = pd.to_datetime(contract_date_input)
                    st.session_state.top_competitors_list = top_competitors_list
                    st.session_state.all_competitors_ranked = all_competitors_ranked
                    st.session_state.market_analysis_done = True
                st.rerun()

    if st.session_state.get('market_analysis_done', False):
        customer_name = st.session_state.selected_customer
        market_df = st.session_state.market_df
        analyzed_product_name = st.session_state.analyzed_product_name
        contract_date = st.session_state.market_contract_date
        top_competitors_list = st.session_state.top_competitors_list
        all_competitors_ranked = st.session_state.all_competitors_ranked
        
        st.subheader(f"'{analyzed_product_name}' 품목 시장 분석 결과 (기준 고객사: {customer_name})")

        with st.expander(f"1. [{analyzed_product_name}] 구매 경쟁력 분석", expanded=True):
            st.markdown("##### Volume 대비 Unit Price 분포 및 시장 추세")
            fig_comp = px.scatter(market_df, x='volume', y='unit_price', trendline="lowess", trendline_color_override="red", hover_data=['importer_name', 'date'], 
                                  title="<b>시장 내 거래 분포 및 평균 가격 추세선</b><br><span style='font-size: 0.8em; color:grey;'>LOWESS 회귀분석 기반</span>",
                                  labels={'volume': '수입량(KG)', 'unit_price': '단가(USD/KG)'})
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("##### 구매 경쟁력 상위 10개사")
            top_10_competitors = all_competitors_ranked.head(10)
            
            def highlight_customer(row):
                color = 'background-color: lightblue' if row.importer_name == customer_name else ''
                return [color] * len(row)
            
            st.dataframe(top_10_competitors.style.apply(highlight_customer, axis=1).format({'competitiveness_index': '{:,.2f}'}))
            
            customer_rank_info = all_competitors_ranked[all_competitors_ranked['importer_name'] == customer_name]
            if not customer_rank_info.empty:
                customer_rank = customer_rank_info.index[0] + 1
                if customer_rank > 10:
                    st.info(f"참고: **{customer_name}**의 구매 경쟁력 순위는 전체 {len(all_competitors_ranked)}개사 중 **{customer_rank}위**입니다.")

        with st.expander(f"2. [{analyzed_product_name}] 단가 추세 및 경쟁 우위 그룹 벤치마킹", expanded=True):
            # --- 구매 경쟁력 꺾은선 그래프: 구매 경쟁력 지수 추세 ---
            st.markdown("##### 구매 경쟁력 지수 월별 추이")
            st.caption("구매 경쟁력 = (시장 기대 가격) - (실제 구매 가격)")
            
            monthly_competitiveness = market_df.groupby(['year_month', 'importer_name'])['competitiveness_index'].mean().unstack()
            
            market_avg_monthly_comp = monthly_competitiveness.mean(axis=1)
            customer_monthly_comp = monthly_competitiveness.get(customer_name)
            
            fig_comp_trend = go.Figure()

            fig_comp_trend.add_trace(go.Scatter(x=market_avg_monthly_comp.index.to_timestamp(), y=market_avg_monthly_comp, mode='lines+markers', name='시장 전체 평균 지수', line=dict(color='blue', width=1)))

            if customer_monthly_comp is not None:
                fig_comp_trend.add_trace(go.Scatter(x=customer_monthly_comp.index.to_timestamp(), y=customer_monthly_comp, mode='lines+markers', name=f'{customer_name} 경쟁력 지수', line=dict(color='red')))

            if top_competitors_list:
                top_competitors_monthly_comp = monthly_competitiveness[top_competitors_list]
                top_competitors_avg_monthly_comp = top_competitors_monthly_comp.mean(axis=1)
                fig_comp_trend.add_trace(go.Scatter(x=top_competitors_avg_monthly_comp.index.to_timestamp(), y=top_competitors_avg_monthly_comp, mode='lines+markers', name='경쟁 우위 그룹 평균 지수', line=dict(color='green', dash='dash')))

            fig_comp_trend.update_layout(title=f'<b>[{analyzed_product_name}] 구매 경쟁력 지수 월별 추이</b>', xaxis_title='연-월', yaxis_title='구매 경쟁력 지수')
            st.plotly_chart(fig_comp_trend, use_container_width=True)
            st.caption("※ 이 그래프는 시장의 기대 단가 대비 실제 구매 단가의 차이(경쟁력 지수)가 시간에 따라 어떻게 변하는지를 보여줍니다.")
            st.markdown("---")


            
            # --- 단가 추세 variables define ---
            market_avg_price = market_df.groupby('year_month')['unit_price'].mean().rename('market_avg_price')
            customer_market_df = market_df[market_df['importer_name'] == customer_name]
            customer_avg_price = customer_market_df.groupby('year_month')['unit_price'].mean().rename('customer_avg_price')

            # --- 기존 그래프: 단가 추세 ---
            st.markdown("##### 월별 평균 단가 추세")
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=market_avg_price.index.to_timestamp(), y=market_avg_price, mode='lines+markers', name='시장 전체 평균 단가', line=dict(width=3)))
            fig4.add_trace(go.Scatter(x=customer_avg_price.index.to_timestamp(), y=customer_avg_price, mode='lines+markers', name=f'{customer_name} 평균 단가', line=dict(color='red')))
            
            if top_competitors_list:
                st.info(f"**벤치마크: 경쟁 우위 그룹 평균**")
                st.caption("※ '경쟁 우위 그룹'은 '구매 경쟁력 분석'의 순위에서 현재 선택된 고객사보다 높은 순위를 기록한 모든 기업들의 평균입니다.")
                top_competitors_df = market_df[market_df['importer_name'].isin(top_competitors_list)]
                top_competitors_avg_price = top_competitors_df.groupby('year_month')['unit_price'].mean().rename('top_competitors_avg_price')
                fig4.add_trace(go.Scatter(x=top_competitors_avg_price.index.to_timestamp(), y=top_competitors_avg_price, mode='lines+markers', name='경쟁 우위 그룹 평균', line=dict(color='green', dash='dash')))
            else:
                st.success(f"**벤치마크 분석:** `{customer_name}`님이 현재 시장에서 가장 우수한 구매 경쟁력을 보이고 있습니다!")

            fig4.update_layout(title=f'<b>[{analyzed_product_name}] 단가 추세</b>', xaxis_title='연-월', yaxis_title='평균 단가(USD/KG)')
            st.plotly_chart(fig4, use_container_width=True)

            # --- 평균 단가 수치 비교 ---
            st.markdown("##### 전체 기간 평균 단가 비교")
            col1, col2, col3 = st.columns(3)
            col1.metric("시장 전체 평균", f"${market_df['unit_price'].mean():.2f}")
            col2.metric(f"{customer_name} 평균", f"${customer_market_df['unit_price'].mean():.2f}")
            if top_competitors_list:
                col3.metric("경쟁 우위 그룹 평균", f"${top_competitors_df['unit_price'].mean():.2f}")

            if top_competitors_list:
                st.subheader("경쟁 우위 그룹 벤치마킹 시뮬레이션")
                with st.form("simulation_form"):
                    sim_start_date = st.date_input("시뮬레이션 시작일", contract_date)
                    sim_end_date = st.date_input("시뮬레이션 종료일")
                    run_simulation = st.form_submit_button("예상 절감액 계산")
                
                if run_simulation:
                    sim_df = pd.merge(customer_avg_price, top_competitors_avg_price, left_index=True, right_index=True, how='inner')
                    customer_volume_monthly = customer_market_df.groupby('year_month')['volume'].sum()
                    sim_df = pd.merge(sim_df, customer_volume_monthly, left_index=True, right_index=True, how='inner')
                    
                    sim_period_start = pd.to_datetime(sim_start_date).to_period('M')
                    sim_period_end = pd.to_datetime(sim_end_date).to_period('M')
                    sim_df = sim_df[(sim_df.index >= sim_period_start) & (sim_df.index <= sim_period_end)]
                    
                    if not sim_df.empty:
                        sim_df['potential_savings'] = (sim_df['customer_avg_price'] - sim_df['top_competitors_avg_price']) * sim_df['volume']
                        total_potential_savings = sim_df[sim_df['potential_savings'] > 0]['potential_savings'].sum()
                        st.success(f"해당 기간 동안 **경쟁 우위 그룹**의 평균 단가를 따랐다면 **${total_potential_savings:,.2f}**를 추가로 절감할 수 있었습니다.")
                        st.caption("※ 이 금액은 고객사의 월평균 단가가 경쟁 우위 그룹보다 높았던 달의 절감 가능액만을 합산한 값입니다.")
                    else:
                        st.warning("해당 기간에 비교할 데이터가 없습니다.")

        with st.expander(f"3. [{analyzed_product_name}] 시장 점유율 및 경쟁사 비교", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                years_with_data = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data:
                    selected_year_ms = st.selectbox("시장 점유율 분석 연도 선택", options=years_with_data, key=f"ms_year_{analyzed_product_name}")
                    ms_df = market_df[market_df['year'] == selected_year_ms]
                    ms_data = ms_df.groupby('importer_name')['volume'].sum().sort_values(ascending=False).reset_index()
                    display_data = ms_data.head(5)
                    if customer_name not in display_data['importer_name'].tolist() and not ms_data[ms_data['importer_name']==customer_name].empty:
                        customer_data = ms_data[ms_data['importer_name']==customer_name]
                        display_data = pd.concat([customer_data, display_data.head(4)])
                    others_volume = ms_data[~ms_data['importer_name'].isin(display_data['importer_name'])]['volume'].sum()
                    if others_volume > 0: display_data.loc[len(display_data)] = {'importer_name': '기타', 'volume': others_volume}
                    fig5 = px.pie(display_data, values='volume', names='importer_name', title=f"<b>[{analyzed_product_name}] {selected_year_ms}년 시장 점유율</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준</span>", hole=0.3)
                    fig5.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig5, use_container_width=True)
            with col2:
                years_with_data_price = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_price:
                    selected_year_price = st.selectbox("수입 상위 5개사 단가 비교 연도", options=years_with_data_price, key=f"price_year_{analyzed_product_name}")
                    price_comp_df = market_df[market_df['year'] == selected_year_price]
                    top_importers_by_vol = price_comp_df.groupby('importer_name')['volume'].sum().nlargest(5).index.tolist()
                    if customer_name not in top_importers_by_vol: top_importers_by_vol.append(customer_name)
                    price_comp_data = price_comp_df[price_comp_df['importer_name'].isin(top_importers_by_vol)]
                    avg_price_by_importer = price_comp_data.groupby('importer_name')['unit_price'].mean().sort_values().reset_index()
                    fig6 = px.bar(avg_price_by_importer, x='importer_name', y='unit_price', title=f"<b>{selected_year_price}년 고객사와 수입 상위 5개사 단가 비교</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 5개사</span>", labels={'importer_name': '수입사', 'unit_price': '평균 단가(USD/KG)'}, color='importer_name', color_discrete_map={customer_name: 'red'})
                    st.plotly_chart(fig6, use_container_width=True)
        
        if 'Exporter' in market_df.columns and 'origin_country' in market_df.columns:
            with st.expander(f"4. [{analyzed_product_name}] 공급망(공급사/원산지) 분석", expanded=True):
                years_with_data_exporter = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_exporter:
                    selected_year_exporter = st.selectbox("공급망 분석 연도 선택", options=years_with_data_exporter, key=f"exporter_year_{analyzed_product_name}")
                    exporter_analysis_df = market_df[market_df['year'] == selected_year_exporter]
                    
                    top_10_exporters_by_vol = exporter_analysis_df.groupby('Exporter')['volume'].sum().nlargest(10).index
                    exporter_analysis_df_top10 = exporter_analysis_df[exporter_analysis_df['Exporter'].isin(top_10_exporters_by_vol)]

                    st.subheader(f"{selected_year_exporter}년 분기별 공급사 단가 분포")
                    fig9 = px.box(exporter_analysis_df_top10, x='quarter', y='unit_price', color='Exporter', 
                                  title=f"<b>{selected_year_exporter}년 분기별 공급사 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 공급사</span>", 
                                  labels={'quarter': '분기', 'unit_price': '단가(USD/KG)'})
                    st.plotly_chart(fig9, use_container_width=True)
                    
                    customer_exporters_in_year = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['Exporter'].unique()
                    st.info(f"**{customer_name}**가 {selected_year_exporter}년에 거래한 공급사: **{', '.join(customer_exporters_in_year)}**")

                    for exporter in customer_exporters_in_year:
                        st.markdown(f"--- \n #### 공급사 '{exporter}' 비교 분석")
                        single_exporter_df = exporter_analysis_df[exporter_analysis_df['Exporter'] == exporter]
                        
                        st.subheader(f"Volume 및 평균 단가 비교")
                        importer_summary = single_exporter_df.groupby('importer_name').agg(
                            total_volume=('volume', 'sum'),
                            avg_unit_price=('unit_price', 'mean')
                        ).sort_values('total_volume', ascending=False).reset_index()

                        fig8 = go.Figure()
                        fig8.add_trace(go.Bar(
                            x=importer_summary['importer_name'],
                            y=importer_summary['total_volume'],
                            name='총 수입량(KG)',
                            marker_color=['red' if imp == customer_name else 'lightblue' for imp in importer_summary['importer_name']]
                        ))
                        fig8.add_trace(go.Scatter(
                            x=importer_summary['importer_name'],
                            y=importer_summary['avg_unit_price'],
                            name='평균 수입단가(USD/KG)',
                            yaxis='y2',
                            mode='lines+markers',
                            line=dict(color='orange')
                        ))
                        fig8.update_layout(
                            title=f"<b>'{exporter}' 거래 업체별 Volume 및 평균 단가</b>",
                            xaxis_title='수입사',
                            yaxis=dict(title='총 수입량(KG)'),
                            yaxis2=dict(title='평균 수입단가(USD/KG)', overlaying='y', side='right'),
                            legend=dict(x=0, y=1.1, orientation='h')
                        )
                        st.plotly_chart(fig8, use_container_width=True)

                        st.subheader(f"단가 분포 비교")
                        top_10_importers_by_vol = single_exporter_df.groupby('importer_name')['volume'].sum().nlargest(10).index
                        single_exporter_df_top10 = single_exporter_df[single_exporter_df['importer_name'].isin(top_10_importers_by_vol)]
                        fig10 = px.box(single_exporter_df_top10, x='importer_name', y='unit_price', 
                                       title=f"<b>'{exporter}' 거래 업체별 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 수입사</span>", 
                                       labels={'importer_name': '수입사', 'unit_price': '단가(USD/KG)'}, color='importer_name', color_discrete_map={customer_name: 'red'})
                        st.plotly_chart(fig10, use_container_width=True)

                    st.subheader(f"{selected_year_exporter}년 분기별 대안 소싱 옵션")
                    customer_origins = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['origin_country'].unique()
                    avg_prices = exporter_analysis_df.groupby(['quarter', 'Exporter', 'origin_country']).agg(avg_price=('unit_price', 'mean'), representative_product=('product_name', 'first')).reset_index()
                    
                    for q in range(1, 5):
                        st.markdown(f"--- \n #### {q}분기")
                        q_df = avg_prices[avg_prices['quarter'] == q]
                        if q_df.empty:
                            st.write("- 해당 분기에 거래 데이터가 없습니다.")
                            continue
                        
                        st.markdown("**현재 소싱 옵션**")
                        customer_exporters_q_df = q_df[q_df['Exporter'].isin(customer_exporters_in_year)].sort_values('avg_price')
                        if not customer_exporters_q_df.empty:
                            st.dataframe(customer_exporters_q_df[['Exporter', 'avg_price']].rename(columns={'Exporter': '공급사', 'avg_price': '평균 단가(USD/KG)'}).style.format({'평균 단가(USD/KG)': '${:,.2f}'}))
                        else:
                            st.write("- 공급사 거래 없음")
                        customer_origins_q_df = q_df[q_df['origin_country'].isin(customer_origins)].groupby('origin_country')['avg_price'].mean().reset_index().sort_values('avg_price')
                        if not customer_origins_q_df.empty:
                            st.dataframe(customer_origins_q_df.rename(columns={'origin_country': '원산지', 'avg_price': '평균 단가(USD/KG)'}).style.format({'평균 단가(USD/KG)': '${:,.2f}'}))
                        else:
                            st.write("- 원산지 거래 없음")

                        st.markdown("**대안 추천 옵션**")
                        customer_avg_price_q = q_df[q_df['Exporter'].isin(customer_exporters_in_year)]['avg_price'].mean()
                        if not pd.isna(customer_avg_price_q):
                            cheaper_exporters = q_df[(~q_df['Exporter'].isin(customer_exporters_in_year)) & (q_df['avg_price'] < customer_avg_price_q)].sort_values('avg_price')
                            if not cheaper_exporters.empty:
                                st.dataframe(cheaper_exporters[['Exporter', 'representative_product', 'avg_price']].rename(columns={'Exporter': '추천 공급사', 'representative_product': '대표 품목', 'avg_price': '평균 단가(USD/KG)'}).style.format({'평균 단가(USD/KG)': '${:,.2f}'}))
                            else:
                                st.write("- 더 저렴한 공급사 없음")
                        
                        customer_origin_avg_price_q = q_df[q_df['origin_country'].isin(customer_origins)].groupby('origin_country')['avg_price'].mean().mean()
                        if not pd.isna(customer_origin_avg_price_q):
                            cheaper_origins = q_df.groupby('origin_country')['avg_price'].mean().reset_index()
                            cheaper_origins = cheaper_origins[(~cheaper_origins['origin_country'].isin(customer_origins)) & (cheaper_origins['avg_price'] < customer_origin_avg_price_q)].sort_values('avg_price')
                            if not cheaper_origins.empty:
                                st.dataframe(cheaper_origins.rename(columns={'origin_country': '추천 원산지', 'avg_price': '평균 단가(USD/KG)'}).style.format({'평균 단가(USD/KG)': '${:,.2f}'}))
                            else:
                                st.write("- 더 저렴한 원산지 없음")
        else:
            st.warning("'Exporter' 또는 'Origin Country' 컬럼이 없어 공급망 분석을 수행할 수 없습니다.")
