import json
import functools
import streamlit as st
import pandas as pd
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from io import BytesIO
from uuid import uuid4
from typing import Dict, Any

# Importações de Banco de Dados (Mantidas do seu ambiente original)
from feedback_local import MONGO
from feedback_local import get_local_datetime_for_mongo

# Importações dos seus módulos
from app.projects.agente_verbatins.cliente.nodes import (
    create_classificator_agent_cliente,
    classifying_categories_node_cliente,
    sentiment_node_cliente,
)
from .cliente.agent_state import ClienteAgentState

# --- NOVAS IMPORTAÇÕES MODULARES (CORRETOR) ---
# Importações NPS
from app.projects.agente_verbatins.corretor.nps.nodes import classifying_categories_node_corretor as node_nps
from app.projects.agente_verbatins.corretor.nps.agent_state import CorretorNPSAgentState

# Importações Processos
from app.projects.agente_verbatins.corretor.processos.nodes import classifying_categories_node_processos as node_processos
from app.projects.agente_verbatins.corretor.processos.agent_state import CorretorProcessosAgentState

AGENT_NAME = "agente_verbatins"

# ==========================================================
# LAYOUT
# ==========================================================

def apply_custom_styles():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; max-width: 1200px; }
        .header-container {
            background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%);
            color: white; padding: 30px; border-radius: 14px; margin-bottom: 20px;
            text-align: center; box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
        }
        .section-title {
            font-size: 20px; font-weight: 700; color: #1E3A8A;
            margin-top: 20px; border-left: 5px solid #1E3A8A; padding-left: 10px;
        }
        div.stDownloadButton > button {
            background-color: #1E3A8A !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.5rem 2rem !important;
            border: none !important;
            font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)


# ==========================================================
# GESTÃO DE ESTADO E FEEDBACK
# ==========================================================

def initialize_run_state():
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = str(uuid4())

    feedback_control_key = f"feedback_saved_{st.session_state['run_id']}"
    if feedback_control_key not in st.session_state:
        st.session_state[feedback_control_key] = False

    if "response_data" not in st.session_state:
        st.session_state["response_data"] = None

def reset_run_state():
    new_id = str(uuid4())
    st.session_state["run_id"] = new_id
    st.session_state[f"feedback_saved_{new_id}"] = False
    st.session_state["response_data"] = None

def save_feedback(score: int, response_data: Dict[str, Any], agent_name: str = AGENT_NAME, justification: str = None):
    run_id = st.session_state.get("run_id")
    if st.session_state.get(f"feedback_saved_{run_id}", False):
        return

    try:
        collection = MONGO["agente_verbatins"]

        df_results = response_data.get("df")
        if isinstance(df_results, pd.DataFrame):
            analysis_results = df_results.fillna("").to_dict(orient="records")
        else:
            analysis_results = str(df_results)

        user_input_raw = response_data.get("payload_for_db", {}).get("ficha_data", {})

        if isinstance(user_input_raw, (pd.DataFrame, pd.Series)):
            user_input = user_input_raw.fillna("").to_dict(orient="records")
        elif isinstance(user_input_raw, dict):
            user_input = {str(k): v for k, v in user_input_raw.items()}
        else:
            user_input = str(user_input_raw)

        now = get_local_datetime_for_mongo()

        feedback_entry = {
            "timestamp": now,
            "response_id": str(uuid4()),
            "analysis_output": analysis_results,
            "user_input_summary": user_input,
            "feedback": score,
            "justification": justification 
        }

        collection.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "agent_name": agent_name, 
                    "datetime": now
                },
                "$push": {"interactions": feedback_entry}
            },
            upsert=True
        )

        st.session_state[f"feedback_saved_{run_id}"] = True
        st.toast("✅ Feedback salvo com sucesso!")

    except Exception as e:
        import traceback
        print(f"--- ERRO MONGODB ---")
        print(traceback.format_exc())
        st.error(f"Erro ao salvar feedback: {str(e)}")

# ==========================================================
# UTILITÁRIOS
# ==========================================================

def formatar_nota_corretor_nps(nota):
    try:
        n = int(nota)
        if n >= 9: return f"Promotor({n})"
        if 7 <= n <= 8: return f"Neutro({n})"
        return f"Detrator({n})"
    except: return f"N/A({nota})"

def formatar_nota_corretor_processos(nota):
    try:
        n = int(nota)
        if 1 <= n <= 5: label = "Insatisfeito"
        elif 6 <= n <= 7: label = "Neutro"
        else: label = "Satisfeito"
        return f"{label}({n})"
    except: return f"N/A({nota})"

def categoria_nota(nota):
    try:
        nota = int(nota)
    except (ValueError, TypeError):
        return "N/A"
    if nota is None:
        return "N/A"
    if nota <= 6:
        return "Detrator"
    elif nota in [7, 8]:
        return "Neutro"
    elif nota >= 9:
        return "Promotor"

def to_csv(df: pd.DataFrame):
    output = BytesIO()
    output.write("\ufeff".encode("utf-8"))
    df.to_csv(output, index=False, encoding="utf-8", sep=";")
    return output.getvalue()

def show_sentiment_analisys_cliente(analise):
    if isinstance(analise, str): return analise
    res = f"Sentimento Geral: {analise.get('sentimento_geral', 'N/A')}\n\n"
    for s in analise.get("sentencas", []):
        res += f"- {s.get('sentenca', '')} ({s.get('sentimento', '')})\n"
    return res

# ==========================================================
# GRAFO
# ==========================================================

def build_graph(tipo_nps, tipo_fluxo_corretor=None):
    if tipo_nps == "Cliente":
        AgentState = ClienteAgentState
        graph_builder = StateGraph(AgentState)
        classify_node = functools.partial(classifying_categories_node_cliente, agent=create_classificator_agent_cliente())
        graph_builder.add_node("classify", classify_node)
        graph_builder.add_node("sentiment", sentiment_node_cliente)
        graph_builder.add_edge(START, "classify")
        graph_builder.add_edge("classify", "sentiment")
        graph_builder.add_edge("sentiment", END)
    else:
        # Lógica para Corretor dividida por fluxo
        if tipo_fluxo_corretor == "Processos":
            AgentState = CorretorProcessosAgentState
            node_func = node_processos
        else:
            AgentState = CorretorNPSAgentState
            node_func = node_nps

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("classify", node_func)
        graph_builder.add_edge(START, "classify")
        graph_builder.add_edge("classify", END)
    return graph_builder.compile()

# ==========================================================
# FLUXO PRINCIPAL
# ==========================================================

def handle_analysis_flow(tipo_nps, input_data, is_table=False, macro_escolhido=None, tipo_fluxo_corretor=None, ramo_escolhido=None):
    graph = build_graph(tipo_nps, tipo_fluxo_corretor)
    outputs = []
    data_source = input_data if is_table else [input_data]

    with st.spinner("🔄 Processando..."):
        iterable = data_source.iterrows() if is_table else enumerate(data_source)

        for _, item in iterable:
            row_dict = {}
            row = item if is_table else item

            if tipo_nps == "Corretor":
                # --- AJUSTE: Local Response Date sem 'nan' ---
                lrd = row.get("Local Response Date")
                if pd.isna(lrd) or str(lrd).lower() == "nan" or str(lrd).strip() == "":
                    row_dict["Local Response Date"] = ""
                else:
                    row_dict["Local Response Date"] = str(lrd)

                row_dict["Survey ID"] = str(row.get("Survey ID", ""))

                # Para Processos em tabela, respeita o Ramo e Macro 1 definidos na UI
                ramo = ramo_escolhido if ramo_escolhido else row.get("Ramo", "Auto")

                comentario = str(row.get("Motivo da nota", row.get("Motivo da Nota", row.get("Comentário", "")))).replace("\n", " ")
                nota_raw = row.get("NPS", row.get("Nota", row.get("Satisfação", row.get("Nota do Processo", 0))))

                try: nota = int(nota_raw)
                except: nota = 0

                current_input = {
                    "comentario": comentario, "ramo": ramo, 
                    "nota": nota, "macro_tema": macro_escolhido
                }

                final_state = graph.invoke(current_input)

                # Regra de Formatação de Nota por Fluxo
                if tipo_fluxo_corretor == "Processos":
                    nota_formatada = formatar_nota_corretor_processos(nota)
                else:
                    nota_formatada = formatar_nota_corretor_nps(nota)

                row_dict.update({
                    "Ramo": ramo, 
                    "Canal Ajuda": row.get("Canal de Ajuda", row.get("Canal Ajuda", "")),
                    "Motivo da Nota": comentario, 
                    "Sentimento": final_state.get("sentimento", ""),
                    "Análise de Sentimento": final_state.get("analise_sentimento", "")
                })

                # --- AJUSTE SOLICITADO: NPS -> Satisfação em Processos ---
                if tipo_fluxo_corretor == "Processos":
                    row_dict["Satisfação"] = nota_formatada
                else:
                    row_dict["NPS"] = nota_formatada

                # --- NOVA LÓGICA DE CATEGORIAS ---
                try:
                    res_ia = json.loads(final_state["categorias"])
                    classif = res_ia.get("classificacoes", res_ia) 

                    if tipo_fluxo_corretor == "Processos":
                        # Tabela de saída para Processos
                        row_dict.update({
                            "Macro 1": macro_escolhido,
                            "Micro 1": classif.get("micro_1", ""),
                            "Micro 2": classif.get("micro_2", "")
                        })
                    else:
                        # Tabela de saída para NPS
                        row_dict.update({
                            "MACRO 1": classif.get("macro_1", ""),
                            "MICRO 1": classif.get("micro_1", ""),
                            "DOR 1": classif.get("dor_1", ""),
                            "MACRO 2": classif.get("macro_2", ""),
                            "MICRO 2": classif.get("micro_2", ""),
                            "DOR 2": classif.get("dor_2", ""),
                            "MACRO 3": classif.get("macro_3", ""),
                            "MICRO 3": classif.get("micro_3", ""),
                            "DOR 3": classif.get("dor_3", "")
                        })
                except Exception as e:
                    row_dict["Erro"] = f"Erro na extração: {str(e)}"

            else: # Lógica Cliente
                if is_table:
                    # Formatação de campos específicos da tabela de cliente
                    lrd = row.get("Local Response Date")
                    if pd.isna(lrd) or str(lrd).lower() == "nan" or str(lrd).strip() == "":
                        row_dict["Local Response Date"] = ""
                    elif isinstance(lrd, datetime):
                        row_dict["Local Response Date"] = lrd.strftime("%d/%m/%Y")
                    else:
                        row_dict["Local Response Date"] = str(lrd)

                    row_dict["Survey ID"] = str(row.get("Survey ID", ""))
                    ramo = row.get("Ramo", "Geral")

                    comentario_col = "Motivo da nota (Relacionamento Cliente) Comentário"
                    if comentario_col in row:
                        comentario = str(row[comentario_col]).replace("\n", " ")
                    else:
                        comentario = str(row.get("Comentário", row.get("Motivo da nota", ""))).replace("\n", " ")

                    nota_raw = row.get("Nota (Relacionamento Cliente)", row.get("Nota", 10))
                    try: nota = int(nota_raw)
                    except: nota = 0
                else:
                    ramo = row.get("Ramo", "Geral")
                    comentario = str(row.get("Comentário", row.get("Motivo da nota", "")))
                    nota = row.get("Nota", 10)

                final_state = graph.invoke({
                    "comentario": comentario, 
                    "nota": nota, 
                    "ramo": ramo,
                    "macro_tema": "NPS"
                })

                row_dict["Ramo"] = ramo
                row_dict["Motivo da nota"] = comentario

                if not is_table:
                    n = int(nota)
                    label = "Promotor" if n >= 9 else "Neutro" if n >= 7 else "Detrator"
                    row_dict["Nota"] = label
                else:
                    categoria = categoria_nota(nota)
                    row_dict["Nota"] = f"{categoria} ({nota})" if categoria != "N/A" else f"N/A ({nota})"

                # Extração de categorias (Macro, Micro, Dor) para Cliente
                try:
                    raw_cats = final_state.get("categorias", {})
                    data_json = json.loads(raw_cats) if isinstance(raw_cats, str) else raw_cats
                    cats = data_json.get("categorias", data_json) if isinstance(data_json, dict) else data_json

                    if isinstance(cats, list):
                        for i, c in enumerate(cats):
                            if i < 3: # Limita a 3 macros
                                row_dict[f"Macro {i+1}"] = c.get("macro", "")
                                row_dict[f"Micro {i+1}"] = c.get("micro", "")
                                row_dict[f"Dor {i+1}"] = c.get("dor", "")
                    elif isinstance(cats, dict):
                        row_dict["Macro 1"] = cats.get("macro", "")
                        row_dict["Micro 1"] = cats.get("micro", "")
                        row_dict["Dor 1"] = cats.get("dor", "")
                except Exception as e:
                    row_dict["Erro"] = f"Erro na extração: {str(e)}"

                # Análise de Sentimento
                sentimento_ret = final_state.get("sentimento", [{}])
                analise_sentimento_input = sentimento_ret[0] if isinstance(sentimento_ret, list) and len(sentimento_ret) > 0 else sentimento_ret
                row_dict["Análise de Sentimento"] = show_sentiment_analisys_cliente(analise_sentimento_input)

            outputs.append(row_dict)

    # Criação do DataFrame
    df_result = pd.DataFrame(outputs)

    # --- AJUSTE GLOBAL: Limpa qualquer 'nan' residual em toda a tabela ---
    df_result = df_result.fillna("").replace(["nan", "None", ""], "")

    # Reordenar colunas conforme especificação do usuário
    if tipo_nps == "Corretor":
        if tipo_fluxo_corretor == "Processos":
            # --- AJUSTE SOLICITADO: Coluna Satisfação na ordenação ---
            cols = ["Survey ID", "Local Response Date", "Ramo", "Canal Ajuda", "Satisfação", "Motivo da Nota", "Macro 1", "Micro 1", "Micro 2", "Sentimento", "Análise de Sentimento"]
        else:
            cols = [
                "Survey ID", "Local Response Date", "Ramo", "Canal Ajuda", "NPS", "Motivo da Nota",
                "MACRO 1", "MICRO 1", "DOR 1", 
                "MACRO 2", "MICRO 2", "DOR 2", 
                "MACRO 3", "MICRO 3", "DOR 3",
                "Sentimento", "Análise de Sentimento"
            ]
        # Filtra apenas colunas que existem no df para evitar erros
        df_result = df_result[[c for c in cols if c in df_result.columns]]
    else:
        # Reordenação de colunas para Cliente
        cols = [
            "Survey ID", "Local Response Date", "Ramo", "Nota", "Motivo da nota",
            "Macro 1", "Micro 1", "Dor 1", 
            "Macro 2", "Micro 2", "Dor 2", 
            "Macro 3", "Micro 3", "Dor 3",
            "Análise de Sentimento"
        ]
        cols_existentes = [c for c in cols if c in df_result.columns]
        cols_extras = [c for c in df_result.columns if c not in cols]
        df_result = df_result[cols_existentes + cols_extras]

    st.session_state["response_data"] = {
        "df": df_result, 
        "csv_name": f'analise_{tipo_nps.lower()}.csv',
        "payload_for_db": {"email_conteudo": str(outputs), "ficha_data": input_data}
    }

# ==========================================================
# INTERFACE
# ==========================================================

def build_page(is_authenticated=True):
    initialize_run_state()
    apply_custom_styles()

    st.markdown("""
        <div class="header-container">
            <h1 style="margin:0;">Agente de Análise de Verbatins</h1>
            <p style="opacity:0.9;">Análise Inteligente e Categorização de Notificações</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Como usar este sistema", expanded=False):
        st.markdown(f"""
        **Passos para uso:**
        1. 👤 **Selecione o Agente:** Escolha entre **Cliente** ou **Corretor** no menu superior.
        2. 📥 **Escolha a Entrada:** - **Única:** Digite um comentário manualmente e preencha os detalhes (Ramo/Tipo de Cliente ou Contexto).
            - **Tabela:** Faça o upload de um arquivo Excel (.xlsx) para processamento em lote.
        3. 🚀 **Inicie a Análise:** Clique em 'Analisar' ou 'Processar Planilha'.
        4. 📊 **Revise e Baixe:** Verifique as categorizações (Macro, Micro, Dor) e o sentimento, então exporte para CSV.
        """)

    tipo_nps = st.selectbox("Agente:", ["Cliente", "Corretor"], on_change=reset_run_state)
    st.write("---")

    tipo_fluxo_corretor = None
    if tipo_nps == "Corretor":
        tipo_fluxo_corretor = st.radio("Escolha o Fluxo:", ["NPS", "Processos"], horizontal=True, on_change=reset_run_state)

    selection = st.radio("Entrada", ["Única", "Tabela"], label_visibility="collapsed", on_change=reset_run_state)

    if selection == "Única":
        coment = st.text_area("Digite o Feedback/Comentário", height=150)
        st.markdown("---")

        if tipo_nps == "Cliente":
            st.markdown("### 👤 Entradas Específicas: Cliente")
            ramo = st.selectbox("Qual é o ramo?", ("Auto", "Capitalização", "Previdência", "Residencial", "Saúde", "Vida"), index=None, placeholder="Selecione o ramo")
            tipo_cliente = st.selectbox("Qual é o tipo de cliente?", ("Promotor", "Neutro", "Detrator"), index=None, placeholder="Choose an option")

            nota_map = {"Promotor": 10, "Neutro": 8, "Detrator": 1}
            nota_final = nota_map.get(tipo_cliente, 10)

            if st.button("Analisar") and coment:
                if ramo and tipo_cliente:
                    reset_run_state()
                    data_input = {"Comentário": coment, "Nota": nota_final, "Ramo": ramo}
                    handle_analysis_flow(tipo_nps, data_input, is_table=False)
                    st.rerun()
                else:
                    st.warning("Por favor, preencha o Ramo e o Tipo de Cliente.")

        elif tipo_nps == "Corretor":
            st.markdown(f"### 💼 Entradas Específicas: Corretor ({tipo_fluxo_corretor})")

            macro_tema = None
            if tipo_fluxo_corretor == "Processos":
                macro_tema = st.selectbox("Selecione o Macro Tema (Processo):", ["Assistência", "Cotação", "Emissão", "Gestão", "Movimentação", "Pagamento", "Portabilidade", "Pós-Venda", "Prospecção", "Renovação", "Resgate", "Sinistro"], index=None, placeholder="Choose an option")
                ramo_corretor = st.selectbox("Qual é o ramo?", ["Vida", "RE", "Auto", "Previdência", "Saúde"], index=None, placeholder="Choose an option")
                canal_ajuda = st.selectbox("Qual é o Canal de Ajuda?", ["Mercado", "Corporate", "Rede", "Outro"], index=None, placeholder="Choose an option")
                nps_corretor = st.selectbox("Qual é a Satisfação do Corretor?", ["Satisfeito", "Neutro", "Insatisfeito"], index=None, placeholder="Choose an option")

                nota_map = {"Satisfeito": 10, "Neutro": 8, "Insatisfeito": 1}
                nota_final = nota_map.get(nps_corretor, 10)

            else:
                macro_tema = "NPS"
                ramo_corretor = st.selectbox("Qual é o ramo?", ["Vida", "RE", "Auto", "Previdência", "Saúde"], index=None, placeholder="Choose an option")
                canal_ajuda = st.selectbox("Qual é o Canal de Ajuda?", ["Mercado", "Corporate", "Rede", "Outro"], index=None, placeholder="Choose an option")
                nps_corretor = st.selectbox("Qual é a Satisfação do Corretor?", ["Promotor", "Neutro", "Detrator"], index=None, placeholder="Choose an option")

                nota_map = {"Promotor": 10, "Neutro": 8, "Detrator": 1}
                nota_final = nota_map.get(nps_corretor, 10)

            if st.button("Analisar", key="btn_analisar_corretor"):
                valido = coment and ramo_corretor and canal_ajuda and nps_corretor
                if tipo_fluxo_corretor == "Processos" and not macro_tema:
                    valido = False

                if valido:
                    reset_run_state()
                    data_input = {
                        "Comentário": coment, 
                        "Nota": nota_final, 
                        "Ramo": ramo_corretor,
                        "Canal de Ajuda": canal_ajuda 
                    }
                    handle_analysis_flow(tipo_nps, data_input, is_table=False, macro_escolhido=macro_tema, tipo_fluxo_corretor=tipo_fluxo_corretor)
                    st.rerun()
                else:
                    st.warning("⚠️ Por favor, preencha todos os campos antes de analisar.")

    elif selection == "Tabela":
        macro_tema = None
        ramo_fixo = None
        if tipo_nps == "Corretor":
            st.markdown(f"### 💼 Configuração de Lote: Corretor ({tipo_fluxo_corretor})")
            if tipo_fluxo_corretor == "Processos":
                col1, col2 = st.columns(2)
                with col1:
                    ramo_fixo = st.selectbox("Selecione o Ramo:", ["Vida", "RE", "Auto", "Previdência", "Saúde"])
                with col2:
                    macro_tema = st.selectbox("Selecione o Macro 1:", ["Assistência", "Cotação", "Emissão", "Gestão", "Movimentação", "Pagamento", "Portabilidade", "Pós-Venda", "Prospecção", "Renovação", "Resgate", "Sinistro"])
            else:
                macro_tema = "NPS"
        else:
            macro_tema = "NPS"
            st.info("O Agente Cliente processa o Macro Tema padrão **NPS**.")

        file = st.file_uploader("Upload Excel", type=["xlsx"])
        if st.button("Processar Planilha") and file:
            reset_run_state()
            handle_analysis_flow(
                tipo_nps, 
                pd.read_excel(file), 
                is_table=True, 
                macro_escolhido=macro_tema,
                tipo_fluxo_corretor=tipo_fluxo_corretor,
                ramo_escolhido=ramo_fixo
            )
            st.rerun()

    if st.session_state.get("response_data"):
        data = st.session_state["response_data"]
        st.markdown("---")
        st.subheader("Resultados")
        st.dataframe(data["df"], use_container_width=True)
        st.download_button("Baixar CSV", to_csv(data["df"]), data["csv_name"])

        st.markdown("---")
        st.subheader("Avalie esta Análise")

        rid = st.session_state['run_id']
        feedback_key = f"feedback_widget_{rid}"

        if st.session_state.get(f"feedback_saved_{rid}"):
            st.success("✅ Obrigado pelo seu feedback! Ele foi registrado em nosso banco de dados.")
        else:
            score = st.feedback("thumbs", key=feedback_key)
            if score is not None:
                if score == 0:
                    st.warning("Sentimos muito que o resultou não foi o esperado.")
                    justification = st.text_area("Descreva o erro:", key=f"text_area_{rid}")
                    if st.button("Enviar Justificativa", key=f"btn_submit_{rid}"):
                        save_feedback(score=score, response_data=data, justification=justification)
                        st.rerun()
                else:
                    save_feedback(score=score, response_data=data)
                    st.rerun()

if __name__ == "__main__":
    build_page()