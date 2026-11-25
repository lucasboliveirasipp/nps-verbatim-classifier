import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# CONFIGURAÇÃO DE CAMINHOS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_classificacao_verbatins.joblib")
ARQUIVO_TREINO = os.path.join(BASE_DIR, "Base de Dados Corretor.xlsx")

def treinar_modelo_verbatins(caminho_planilha):
    try:
        if not os.path.exists(caminho_planilha):
            print(f"❌ Erro: Arquivo não encontrado em {caminho_planilha}")
            return None

        print(f"📂 Lendo base de treino: {caminho_planilha}...")
        df = pd.read_excel(caminho_planilha)

        # Mapeamento atualizado para a nova estrutura de 9 colunas de saída
        col_x = "Motivo da Nota"
        cols_y = [
            "MACRO 1", "MICRO 1", "DOR 1",
            "MACRO 2", "MICRO 2", "DOR 2",
            "MACRO 3", "MICRO 3", "DOR 3"
        ]

        # Verificação de colunas necessárias
        if not all(c in df.columns for c in [col_x] + cols_y):
            print(f"❌ Erro: Colunas necessárias não encontradas.")
            print(f"Colunas esperadas: {[col_x] + cols_y}")
            print(f"Colunas encontradas: {df.columns.tolist()}")
            return None

        # Limpeza: Preenche valores vazios com "N/A" para não perder linhas com apenas 1 classificação
        df[cols_y] = df[cols_y].fillna("Não Informado")
        df = df.dropna(subset=[col_x])

        # Garantir que os textos são strings e remover espaços extras
        X = df[col_x].astype(str).str.strip()

        # Dicionário para armazenar os Encoders de cada coluna
        encoders = {}
        y_encoded_data = {}

        for col in cols_y:
            le = LabelEncoder()
            # O fit_transform precisa lidar com todas as categorias possíveis daquela coluna
            y_encoded_data[col] = le.fit_transform(df[col].astype(str))
            encoders[f"le_{col.lower().replace(' ', '_')}"] = le

        # DataFrame de labels para o ClassifierChain (9 saídas)
        y = pd.DataFrame(y_encoded_data)

        # ==========================================================
        # CONSTRUÇÃO DO PIPELINE (Hierarquia Expandida)
        # ==========================================================
        # ClassifierChain agora percorre a ordem: Macro1 -> Micro1 -> Dor1 ... -> Dor3
        chain = ClassifierChain(
            LinearSVC(random_state=42, class_weight='balanced', max_iter=5000),
            order=list(range(len(cols_y))) 
        )

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                analyzer='word',
                max_features=5000, # Aumentado para cobrir a maior diversidade de labels
                sublinear_tf=True,
                strip_accents='unicode'
            )),
            ('clf_chain', chain)
        ])

        print(f"⏳ Treinando modelo Multi-Output (9 alvos: {', '.join(cols_y)})...")
        pipeline.fit(X, y)

        # Salvamento do modelo e de TODOS os encoders
        data_to_save = {
            "model": pipeline,
            "encoders": encoders, # Dicionário com os 9 encoders
            "features": {
                "input_column": col_x,
                "target_columns": cols_y
            }
        }

        joblib.dump(data_to_save, MODEL_PATH)
        print(f"✅ Sucesso! Modelo de Verbatins (Multicamadas) salvo em: {MODEL_PATH}")
        return data_to_save

    except Exception as e:
        import traceback
        print(f"❌ Erro crítico no processo de treino: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    treinar_modelo_verbatins(ARQUIVO_TREINO)