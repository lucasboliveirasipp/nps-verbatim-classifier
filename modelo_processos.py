import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_classificacao_verbatins_processos.joblib")
ARQUIVO_TREINO = os.path.join(BASE_DIR, "Base de Dados Processos Corretor.xlsx")

def treinar_modelo_verbatins(caminho_planilha):
    try:
        if not os.path.exists(caminho_planilha):
            print(f"❌ Erro: Arquivo não encontrado em {caminho_planilha}")
            return None

        print(f"📂 Lendo base de treino: {caminho_planilha}...")
        df = pd.read_excel(caminho_planilha)

        col_x = "Motivo da Nota"
        # Foco apenas nas colunas de saída que o ML deve aprender
        cols_y = ["Micro 1", "Micro 2"]

        if not all(c in df.columns for c in [col_x] + cols_y):
            print(f"❌ Erro: Colunas necessárias não encontradas no arquivo.")
            return None

        # Limpeza
        df[cols_y] = df[cols_y].fillna("Não Informado")
        df = df.dropna(subset=[col_x])
        X = df[col_x].astype(str).str.strip()

        encoders = {}
        y_encoded_data = {}

        for col in cols_y:
            le = LabelEncoder()
            y_encoded_data[col] = le.fit_transform(df[col].astype(str))
            encoders[f"le_{col.lower().replace(' ', '_')}"] = le

        y = pd.DataFrame(y_encoded_data)

        chain = ClassifierChain(
            LinearSVC(random_state=42, class_weight='balanced', max_iter=5000),
            order=list(range(len(cols_y))) 
        )

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                strip_accents='unicode'
            )),
            ('clf_chain', chain)
        ])

        print(f"⏳ Treinando modelo Processos (Alvos: {cols_y})...")
        pipeline.fit(X, y)

        data_to_save = {
            "model": pipeline,
            "encoders": encoders,
            "features": {"input_column": col_x, "target_columns": cols_y}
        }

        joblib.dump(data_to_save, MODEL_PATH)
        print(f"✅ Sucesso! Modelo salvo em: {MODEL_PATH}")
        return data_to_save

    except Exception as e:
        print(f"❌ Erro no treino: {e}")
        return None

if __name__ == "__main__":
    treinar_modelo_verbatins(ARQUIVO_TREINO)