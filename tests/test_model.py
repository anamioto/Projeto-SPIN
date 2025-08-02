import pytest
import pandas as pd
from treina_modelo_mlflow_azure import treinar_modelo_xgb

@pytest.fixture
def dataset_exemplo():
    X = pd.DataFrame({
        "f1": [0.1, 0.2, 0.9, 1.2],
        "f2": [1, 2, 1, 3]
    })
    y = pd.Series([0, 0, 1, 1])
    return X, X, y, y  # X_train, X_test, y_train, y_test (para facilitar o mock)

def test_treinar_modelo_xgb(dataset_exemplo):
    X_train, X_test, y_train, y_test = dataset_exemplo
    model, score = treinar_modelo_xgb(X_train, y_train, X_test, y_test)
    # Checa se retorna um modelo treinado e um score válido
    assert hasattr(model, "predict")
    assert 0 <= score <= 1
