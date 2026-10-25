
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def test_model_training():
    iris = load_iris()
    model = RandomForestClassifier(random_state=42)

    model.fit(iris.data, iris.target)

    predictions = model.predict(iris.data[:5])

    assert len(predictions) == 5
    assert all(p in [0, 1, 2] for p in predictions)

    print("Model test passed!")


if __name__ == "__main__":
    test_model_training()
