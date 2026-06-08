from brain_tumor_ml.metrics import classification_metrics


def test_perfect_predictions_have_perfect_discrimination():
    metrics = classification_metrics(
        [0, 1, 2, 3],
        [
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ],
    )
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["glioma"]["sensitivity_recall"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == 1.0
    assert metrics["confidence_triage"][0]["coverage"] == 1.0
    assert metrics["confidence_triage"][0]["selective_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
