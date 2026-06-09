import numpy as np
from PIL import Image

from brain_tumor_ml.data import ImageRecord
from brain_tumor_ml.error_analysis import save_error_analysis


def test_error_analysis_saves_mistake_and_report(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("L", (24, 24), color=100).save(image_path)
    records = [
        ImageRecord(
            path=str(image_path),
            label=0,
            class_name="glioma",
            patient_id="patient_1",
            split="test",
        )
    ]
    rows = save_error_analysis(
        records,
        np.asarray([[0.1, 0.7, 0.1, 0.1]]),
        tmp_path / "report",
    )

    assert len(rows) == 1
    assert rows[0]["true_label"] == "glioma"
    assert rows[0]["predicted_label"] == "meningioma"
    assert (tmp_path / "report" / "error_analysis.csv").is_file()
    assert len(list((tmp_path / "report" / "images").glob("*.png"))) == 1
