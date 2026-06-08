from brain_tumor_ml.api import index


def test_index_places_heatmap_element_outside_script():
    page = index()
    image_position = page.index('id="heatmap"')
    script_position = page.index("<script>")

    assert image_position < script_position
