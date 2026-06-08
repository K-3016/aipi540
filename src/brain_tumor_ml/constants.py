from pathlib import Path

CLASS_NAMES = ("glioma", "meningioma", "pituitary", "normal")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_SIZE = 128
DEFAULT_ARTIFACT_DIR = Path("models")
