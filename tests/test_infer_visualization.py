import os
from pathlib import Path
import importlib.util

from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")


def _load_infer_module():
    infer_path = Path(__file__).resolve().parents[1] / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("infer_script", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load infer module from {infer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_infer_module = _load_infer_module()
render_image2text_figure = _infer_module.render_image2text_figure
render_text2image_figure = _infer_module.render_text2image_figure


def _make_image(path: Path, value: int):
    image = Image.new("RGB", (32, 32), color=(value, value, value))
    image.save(path)


def test_render_text2image_figure_saves_file(tmp_path):
    image1 = tmp_path / "img1.jpg"
    image2 = tmp_path / "img2.jpg"
    _make_image(image1, 50)
    _make_image(image2, 120)

    results = [
        {
            "query_text": "a cat on the mat",
            "matches": [
                {"image_path": str(image1), "score": 0.91},
                {"image_path": str(image2), "score": 0.83},
            ],
        }
    ]

    output = tmp_path / "text2image.png"
    render_text2image_figure(
        results=results,
        top_k=2,
        figure_path=str(output),
        mode_title="test-text2image",
        show=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_render_image2text_figure_saves_file(tmp_path):
    image1 = tmp_path / "img3.jpg"
    _make_image(image1, 80)

    results = [
        {
            "query_image": str(image1),
            "matches": [
                {"text": "a running dog", "score": 0.93},
                {"text": "a person in park", "score": 0.81},
            ],
        }
    ]

    output = tmp_path / "image2text.png"
    render_image2text_figure(
        results=results,
        top_k=2,
        figure_path=str(output),
        mode_title="test-image2text",
        show=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0
