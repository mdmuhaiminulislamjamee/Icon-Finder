from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_and_runs_demo():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert not app.exception
    assert "Find familiar shapes" in app.title[0].value
    demo = next(s for s in app.selectbox if s.label == "Or try an example")
    demo.select("Desktop · light").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Find icons →").click().run()
    assert not app.exception
    assert int(next(m for m in app.metric if m.label == "Matches").value) == 8
    assert len(app.session_state["detection"][1].results) == 8
    next(s for s in app.slider if s.label == "Minimum match score").set_value(0.95).run()
    assert not app.exception
    assert not app.metric
