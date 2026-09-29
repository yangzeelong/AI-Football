"""Touch business logic for the AI-Football pipeline, in Python.

The C++ side of this repository is the base model: it runs the detector and the
pose network and writes one line per frame to `observations.jsonl`. Everything
downstream of that file - reading annotations, deciding which contacts are
touches, scoring them and rendering the review video - lives here.

`detect.py` is a line by line port of `examples/touch_eval/src/TouchDetector.cpp`
and `observations.py` of `ObservationJsonlReader.cpp`; `tests/test_parity_cpp.py`
runs both implementations over the whole recording and requires them to agree
event for event, which is what makes the port safe to rely on.
"""

__all__ = ["detect", "observations", "params"]
__version__ = "0.1.0"
