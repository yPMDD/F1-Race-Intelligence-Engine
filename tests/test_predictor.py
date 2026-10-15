"""Tests for ml/predictor.py — track resolution and feature construction.

Two defects:
  * `self.mappings["tracks"].get(track_name, 0)` is an exact-match lookup that
    silently falls back to index 0. "Bahrain" instead of "Bahrain Grand Prix"
    returned Abu Dhabi (idx 0) with no error -- a confidently wrong forecast.
  * `track_history` was hardcoded to 10.0 at inference even though the model
    ranks it 4th of 6 features (importance 0.103) and training values span
    1.0-20.0. The feature was effectively dead.
"""
import pytest

from ml.predictor import F1Predictor


class DummyPredictor(F1Predictor):
    """Bypass __init__ (which needs the joblib artifact + a live engine)."""

    def __init__(self, tracks, drivers=None):
        self.mappings = {
            "tracks": tracks,
            "drivers": drivers or {f"D{i}": i for i in range(30)},
        }
        self.model = None
        self.engine = None


REAL_TRACKS = {
    "Abu Dhabi Grand Prix": 0,
    "Australian Grand Prix": 1,
    "Austrian Grand Prix": 2,
    "Azerbaijan Grand Prix": 3,
    "Bahrain Grand Prix": 4,
    "Monaco Grand Prix": 5,
    "Zandvoort Grand Prix": 6,
}


# --- track resolution ------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "Bahrain Grand Prix",   # exact
        "bahrain grand prix",   # case
        "  Bahrain Grand Prix ",  # whitespace
        "Bahrain",              # bare circuit name
        "Bahrain GP",           # common abbreviation
        "Grand Prix of Bahrain",
    ],
)
def test_resolves_track_variants(query):
    p = DummyPredictor(REAL_TRACKS)
    name, idx = p.resolve_track(query)
    assert name == "Bahrain Grand Prix"
    assert idx == REAL_TRACKS["Bahrain Grand Prix"]


def test_does_not_silently_fall_back_to_index_zero():
    """The old code returned mappings['tracks'].get(name, 0) for anything."""
    p = DummyPredictor(REAL_TRACKS)
    with pytest.raises(ValueError):
        p.resolve_track("Nurburgring Grand Prix")


def test_error_message_lists_known_tracks():
    p = DummyPredictor(REAL_TRACKS)
    with pytest.raises(ValueError) as exc:
        p.resolve_track("Nurburgring")
    msg = str(exc.value)
    assert "Nurburgring" in msg
    assert "Bahrain Grand Prix" in msg


def test_ambiguous_prefix_is_rejected():
    """'Grand' alone matches everything -- must not silently pick one."""
    p = DummyPredictor(REAL_TRACKS)
    with pytest.raises(ValueError):
        p.resolve_track("Grand")


def test_empty_track_name_rejected():
    p = DummyPredictor(REAL_TRACKS)
    with pytest.raises(ValueError):
        p.resolve_track("")


def test_every_configured_track_resolves_to_itself():
    p = DummyPredictor(REAL_TRACKS)
    for name, idx in REAL_TRACKS.items():
        assert p.resolve_track(name) == (name, idx)


def test_alias_resolves_common_circuit_nicknames():
    """fastf1 calls Zandvoort 'Dutch GP'; the agent will be asked 'Zandvoort'."""
    tracks = dict(REAL_TRACKS)
    tracks["Dutch Grand Prix"] = 9
    tracks["British Grand Prix"] = 10
    p = DummyPredictor(tracks)

    assert p.resolve_track("Zandvoort") == ("Dutch Grand Prix", 9)
    assert p.resolve_track("Silverstone") == ("British Grand Prix", 10)
    assert p.resolve_track("ZANDVOORT") == ("Dutch Grand Prix", 9)


def test_alias_pointing_at_absent_track_is_rejected():
    """An alias whose canonical name is not in the mapping must not pass."""
    p = DummyPredictor(REAL_TRACKS)  # no "Malaysian Grand Prix" here
    with pytest.raises(ValueError):
        p.resolve_track("Sepang")


def test_accents_are_folded():
    tracks = dict(REAL_TRACKS)
    tracks["São Paulo Grand Prix"] = 20
    p = DummyPredictor(tracks)
    assert p.resolve_track("Sao Paulo") == ("São Paulo Grand Prix", 20)
    assert p.resolve_track("São Paulo Grand Prix") == ("São Paulo Grand Prix", 20)


def test_generic_boilerplate_is_stripped_before_fuzzy_matching():
    """The shared ' Grand Prix' suffix must not inflate similarity."""
    p = DummyPredictor(REAL_TRACKS)
    # "Nurburgring Grand Prix" scored 0.73 against "Austrian Grand Prix" when
    # compared as raw strings, and wrongly resolved.
    with pytest.raises(ValueError):
        p.resolve_track("Nurburgring Grand Prix")


# --- feature construction --------------------------------------------------

def test_build_features_uses_one_row_per_grid_driver(monkeypatch):
    from ml.config import GRID_2026

    p = DummyPredictor(REAL_TRACKS, {d["driver_id"]: i for i, d in enumerate(GRID_2026)})
    p._track_history = {d["driver_id"]: 4.0 for d in GRID_2026}

    df = p._build_features("Bahrain Grand Prix", {}, track_idx=4)
    assert len(df) == len(GRID_2026) == 22
    assert set(df.columns) >= {
        "driver_momentum", "track_history", "season_points",
        "driver_idx", "track_idx", "reg_impact",
    }


def test_track_history_is_not_a_constant(monkeypatch):
    """Each driver must contribute their own historical track form."""
    from ml.config import GRID_2026

    p = DummyPredictor(REAL_TRACKS, {d["driver_id"]: i for i, d in enumerate(GRID_2026)})
    p._track_history = {
        d["driver_id"]: float(i + 1) for i, d in enumerate(GRID_2026)
    }
    df = p._build_features("Bahrain Grand Prix", {}, track_idx=4)
    assert df["track_history"].nunique() > 1


def test_missing_track_history_falls_back_to_grid_median():
    """prepare_data.py fills NaN momentum/history with 12.0 -- stay consistent."""
    from ml.config import GRID_2026

    p = DummyPredictor(REAL_TRACKS, {d["driver_id"]: i for i, d in enumerate(GRID_2026)})
    p._track_history = {}
    df = p._build_features("Bahrain Grand Prix", {}, track_idx=4)
    assert (df["track_history"] == 12.0).all()


def test_unknown_driver_does_not_get_negative_one():
    """-1 is an out-of-vocab code that sends a tree model down a stray branch."""
    from ml.config import GRID_2026

    p = DummyPredictor(REAL_TRACKS, {})  # empty driver mapping
    p._track_history = {}
    df = p._build_features("Bahrain Grand Prix", {}, track_idx=4)
    assert (df["driver_idx"] >= 0).all()


def test_null_stats_do_not_produce_null_features():
    p = DummyPredictor(REAL_TRACKS)
    p._track_history = {}
    stats = {"VER": {"momentum": None, "season_points": None}}
    df = p._build_features("Bahrain Grand Prix", stats, track_idx=0)
    assert not df.isnull().any().any()
    assert (df["driver_momentum"] == 12.0).any()


def test_reg_impact_is_always_one_for_the_2026_grid():
    p = DummyPredictor(REAL_TRACKS)
    p._track_history = {}
    df = p._build_features("Bahrain Grand Prix", {}, track_idx=0)
    assert (df["reg_impact"] == 1.0).all()


# --- caching ---------------------------------------------------------------

def test_invalidate_cache_resets_class_level_stats():
    F1Predictor._stats_cache = {"VER": {"momentum": 1.0}}
    F1Predictor.invalidate_cache()
    assert F1Predictor._stats_cache is None
