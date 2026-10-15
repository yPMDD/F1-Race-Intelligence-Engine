import joblib
import json
import re
import difflib
import pandas as pd
from sqlalchemy import create_engine
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Prior used when a driver has no history at a circuit. Matches the NaN fill in
# prepare_data.py, so inference is consistent with training.
DEFAULT_POSITION = 12.0

# Similarity floor for fuzzy track matching. Tuned so "Bahrain GP" and "Grand
# Prix of Bahrain" resolve, while a genuinely unknown circuit does not.
TRACK_MATCH_THRESHOLD = 0.62

# fastf1 EventName values (what the model was trained on) differ from the
# names people actually use. "Dutch Grand Prix" is Zandvoort, "British Grand
# Prix" is Silverstone, and so on. Without this the agent is asked to predict
# "Zandvoort", fails to resolve it, and returns an error -- or worse, used to
# silently fall through to index 0.
TRACK_ALIASES = {
    "zandvoort": "Dutch Grand Prix",
    "zandvoort gp": "Dutch Grand Prix",
    "dutch gp": "Dutch Grand Prix",
    "silverstone": "British Grand Prix",
    "silverstone circuit": "British Grand Prix",
    "monza": "Italian Grand Prix",
    "autodromo di monza": "Italian Grand Prix",
    "imola": "Emilia Romagna Grand Prix",
    "imola gp": "Emilia Romagna Grand Prix",
    "spa": "Belgian Grand Prix",
    "spa francorchamps": "Belgian Grand Prix",
    "interlagos": "São Paulo Grand Prix",
    "interlagos gp": "São Paulo Grand Prix",
    "sepang": "Malaysian Grand Prix",
    "sepang international circuit": "Malaysian Grand Prix",
    "yas marina": "Abu Dhabi Grand Prix",
    "yas marina circuit": "Abu Dhabi Grand Prix",
    "marina bay": "Singapore Grand Prix",
    "marina bay street circuit": "Singapore Grand Prix",
    "shanghai": "Chinese Grand Prix",
    "suzuka": "Japanese Grand Prix",
    "hungaroring": "Hungarian Grand Prix",
    "hungaroring gp": "Hungarian Grand Prix",
    "cota": "United States Grand Prix",
    "circuit of the americas": "United States Grand Prix",
    "miami international autodrome": "Miami Grand Prix",
    "albert park": "Australian Grand Prix",
    "albert park circuit": "Australian Grand Prix",
    "catalunya": "Spanish Grand Prix",
    "catalunya circuit": "Spanish Grand Prix",
    "monza circuit": "Italian Grand Prix",
    "yas marina gp": "Abu Dhabi Grand Prix",
    "baku city circuit": "Azerbaijan Grand Prix",
    "baku": "Azerbaijan Grand Prix",
    "mexican autodrome": "Mexico City Grand Prix",
    "autodromo brothers rodriguez": "Mexico City Grand Prix",
    "jeddah corniche circuit": "Saudi Arabian Grand Prix",
    "jeddah": "Saudi Arabian Grand Prix",
    "losail international circuit": "Qatar Grand Prix",
    "losail": "Qatar Grand Prix",
    "bahrain international circuit": "Bahrain Grand Prix",
    "monaco circuit": "Monaco Grand Prix",
    "circuit de monaco": "Monaco Grand Prix",
}

# Tokens that carry no discriminating information in circuit names.
_TRACK_NOISE = re.compile(
    r"\b(grand|prix|gp|formula|world|fia|at|the|of|circuit|autodromo|autódromo)\b"
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class F1Predictor:
    # Class-level cache of driver form pulled from Postgres. Shared across
    # instances because recomputing it per request is wasteful; must be
    # explicitly invalidated when results change (see invalidate_cache).
    _stats_cache = None

    def __init__(self):
        base_path = "storage/models"
        self.model_path = os.path.join(base_path, "f1_predictor.joblib")
        self.mapping_path = os.path.join(base_path, "mappings.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("ML Model not found. Run train_model.py first.")

        self.model = joblib.load(self.model_path)
        with open(self.mapping_path, "r") as f:
            self.mappings = json.load(f)

        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://f1user:f1password@localhost:5433/f1db")
        self.engine = create_engine(DATABASE_URL)

    # -- cache ------------------------------------------------------------
    @classmethod
    def invalidate_cache(cls):
        """Drop the memoised driver stats. Call after new results land."""
        cls._stats_cache = None

    # -- track resolution -------------------------------------------------
    @staticmethod
    def _normalize_track(name: str) -> str:
        """Reduce a circuit name to its distinguishing part.

        "Grand Prix of Bahrain", "Bahrain GP" and "Bahrain" all collapse to
        "bahrain". Folding accents means "Sao Paulo" matches "São Paulo".
        """
        import unicodedata

        n = unicodedata.normalize("NFKD", str(name).lower())
        n = "".join(ch for ch in n if not unicodedata.combining(ch))
        n = _NON_ALNUM.sub(" ", n)
        n = _TRACK_NOISE.sub(" ", n)
        return " ".join(n.split())

    def resolve_track(self, track_name: str) -> tuple:
        """Map a free-text circuit name onto a trained (name, index) pair.

        The previous code did `self.mappings["tracks"].get(track_name, 0)`, an
        exact lookup whose fallback was index 0 -- so "Bahrain" silently
        returned "Abu Dhabi Grand Prix" and the model produced a confidently
        wrong forecast with no error anywhere.

        Resolution order:
          1. exact (case/space insensitive)
          2. alias table ("Zandvoort" -> "Dutch Grand Prix")
          3. normalised match ("Grand Prix of Bahrain" -> "bahrain")
          4. fuzzy, above TRACK_MATCH_THRESHOLD

        Raises ValueError rather than guessing when nothing is confident.
        """
        if not track_name or not str(track_name).strip():
            raise ValueError("Track name is empty; cannot build a prediction.")

        tracks = self.mappings["tracks"]
        query = str(track_name).strip()

        # 1. exact, ignoring case and surrounding whitespace
        for name in tracks:
            if name.strip().lower() == query.lower():
                return name, tracks[name]

        # 2. alias table, for the circuit names people actually use
        alias = TRACK_ALIASES.get(self._normalize_track(query))
        if alias is None:
            alias = TRACK_ALIASES.get(query.lower().strip())
        if alias and alias in tracks:
            logger.info("Resolved track %r via alias -> %r", track_name, alias)
            return alias, tracks[alias]

        # 3. normalised
        target = self._normalize_track(query)
        if target:
            matches = [n for n in tracks if self._normalize_track(n) == target]
            if len(matches) == 1:
                return matches[0], tracks[matches[0]]
            if len(matches) > 1:
                raise ValueError(
                    f"Track name {track_name!r} is ambiguous between "
                    f"{sorted(matches)}. Pass a more specific name."
                )

        # 4. Fuzzy match on the NORMALISED forms only.
        #    Comparing the raw strings was actively harmful: every candidate
        #    ends in " Grand Prix", so an unrelated circuit scored ~0.73 and
        #    cleared the threshold ("Nurburgring Grand Prix" -> "Austrian
        #    Grand Prix"). Stripping the shared boilerplate first leaves
        #    "nurburgring" vs "austrian" = 0.42, which correctly fails.
        best_name, best_score = None, 0.0
        if target:
            for name in tracks:
                score = difflib.SequenceMatcher(
                    None, target, self._normalize_track(name)
                ).ratio()
                if score > best_score:
                    best_name, best_score = name, score

        if best_name is not None and best_score >= TRACK_MATCH_THRESHOLD:
            logger.info(
                "Resolved track %r -> %r (similarity %.2f)",
                track_name, best_name, best_score,
            )
            return best_name, tracks[best_name]

        raise ValueError(
            f"Unknown track {track_name!r} (best match {best_name!r} scored "
            f"{best_score:.2f}, threshold {TRACK_MATCH_THRESHOLD}). "
            f"Known tracks: {', '.join(sorted(tracks))}"
        )

    # -- data access ------------------------------------------------------
    def _fetch_driver_stats(self) -> dict:
        """Per-driver recent form, cached across requests."""
        if F1Predictor._stats_cache is None:
            query = """
            WITH ranked AS (
                SELECT
                    res.driver_id,
                    res.position,
                    res.points,
                    r.year,
                    r.round_num,
                    ROW_NUMBER() OVER (
                        PARTITION BY res.driver_id
                        ORDER BY r.year DESC, r.round_num DESC
                    ) AS recency
                FROM results res
                JOIN races r ON res.race_id = r.id
            )
            SELECT
                driver_id,
                AVG(position) FILTER (WHERE recency <= 5) AS momentum,
                SUM(points) FILTER (WHERE year = 2026) AS season_points
            FROM ranked
            GROUP BY driver_id
            """
            stats_df = pd.read_sql(query, self.engine)
            F1Predictor._stats_cache = stats_df.set_index("driver_id").to_dict("index")
        return F1Predictor._stats_cache

    def _fetch_track_history(self, canonical_track: str) -> dict:
        """Driver's mean finishing position at this circuit.

        Mirrors the training-time `track_history` feature (expanding mean of
        position at that race name), evaluated over whatever history exists now.
        """
        query = """
        SELECT res.driver_id, AVG(res.position) AS avg_position
        FROM results res
        JOIN races r ON res.race_id = r.id
        WHERE r.name = :track
        GROUP BY res.driver_id
        """
        try:
            df = pd.read_sql(query, self.engine, params={"track": canonical_track})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Track-history lookup failed for %r: %s", canonical_track, exc)
            return {}
        if df.empty:
            return {}
        return {row["driver_id"]: float(row["avg_position"]) for _, row in df.iterrows()}

    # -- feature assembly -------------------------------------------------
    def _build_features(self, canonical_track: str, stats_dict: dict, track_idx: int) -> pd.DataFrame:
        from ml.config import GRID_2026

        driver_map = self.mappings["drivers"]
        known_idx = [driver_map.get(d["driver_id"]) for d in GRID_2026]
        # Unknown drivers get the median known code rather than -1: a negative
        # index is out-of-vocabulary for the categorical split and sends the
        # tree down an arbitrary branch.
        fallback_idx = int(pd.Series([i for i in known_idx if i is not None]).median()) \
            if any(i is not None for i in known_idx) else 0

        rows = []
        for entry in GRID_2026:
            driver_id = entry["driver_id"]
            d_stats = stats_dict.get(driver_id) or {}
            momentum = d_stats.get("momentum")
            season_points = d_stats.get("season_points")
            track_history = self._track_history.get(driver_id)

            rows.append({
                "driver_momentum": (
                    float(momentum) if momentum is not None and pd.notnull(momentum)
                    else DEFAULT_POSITION
                ),
                "track_history": (
                    float(track_history) if track_history is not None and pd.notnull(track_history)
                    else DEFAULT_POSITION
                ),
                "season_points": (
                    float(season_points) if season_points is not None and pd.notnull(season_points)
                    else 0.0
                ),
                "driver_idx": driver_map.get(driver_id, fallback_idx),
                "track_idx": track_idx,
                # Flag for the 2026 regulation era, matching training.
                "reg_impact": 1.0,
            })

        return pd.DataFrame(rows)

    # -- public API -------------------------------------------------------
    def predict_race_order(self, track_name: str, include_history: bool = True):
        """Batch-predict the finishing order for the official 2026 grid."""
        from ml.config import GRID_2026

        # 1. Resolve the circuit before doing any work, so an unrecognised name
        #    fails fast instead of silently forecasting the wrong track.
        canonical_track, track_idx = self.resolve_track(track_name)

        # 2. Driver form (cached across requests).
        stats_dict = self._fetch_driver_stats()

        # 3. Per-circuit history for the track_history feature.
        self._track_history = (
            self._fetch_track_history(canonical_track) if include_history else {}
        )

        # 4. Assemble features for the whole grid, then one batched predict.
        features_df = self._build_features(canonical_track, stats_dict, track_idx)
        scores = self.model.predict(features_df)

        predictions = []
        for i, entry in enumerate(GRID_2026):
            predictions.append({
                "driver_id": entry["driver_id"],
                "name": entry["name"],
                "team": entry["team"],
                "score": float(scores[i]),
                "track_history": float(features_df.iloc[i]["track_history"]),
            })

        # Lower predicted position is better.
        ranked = sorted(predictions, key=lambda x: x["score"])
        for i, item in enumerate(ranked):
            item["final_pos"] = i + 1

        return ranked


if __name__ == "__main__":
    predictor = F1Predictor()
    print("Testing ML Prediction for Zandvoort...")
    results = predictor.predict_race_order("Dutch Grand Prix")
    for res in results[:10]:
        print(
            f"P{res['final_pos']}: {res['driver_id']} "
            f"(Score: {res['score']:.2f}, track_history: {res['track_history']:.1f})"
        )
