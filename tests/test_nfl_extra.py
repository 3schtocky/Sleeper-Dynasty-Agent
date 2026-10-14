import pyarrow as pa
import pyarrow.parquet as pq

from dynasty_agent import nfl_extra


def write_injuries(cache_dir, season, rows):
    """A file shaped like nflverse's: season and week arrive as floats."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "season": pa.array([float(r[0]) for r in rows], pa.float64()),
        "week": pa.array([None if r[1] is None else float(r[1]) for r in rows], pa.float64()),
        "team": [r[2] for r in rows],
        "gsis_id": [r[3] for r in rows],
        "position": [r[4] for r in rows],
        "report_status": [r[5] for r in rows],
    })
    pq.write_table(table, cache_dir / f"injuries_{season}.parquet")


def test_ingest_casts_floats_skips_unusable_rows_and_is_rerunnable(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_extra, "NFLVERSE_CACHE_DIR", tmp_path)
    write_injuries(tmp_path, 2024, [
        (2024, 5, "KC", "00-1", "WR", "Out"),
        (2024, 5, "KC", "00-2", "RB", "Questionable"),
        (2024, 5, "KC", None, "TE", "Out"),   # no player id: unusable
        (2024, None, "KC", "00-3", "QB", "Out"),  # no week: unusable
    ])
    assert nfl_extra.ingest_injuries(conn, 2024) == 2
    assert nfl_extra.ingest_injuries(conn, 2024) == 2  # upsert, not duplicate
    assert conn.execute("SELECT COUNT(*) FROM nfl_injuries").fetchone()[0] == 2
    row = conn.execute("SELECT typeof(season) t, typeof(week) w FROM nfl_injuries LIMIT 1").fetchone()
    assert (row["t"], row["w"]) == ("integer", "integer")


def test_report_status_is_per_week_and_none_means_healthy(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_extra, "NFLVERSE_CACHE_DIR", tmp_path)
    write_injuries(tmp_path, 2024, [(2024, 5, "KC", "00-1", "WR", "Out")])
    nfl_extra.ingest_injuries(conn, 2024)
    assert nfl_extra.report_status_for_week(conn, 2024, 5, "00-1") == "Out"
    assert nfl_extra.report_status_for_week(conn, 2024, 6, "00-1") is None  # not on week 6's report
    assert nfl_extra.report_status_for_week(conn, 2024, 5, "00-9") is None


def test_a_cached_file_is_not_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_extra, "NFLVERSE_CACHE_DIR", tmp_path)
    (tmp_path / "injuries_2024.parquet").write_bytes(b"cached")
    monkeypatch.setattr(nfl_extra.nflverse, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert nfl_extra.injuries_path(2024) == tmp_path / "injuries_2024.parquet"


def test_a_missing_file_downloads_from_the_injuries_release(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_extra, "NFLVERSE_CACHE_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(nfl_extra.nflverse, "download", lambda url, dest: seen.append((url, dest)) or dest)
    nfl_extra.injuries_path(2018)
    assert seen == [("https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2018.parquet",
                     tmp_path / "injuries_2018.parquet")]


def test_injuries_stay_out_of_the_forced_refresh_list():
    from dynasty_agent import nflverse
    assert "injuries" not in nflverse.FILES
