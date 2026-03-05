"""Tests for subprocess_helpers.py -- game score parsing and log file discovery."""
import os
import time
import pytest

from subprocess_helpers import (
    parse_game_score, parse_round_progress, find_latest_game_log,
    GAME_OVER_RE, ROUND_RE,
)


# ---------------------------------------------------------------------------
# parse_game_score
# ---------------------------------------------------------------------------

class TestParseGameScore:
    def test_standard_format(self):
        stderr = "GAME_OVER Score:137\n"
        assert parse_game_score(stderr) == 137

    def test_with_space_after_colon(self):
        stderr = "GAME_OVER Score: 200\n"
        assert parse_game_score(stderr) == 200

    def test_with_surrounding_text(self):
        stderr = (
            "R0/300 Score:0\n"
            "R100/300 Score:50\n"
            "GAME_OVER Score:166\n"
            "Log saved: game_log.jsonl\n"
        )
        assert parse_game_score(stderr) == 166

    def test_multiple_game_over_takes_last(self):
        stderr = (
            "GAME_OVER Score:100\n"
            "Post-optimize...\n"
            "GAME_OVER Score:120\n"
        )
        assert parse_game_score(stderr) == 120

    def test_no_game_over_returns_zero(self):
        stderr = "R50/300 Score:42\nSomething else\n"
        assert parse_game_score(stderr) == 0

    def test_empty_string_returns_zero(self):
        assert parse_game_score("") == 0

    def test_zero_score(self):
        assert parse_game_score("GAME_OVER Score:0\n") == 0

    def test_large_score(self):
        assert parse_game_score("GAME_OVER Score:9999\n") == 9999

    def test_with_spaces_around(self):
        stderr = "GAME_OVER  Score:  42\n"
        # The regex uses \s+ between GAME_OVER and Score, so double space works.
        # But Score: has no flexibility for extra space before the number --
        # check what the regex actually matches.
        result = parse_game_score(stderr)
        # GAME_OVER_RE = r'GAME_OVER\s+Score[:\s]+(\d+)' -- [:\s]+ matches ": " or ":  "
        assert result == 42


# ---------------------------------------------------------------------------
# parse_round_progress
# ---------------------------------------------------------------------------

class TestParseRoundProgress:
    def test_standard_format(self):
        result = parse_round_progress("R50/300 Score:42")
        assert result == (50, 300, 42)

    def test_round_zero(self):
        result = parse_round_progress("R0/300 Score:0")
        assert result == (0, 300, 0)

    def test_last_round(self):
        result = parse_round_progress("R299/300 Score:137")
        assert result == (299, 300, 137)

    def test_with_prefix(self):
        result = parse_round_progress("[INFO] R100/300 Score:80 [SYNC]")
        assert result == (100, 300, 80)

    def test_no_match_returns_none(self):
        assert parse_round_progress("some random text") is None
        assert parse_round_progress("") is None
        assert parse_round_progress("GAME_OVER Score:100") is None

    def test_with_suffix_text(self):
        result = parse_round_progress("R25/300 Score:15 (synced:20 desynced:5)")
        assert result == (25, 300, 15)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_game_over_re_matches(self):
        m = GAME_OVER_RE.search("GAME_OVER Score:137")
        assert m is not None
        assert m.group(1) == "137"

    def test_game_over_re_with_whitespace(self):
        m = GAME_OVER_RE.search("GAME_OVER  Score: 200")
        assert m is not None
        assert m.group(1) == "200"

    def test_round_re_matches(self):
        m = ROUND_RE.search("R50/300 Score:42")
        assert m is not None
        assert m.group(1) == "50"
        assert m.group(2) == "300"
        assert m.group(3) == "42"


# ---------------------------------------------------------------------------
# find_latest_game_log
# ---------------------------------------------------------------------------

class TestFindLatestGameLog:
    def test_empty_directory_returns_none(self, tmp_path):
        result = find_latest_game_log(str(tmp_path))
        assert result is None

    def test_single_log_file(self, tmp_path):
        log = tmp_path / "game_log_12345.jsonl"
        log.write_text("{}\n")
        result = find_latest_game_log(str(tmp_path))
        assert result is not None
        assert os.path.basename(result) == "game_log_12345.jsonl"

    def test_returns_newest_by_mtime(self, tmp_path):
        log1 = tmp_path / "game_log_10000.jsonl"
        log1.write_text("{}\n")
        # Ensure different mtimes
        time.sleep(0.05)
        log2 = tmp_path / "game_log_20000.jsonl"
        log2.write_text("{}\n")
        result = find_latest_game_log(str(tmp_path))
        assert os.path.basename(result) == "game_log_20000.jsonl"

    def test_excludes_existing_logs(self, tmp_path):
        log1 = tmp_path / "game_log_10000.jsonl"
        log1.write_text("{}\n")
        existing = {str(log1)}
        time.sleep(0.05)
        log2 = tmp_path / "game_log_20000.jsonl"
        log2.write_text("{}\n")
        result = find_latest_game_log(str(tmp_path), existing_logs=existing)
        assert os.path.basename(result) == "game_log_20000.jsonl"

    def test_all_excluded_falls_back_to_newest(self, tmp_path):
        log1 = tmp_path / "game_log_10000.jsonl"
        log1.write_text("{}\n")
        time.sleep(0.05)
        log2 = tmp_path / "game_log_20000.jsonl"
        log2.write_text("{}\n")
        existing = {str(log1), str(log2)}
        result = find_latest_game_log(str(tmp_path), existing_logs=existing)
        # Falls back to newest overall
        assert os.path.basename(result) == "game_log_20000.jsonl"

    def test_non_matching_files_ignored(self, tmp_path):
        (tmp_path / "other.txt").write_text("hello")
        (tmp_path / "game_log_99999.jsonl").write_text("{}\n")
        result = find_latest_game_log(str(tmp_path))
        assert os.path.basename(result) == "game_log_99999.jsonl"

    def test_returns_full_path(self, tmp_path):
        log = tmp_path / "game_log_55555.jsonl"
        log.write_text("{}\n")
        result = find_latest_game_log(str(tmp_path))
        assert os.path.isabs(result)
        assert os.path.exists(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
