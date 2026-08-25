"""
Loads frontend/src/data/ipl_2026_squads.json and exposes typed helpers.

Assumptions (no signal in the data to do otherwise):
  - Batting order == playing_xi slot order ("1".."11").
  - playing_xi actually lists 12 players per team (slots "1".."12"). Per
    your clarification: the 12th player is bowling-only by design — they
    are never part of the batting lineup, only ever a bowling option. This
    matches the data exactly: every single 2026 squad's slot-12 player
    carries an `is_pacer` flag (i.e. is a designated bowler), unlike slots
    1-11 where several players have no `is_pacer` key at all (pure
    batters). So slot 12 is excluded from `Team.batting_order` entirely and
    always included in `Team.bowling_pool` — not conditionally, as a firm
    rule (see `Team.bowling_pool`).
  - A player counts as a "bowler" (eligible to be given the ball) iff the
    `is_pacer` key is present on them at all — True means pacer, False means
    a spinner/other bowler. Players with no `is_pacer` key are pure batters
    and are never selected to bowl. This matches your instruction: "the ones
    who have the is_pacer form the squad will be given to [bowler
    selection]".
  - Wicketkeeper is inferred from a "(WK)" suffix in the name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from ml_config import SQUADS_PATH


@dataclass(frozen=True)
class Player:
    name: str  # cleaned, e.g. "Sanju Samson" (no "(WK)"/"(C)" suffix)
    raw_name: str  # as it appears in the squad file
    team: str
    overseas: bool
    is_pacer: Optional[bool]  # None => not a designated bowler
    is_keeper: bool = False

    @property
    def is_bowler(self) -> bool:
        return self.is_pacer is not None


@dataclass(frozen=True)
class Team:
    code: str
    captain: str
    batting_order: list[Player] = field(default_factory=list)  # 11, slot order
    bowling_only_player: Player | None = (
        None  # slot 12 — never bats, see module docstring
    )
    bench: list[Player] = field(default_factory=list)

    @property
    def bowlers(self) -> list[Player]:
        """Designated bowlers within the on-field batting 11 only."""
        return [p for p in self.batting_order if p.is_bowler]

    @property
    def bowling_pool(self) -> list[Player]:
        """
        Every bowler actually available to be given an over: the XI's
        designated bowlers, plus the 12th squad member unconditionally —
        by design they only ever bowl, never bat (see module docstring).
        """
        pool = list(self.bowlers)
        if self.bowling_only_player is not None:
            pool.append(self.bowling_only_player)
        return pool


_NAME_TAG_RE = re.compile(r"\s*\((WK|C|VC|wk|c|vc)\)")


def _clean_name(raw_name: str) -> str:
    return _NAME_TAG_RE.sub("", raw_name).strip()


def _parse_player(raw: dict, team_code: str) -> Player:
    raw_name = raw["name"]
    return Player(
        name=_clean_name(raw_name),
        raw_name=raw_name,
        team=team_code,
        overseas=bool(raw.get("overseas", False)),
        is_pacer=raw.get("is_pacer", None),
        is_keeper="(WK)" in raw_name or "(wk)" in raw_name,
    )


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(SQUADS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_teams() -> dict[str, Team]:
    data = _load_raw()
    teams: dict[str, Team] = {}
    for code, info in data["teams"].items():
        slots = sorted(info["playing_xi"].items(), key=lambda kv: int(kv[0]))
        eleven, twelfth = slots[:11], slots[11:]
        batting_order = [_parse_player(p, code) for _, p in eleven]
        bowling_only_player = _parse_player(twelfth[0][1], code) if twelfth else None
        bench = [_parse_player(p, code) for p in info.get("bench", [])]
        teams[code] = Team(
            code=code,
            captain=info.get("captain", ""),
            batting_order=batting_order,
            bowling_only_player=bowling_only_player,
            bench=bench,
        )
    return teams


def list_team_codes() -> list[str]:
    return list(_load_teams().keys())


def get_team(code: str) -> Team:
    teams = _load_teams()
    if code not in teams:
        raise KeyError(
            f"Unknown team code '{code}'. Available: {', '.join(teams.keys())}"
        )
    return teams[code]


def tournament_name() -> str:
    return _load_raw().get("tournament", "")
