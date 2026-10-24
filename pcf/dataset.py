"""Build one row per (pitcher, forecast date) with as-of features and next-30-day outcomes.

Every feature uses games strictly before the forecast date. Outcomes cover the forecast date
through 29 days later. Run: python3 -m pcf.dataset
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAW = "data/raw/statcast"
OUT = Path("data/processed/forecast_rows.parquet")

HORIZON = 30
FIRST_FORECAST_SEASON = 2021
RECENT = 30          # "current" window for results and pitch characteristics
BASELINE = (31, 120) # earlier window that recent changes are measured against
ELIG_WINDOW, ELIG_BF, ELIG_LAST_SEEN = 60, 50, 14
BATTER_PRIOR_PA, TEAM_PRIOR_PA = 150, 600  # shrinkage strengths toward the league rate

FASTBALLS = {"FF", "SI", "FA"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}


def load_pitches(through_season=None) -> pd.DataFrame:
    files = sorted(glob.glob(f"{RAW}/*/*.parquet"))
    if through_season is not None:
        files = [f for f in files if int(Path(f).parent.name) <= through_season]
    d = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    d = d[d.game_type == "R"].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    d["game_date"] = pd.to_datetime(d.game_date)
    d["season"] = d.game_date.dt.year
    d["pitch_team"] = np.where(d.inning_topbot == "Top", d.home_team, d.away_team)
    d["bat_team"] = np.where(d.inning_topbot == "Top", d.away_team, d.home_team)
    # Arm-side horizontal movement and release side are positive for both hands.
    arm = np.where(d.p_throws == "R", -1.0, 1.0)
    d["hb"] = d.pfx_x * 12 * arm
    d["ivb"] = d.pfx_z * 12
    d["rel_side"] = d.release_pos_x * arm
    d["swing"] = d.description.isin(SWINGS)
    d["whiff"] = d.description.isin(WHIFFS)
    d["in_zone"] = d.zone.between(1, 9)
    d["group"] = np.select(
        [d.pitch_type.isin(FASTBALLS), d.pitch_type == "FC", d.pitch_type.isin(BREAKING),
         d.pitch_type.isin(OFFSPEED)],
        ["fb", "cut", "brk", "off"], "other")
    return d


def plate_appearances(d: pd.DataFrame) -> pd.DataFrame:
    """Batters faced excludes intentional walks and truncated PAs; BB is unintentional only."""
    pa = d[d.events.notna() & (d.events != "truncated_pa") & (d.events != "intent_walk")]
    pa = pa[["game_pk", "game_date", "season", "pitcher", "batter", "pitch_team", "bat_team",
             "events", "at_bat_number"]].copy()
    pa["k"] = pa.events.isin(STRIKEOUTS).astype(int)
    pa["bb"] = (pa.events == "walk").astype(int)
    return pa


def shrunk(num, den, prior_rate, strength):
    return (num + strength * prior_rate) / (den + strength)


def as_of_daily_rates(pa: pd.DataFrame, key: str, strength: float) -> pd.DataFrame:
    """For each (key, date, season), shrunk K% and BB% using that season's games before the date
    plus half-weighted prior-season totals, shrunk toward the league rate as of the same date."""
    daily = pa.groupby([key, "season", "game_date"])[["k", "bb"]].agg(["sum", "count"])
    daily = pd.DataFrame({"k": daily[("k", "sum")], "bb": daily[("bb", "sum")],
                          "bf": daily[("k", "count")]}).reset_index().sort_values("game_date")
    g = daily.groupby([key, "season"])
    for c in ("k", "bb", "bf"):
        daily[f"{c}_before"] = g[c].cumsum() - daily[c]

    season_tot = daily.groupby([key, "season"])[["k", "bb", "bf"]].sum().reset_index()
    season_tot["season"] += 1
    daily = daily.merge(season_tot.rename(columns={"k": "k_ps", "bb": "bb_ps", "bf": "bf_ps"}),
                        on=[key, "season"], how="left").fillna({"k_ps": 0, "bb_ps": 0, "bf_ps": 0})

    lg = pa.groupby("game_date")[["k", "bb"]].agg(["sum", "count"])
    lg = pd.DataFrame({"lk": lg[("k", "sum")], "lbb": lg[("bb", "sum")], "lbf": lg[("k", "count")]})
    lg["season"] = lg.index.year
    lg = lg.groupby("season")[["lk", "lbb", "lbf"]].cumsum() - lg[["lk", "lbb", "lbf"]]
    lg_k, lg_bb = 0.225, 0.085  # opening-day fallback before any games are played
    lg["lg_k"] = np.where(lg.lbf > 0, lg.lk / lg.lbf.replace(0, 1), lg_k)
    lg["lg_bb"] = np.where(lg.lbf > 0, lg.lbb / lg.lbf.replace(0, 1), lg_bb)
    daily = daily.merge(lg[["lg_k", "lg_bb"]], left_on="game_date", right_index=True)

    num_k = daily.k_before + 0.5 * daily.k_ps
    num_bb = daily.bb_before + 0.5 * daily.bb_ps
    den = daily.bf_before + 0.5 * daily.bf_ps
    daily["k_rate"] = shrunk(num_k, den, daily.lg_k, strength)
    daily["bb_rate"] = shrunk(num_bb, den, daily.lg_bb, strength)
    return daily[[key, "game_date", "k_rate", "bb_rate"]]


def forecast_dates(d: pd.DataFrame) -> list[pd.Timestamp]:
    dates = []
    for season, s in d.groupby("season"):
        if season < FIRST_FORECAST_SEASON:
            continue
        first, last = s.game_date.min(), s.game_date.max()
        mondays = pd.date_range(first + pd.Timedelta(days=14), last, freq="W-MON")
        dates += [t for t in mondays if t + pd.Timedelta(days=HORIZON - 1) <= last]
    return dates


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def result_features(pa_w: pd.DataFrame, pitches_w: pd.DataFrame, tag: str) -> pd.DataFrame:
    r = pa_w.groupby("pitcher").agg(bf=("k", "size"), k=("k", "sum"), bb=("bb", "sum"),
                                    opp_k=("opp_k", "mean"), opp_bb=("opp_bb", "mean"))
    sw = pitches_w.groupby("pitcher").agg(swings=("swing", "sum"), whiffs=("whiff", "sum"))
    r = r.join(sw, how="left")
    return r.add_suffix(f"_{tag}")


def pitch_features(p: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Repertoire summary for one window: primary fastball traits, secondary-pitch traits relative
    to that fastball, usage shares, zone rate, and release consistency."""
    p = p[p.pitch_type.notna() & p.release_speed.notna()]
    n = p.groupby("pitcher").size().rename("n_pitches")
    share = p.groupby(["pitcher", "group"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).add_prefix("use_")
    fb = p[p.group == "fb"].groupby("pitcher").agg(
        fb_velo=("release_speed", "mean"), fb_ivb=("ivb", "mean"), fb_hb=("hb", "mean"),
        fb_spin=("release_spin_rate", "mean"), fb_ext=("release_extension", "mean"),
        fb_rel_z=("release_pos_z", "mean"), fb_rel_side=("rel_side", "mean"),
        fb_arm=("arm_angle", "mean"), fb_rel_z_sd=("release_pos_z", "std"),
        fb_velo_sd=("release_speed", "std"), fb_n=("release_speed", "size"))
    out = pd.concat([n, share, fb], axis=1)
    for g in ("brk", "off", "cut"):
        s = p[p.group == g].groupby("pitcher").agg(
            velo=("release_speed", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"),
            spin=("release_spin_rate", "mean"))
        out[f"{g}_velo_gap"] = out.fb_velo - s.velo
        out[f"{g}_ivb_gap"] = out.fb_ivb - s.ivb
        out[f"{g}_hb_gap"] = out.fb_hb - s.hb
        out[f"{g}_spin"] = s.spin
    out["zone_rate"] = p.groupby("pitcher").in_zone.mean()
    out["n_types"] = (p.groupby(["pitcher", "pitch_type"]).size()
                      .div(n, level="pitcher").ge(0.05).groupby("pitcher").sum())
    return out.add_suffix(f"_{tag}")


def build(through_season=None) -> pd.DataFrame:
    d = load_pitches(through_season)
    pa = plate_appearances(d)

    bat = as_of_daily_rates(pa, "batter", BATTER_PRIOR_PA).rename(
        columns={"k_rate": "opp_k", "bb_rate": "opp_bb"})
    pa = pa.merge(bat, on=["batter", "game_date"], how="left")
    team = as_of_daily_rates(pa.rename(columns={"bat_team": "team"}), "team", TEAM_PRIOR_PA)

    # Starter = first pitcher his team used in the game.
    first_pa = pa.sort_values("at_bat_number").groupby(["game_pk", "pitch_team"]).pitcher.first()
    apps = pa.groupby(["pitcher", "game_pk", "game_date", "season", "pitch_team"]).size()
    apps = apps.rename("bf").reset_index()
    apps["started"] = apps.set_index(["game_pk", "pitch_team"]).index.map(first_pa) == apps.pitcher

    games = d.groupby("game_pk").agg(game_date=("game_date", "first"),
                                     home=("home_team", "first"), away=("away_team", "first"))
    sched = pd.concat([games.rename(columns={"home": "team", "away": "opp"}),
                       games.rename(columns={"away": "team", "home": "opp"})])

    d_by = dict(tuple(d.groupby("season")))
    pa_by = dict(tuple(pa.groupby("season")))
    prior = {s: (result_features(pa_by[s - 1], d_by[s - 1], "ps"), pitch_features(d_by[s - 1], "ps"))
             for s in d_by if s - 1 in d_by}
    rows = []
    for t in forecast_dates(d):
        season = t.year
        ds, pas = d_by[season], pa_by[season]

        def before(df, a, b):
            return df[(df.game_date >= t - pd.Timedelta(days=b))
                      & (df.game_date <= t - pd.Timedelta(days=a))]

        recent_pa = before(pas, 1, RECENT)
        elig_pa = before(pas, 1, ELIG_WINDOW)
        elig = elig_pa.groupby("pitcher").agg(bf=("k", "size"), last=("game_date", "max"))
        elig = elig[(elig.bf >= ELIG_BF) & (elig.last >= t - pd.Timedelta(days=ELIG_LAST_SEEN))]
        if elig.empty:
            continue
        ids = elig.index

        f = pd.DataFrame(index=ids)
        f["forecast_date"], f["season"] = t, season
        f["days_since_last"] = (t - elig["last"]).dt.days
        f = f.join(result_features(recent_pa, before(ds, 1, RECENT), "r30"))
        f = f.join(result_features(pas[pas.game_date < t], ds[ds.game_date < t], "std"))
        ps_results, ps_pitches = prior.get(season, (pd.DataFrame(), pd.DataFrame()))
        f = f.join(ps_results)

        a = apps[(apps.season == season) & (apps.game_date < t)
                 & (apps.game_date >= t - pd.Timedelta(days=ELIG_WINDOW))]
        role = a.groupby("pitcher").agg(start_share=("started", "mean"), bf_per_app=("bf", "mean"),
                                        apps_60=("bf", "size"))
        f = f.join(role)
        cur_team = (apps[apps.game_date < t].sort_values("game_date")
                    .groupby("pitcher").pitch_team.last())
        f["team"] = cur_team.reindex(ids)

        # Upcoming opponents from the team schedule, rated as of the forecast date.
        horizon = sched[(sched.game_date >= t)
                        & (sched.game_date <= t + pd.Timedelta(days=HORIZON - 1))]
        team_now = (team[team.game_date < t].sort_values("game_date")
                    .groupby("team")[["k_rate", "bb_rate"]].last())
        up = horizon.join(team_now, on="opp").groupby("team").agg(
            sched_games=("opp", "size"), sched_opp_k=("k_rate", "mean"),
            sched_opp_bb=("bb_rate", "mean"))
        f = f.join(up, on="team")

        f = f.join(pitch_features(before(ds, 1, RECENT), "cur"))
        f = f.join(pitch_features(before(ds, *BASELINE), "base"))
        f = f.join(ps_pitches)

        fut = pas[(pas.game_date >= t) & (pas.game_date <= t + pd.Timedelta(days=HORIZON - 1))]
        y = fut.groupby("pitcher").agg(y_bf=("k", "size"), y_k=("k", "sum"), y_bb=("bb", "sum"))
        f = f.join(y).fillna({"y_bf": 0, "y_k": 0, "y_bb": 0})
        rows.append(f.copy().reset_index(names="pitcher"))

    out = pd.concat(rows, ignore_index=True)
    names = d.groupby("pitcher").player_name.last()
    out["name"] = out.pitcher.map(names)
    return add_derived(out)


def add_derived(f: pd.DataFrame) -> pd.DataFrame:
    """Shrunk rates (logit scale) and recent-minus-baseline changes. The baseline for a change is
    the 31-120 day window when it holds enough pitches, otherwise the prior season."""
    lg_k, lg_bb = 0.225, 0.085
    for w in ("r30", "std", "ps"):
        for c in ("bf", "k", "bb", "swings", "whiffs", "opp_k", "opp_bb"):
            if f"{c}_{w}" not in f:
                f[f"{c}_{w}"] = np.nan
    for w, m in (("r30", 60), ("std", 60), ("ps", 60)):
        f[f"k_logit_{w}"] = logit(shrunk(f[f"k_{w}"].fillna(0), f[f"bf_{w}"].fillna(0), lg_k, m))
        f[f"bb_logit_{w}"] = logit(shrunk(f[f"bb_{w}"].fillna(0), f[f"bf_{w}"].fillna(0), lg_bb, m))
        f[f"whiff_{w}"] = shrunk(f[f"whiffs_{w}"].fillna(0), f[f"swings_{w}"].fillna(0), 0.25, 100)
        f[f"log_bf_{w}"] = np.log1p(f[f"bf_{w}"].fillna(0))

    f = f.copy()
    traits = [c.removesuffix("_cur") for c in f.columns if c.endswith("_cur")]
    missing = pd.Series(np.nan, index=f.index)
    n_base, n_ps = f.get("n_pitches_base", missing), f.get("n_pitches_ps", missing)
    use_base = n_base.fillna(0) >= 150
    chg = {f"chg_{c}": f[f"{c}_cur"] - np.where(use_base, f.get(f"{c}_base", missing),
                                                f.get(f"{c}_ps", missing))
           for c in traits if c not in ("n_pitches", "fb_n")}
    chg["chg_baseline_is_ps"] = (~use_base).astype(int)
    chg["chg_base_pitches"] = np.where(use_base, n_base, n_ps)
    return pd.concat([f, pd.DataFrame(chg, index=f.index)], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-season", type=int, help="exclude later seasons from the build")
    out = build(parser.parse_args().through_season)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} rows, {out.pitcher.nunique()} pitchers, "
          f"{out.forecast_date.nunique()} forecast dates -> {OUT}")
