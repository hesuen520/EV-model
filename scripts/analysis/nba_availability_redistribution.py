"""How are an absent NBA rotation player's minutes replaced?

Analysis behind the availability rule described in docs/MODEL_CARD.md
(basketball, "Availability").  It replays the roster state the feature
builder keeps, finds every team-game in which a rotation player (at least 12
expected minutes) did not play, and compares three ways of predicting the
quality of the rotation that actually took the floor:

* no adjustment;
* proportional redistribution of the missing minutes (what the plugin does);
* replacement at a fraction of the league or bench rate, with the fraction
  fitted on seasons up to 2024 only.

Read-only.  Needs an ingested NBA store:

    sportspred ingest --league nba
    python scripts/analysis/nba_availability_redistribution.py [data/store]

This is an analysis script, not part of the test suite; its figures change as
results are added.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sportspred.storage import Store
from sportspred.sports.basketball.features import slate_batches, BasketballConfig

store = Store(sys.argv[1] if len(sys.argv) > 1 else 'data/store'); cfg = BasketballConfig()
ev = store.events('nba'); ev = ev[~ev.stage.isin(['preseason','allstar','offseason'])]
ev = ev[(ev.status=='final') & ev.winner_id.notna()].sort_values(['start_time_utc','event_id']).reset_index(drop=True)
pb = store.read_table('nba','player_box')
st_ = {c: pb[c].fillna(0.0).to_numpy(float) for c in ("pts","fgm","fga","fta","ftm","oreb","dreb","stl","ast","blk","pf","tov")}
gs = (st_["pts"]+0.4*st_["fgm"]-0.7*st_["fga"]-0.4*(st_["fta"]-st_["ftm"])+0.7*st_["oreb"]+0.3*st_["dreb"]+st_["stl"]+0.7*st_["ast"]+0.7*st_["blk"]-0.4*st_["pf"]-st_["tov"])
minutes = pb["minutes"].fillna(0.0).to_numpy(float)
pidx,_ = pd.factorize(pb["player_id"]); groups = pb.groupby(["event_id","team_id"], sort=False).indices
gs_sum = np.zeros(pidx.max()+1); min_sum = np.zeros(pidx.max()+1)
league_rate = 0.4; rate_num = rate_den = 0.0
state = {}; team_season = {}
batch = slate_batches(ev.start_time_utc); season = pd.to_numeric(ev.season).to_numpy()
lr_f, lr_s, prior, decay = cfg.roster_fast_lr, cfg.roster_slow_lr, cfg.rating_prior_minutes, cfg.rating_decay
rating = lambda p: (gs_sum[p]+prior*league_rate)/(min_sum[p]+prior)
rows = []; pending = []
def flush():
    global league_rate, rate_num, rate_den
    for team, players, mins, scores in pending:
        st = state[team]; played = dict(zip(players.tolist(), mins.tolist()))
        for p in list(st):
            m = played.pop(p, 0.0); v = st[p]; v[0] += lr_f*(m-v[0]); v[1] += lr_s*(m-v[1])
            if m == 0.0 and v[0] < 0.2 and v[1] < 0.2: del st[p]
        for p, m in played.items():
            if m > 0: st[p] = [lr_f*m, lr_s*m]
        for p, m, g in zip(players, mins, scores):
            if m > 0:
                gs_sum[p] = decay*gs_sum[p]+g; min_sum[p] = decay*min_sum[p]+m; rate_num += g; rate_den += m
    pending.clear()
    if rate_den > 0:
        league_rate += 0.05*(rate_num/rate_den-league_rate); rate_num = rate_den = 0.0
eids = ev.event_id.to_numpy(); homes = ev.home_id.to_numpy(); aways = ev.away_id.to_numpy()
for i in range(len(ev)):
    if i and batch[i] != batch[i-1]: flush()
    for team in (homes[i], aways[i]):
        if team_season.get(team) != season[i]: state[team] = {}; team_season[team] = season[i]
        st = state[team]; r = groups.get((eids[i], team))
        if r is None: continue
        mins = minutes[r]; total = mins.sum()
        if total <= 0: continue
        players = pidx[r]; W = sum(v[0] for v in st.values())
        if W >= 150:   # warmed-up rotation
            played = dict(zip(players.tolist(), mins.tolist()))
            absent = [p for p, v in st.items() if v[0] >= 12 and played.get(p, 0.0) == 0.0]
            if absent:
                rat = {p: rating(p) for p in st}
                W_out = sum(st[p][0] for p in absent)
                present = [p for p in st if p not in absent]
                Wp = sum(st[p][0] for p in present); Qp = sum(st[p][0]*rat[p] for p in present)
                bench = [p for p in present if st[p][0] < 12]
                Wb = sum(st[p][0] for p in bench); Qb = sum(st[p][0]*rat[p] for p in bench)
                actual = float(sum(m*rating(p) for p, m in zip(players, mins))/total)
                rows.append(dict(season=season[i], actual=actual, none=sum(v[0]*rat[p] for p, v in st.items())/W,
                                 prop=Qp/Wp if Wp > 0 else np.nan, W=W, W_out=W_out, Wp=Wp, Qp=Qp,
                                 bench_rate=(Qb/Wb) if Wb > 5 else np.nan, league_rate=league_rate,
                                 out_rate=sum(st[p][0]*rat[p] for p in absent)/W_out))
        pending.append((team, players, mins, gs[r]))
df = pd.DataFrame(rows).dropna(subset=['prop'])
print(len(df), 'team-games with an absent rotation player (>=12 expected minutes)')
dev = df[df.season <= 2024]; fin = df[df.season > 2024]
def repl(d, r_rep): return (d.Qp + d.W_out*r_rep)/(d.Wp + d.W_out)
def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def bias(a, b): return float(np.mean(a-b))
# fit alpha for replacement at alpha*league_rate, and at alpha*bench_rate, on dev only
best = {}
for name, base in (('league', 'league_rate'), ('bench', 'bench_rate')):
    d = dev.dropna(subset=[base]); grid = np.linspace(0.5, 1.3, 33)
    errs = [rmse(d.actual, repl(d, a*d[base])) for a in grid]
    best[name] = float(grid[int(np.argmin(errs))])
# mixture: share s of the missing minutes goes to the present rotation proportionally, 1-s at replacement
print('alpha (dev): ', best)
for label, d in (('dev <=2024', dev), ('final >2024', fin)):
    print(f"\n{label}: n={len(d)}  mean absent-rating minus team-avg: {np.mean(d.out_rate - d.none):+.4f}")
    print(f"  no adjustment           rmse {rmse(d.actual, d.none):.5f} bias(actual-pred) {bias(d.actual, d.none):+.5f}")
    print(f"  proportional (current)  rmse {rmse(d.actual, d.prop):.5f} bias {bias(d.actual, d.prop):+.5f}")
    print(f"  replace @ {best['league']:.3f} x league rmse {rmse(d.actual, repl(d, best['league']*d.league_rate)):.5f} bias {bias(d.actual, repl(d, best['league']*d.league_rate)):+.5f}")
    dd = d.dropna(subset=['bench_rate'])
    print(f"  replace @ {best['bench']:.3f} x bench  rmse {rmse(dd.actual, repl(dd, best['bench']*dd.bench_rate)):.5f} bias {bias(dd.actual, repl(dd, best['bench']*dd.bench_rate)):+.5f} (n={len(dd)})")
# does the sign of the effect matter? check when the absent player is below team average
low = dev[dev.out_rate < dev.none]
print(f"\nabsent player rated BELOW team average (dev, n={len(low)}): actual-none {bias(low.actual, low.none):+.5f}; proportional predicts {np.mean(low.prop-low.none):+.5f}; replacement predicts {np.mean(repl(low, best['league']*low.league_rate)-low.none):+.5f}")
high = dev[dev.out_rate >= dev.none]
print(f"absent player rated ABOVE team average (dev, n={len(high)}): actual-none {bias(high.actual, high.none):+.5f}; proportional predicts {np.mean(high.prop-high.none):+.5f}; replacement predicts {np.mean(repl(high, best['league']*high.league_rate)-high.none):+.5f}")
