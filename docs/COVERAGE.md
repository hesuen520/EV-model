# Coverage

Generated from the registry and this installation's store. A status is earned by ingested data and recorded validation runs; it is never set by hand.

| Status | Meaning |
|---|---|
| validated | Real data ingested; chronological walk-forward validation was run with a final test period that played no part in model selection; the selected model passed every configured gate (sample size, skill over the naive baseline with its uncertainty interval, calibration, point-in-time features). |
| experimental | Real data ingested and the model runs end to end with a recorded validation run, but at least one gate is not met (small sample, weak or uncertain skill, calibration, retrospective inputs, or no realistic line history). |
| demo-only | Runs only on synthetic demo data, which is isolated and labelled. No claim about real events. |
| adapter-only | An adapter and/or documented CSV import and a model specification exist, but no real data has been ingested and validated in this installation. |
| unavailable | Not implemented. |

| Sport | League | Market | Status | Basis | Final-test n | Accuracy | Log loss | Results through | Data source |
|---|---|---|---|---|---|---|---|---|---|
| baseball | Major League Baseball (`mlb`) | winner | **validated** | all validation gates passed | 4949 | 56.5% | 0.6798 | 2025-11-01 | Retrosheet game logs and day-by-day files (Chadwick Bureau GitHub mirror) |
| baseball | Major League Baseball (`mlb`) | total | **unavailable** | no model implemented for this market |  |  |  | 2025-11-01 | Retrosheet game logs and day-by-day files (Chadwick Bureau GitHub mirror) |
| basketball | NBA (`nba`) | winner | **validated** | all validation gates passed | 2813 | 67.7% | 0.5999 | 2026-11-12 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NBA (`nba`) | spread | **unavailable** | no model implemented for this market |  |  |  | 2026-11-12 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NBA (`nba`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-11-12 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NBA (`nba`) | player_prop | **unavailable** | no model implemented for this market |  |  |  | 2026-11-12 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | WNBA (`wnba`) | winner | **experimental** | gates not met: calibration | 664 | 65.4% | 0.6261 | 2026-10-29 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | WNBA (`wnba`) | spread | **unavailable** | no model implemented for this market |  |  |  | 2026-10-29 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | WNBA (`wnba`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-10-29 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | WNBA (`wnba`) | player_prop | **unavailable** | no model implemented for this market |  |  |  | 2026-10-29 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I men (`ncaam`) | winner | **validated** | all validation gates passed | 13251 | 75.0% | 0.4876 | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I men (`ncaam`) | spread | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I men (`ncaam`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I women (`ncaaw`) | winner | **validated** | all validation gates passed | 12313 | 78.3% | 0.4386 | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I women (`ncaaw`) | spread | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| basketball | NCAA Division I women (`ncaaw`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data release files (ESPN-derived schedules and box scores) |
| demo | SYNTHETIC demo league (home advantage) (`demo_basketball`) | winner | **demo-only** | synthetic demo league |  |  |  |  | CSV import only |
| demo | SYNTHETIC demo league (no home side) (`demo_duel`) | winner | **demo-only** | synthetic demo league |  |  |  |  | CSV import only |
| football | NFL (`nfl`) | winner | **validated** | all validation gates passed | 705 | 65.4% | 0.6266 | 2026-11-13 | nflverse-data release files (schedule, results, team and quarterback statistics) |
| football | NFL (`nfl`) | spread | **experimental** | gates not met: skill_over_baseline | 696 | 50.3% | 0.6935 | 2026-11-13 | nflverse-data release files (schedule, results, team and quarterback statistics) |
| football | NFL (`nfl`) | total | **experimental** | gates not met: skill_over_baseline, calibration | 704 | 47.3% | 0.6935 | 2026-11-13 | nflverse-data release files (schedule, results, team and quarterback statistics) |
| football | NCAA Division I FBS (`ncaaf`) | winner | **validated** | all validation gates passed | 2168 | 70.8% | 0.5464 | 2026-11-13 | sportsdataverse-data cfb_schedules release files (schedules and results) |
| football | NCAA Division I FBS (`ncaaf`) | spread | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data cfb_schedules release files (schedules and results) |
| football | NCAA Division I FBS (`ncaaf`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data cfb_schedules release files (schedules and results) |
| cricket | Men's T20 cricket (`cricket_t20_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| cricket | Women's T20 cricket (`cricket_t20_women`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| cricket | Men's one-day cricket (`cricket_odi_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| cricket | Men's Test cricket (`cricket_test_men`) | winner_3way | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| rugby | Men's rugby union (`rugby_union_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| rugby | Women's rugby union (`rugby_union_women`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| rugby | Men's rugby league (`rugby_league_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| volleyball | Men's volleyball (`volleyball_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| volleyball | Women's volleyball (`volleyball_women`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| badminton | Badminton men's singles (`badminton_ms`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| badminton | Badminton women's singles (`badminton_ws`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| badminton | Badminton men's doubles (`badminton_md`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| badminton | Badminton women's doubles (`badminton_wd`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| badminton | Badminton mixed doubles (`badminton_xd`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| table_tennis | Table tennis men's singles (`tabletennis_ms`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| table_tennis | Table tennis women's singles (`tabletennis_ws`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| tennis | ATP Tour doubles (`atp_doubles`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| tennis | WTA Tour doubles (`wta_doubles`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| mma | UFC (`ufc`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| boxing | Men's professional boxing (`boxing_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| boxing | Women's professional boxing (`boxing_women`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| esports | Counter-Strike 2 (`cs2`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| esports | Valorant (`valorant`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| esports | League of Legends (`lol`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| esports | Dota 2 (`dota2`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| basketball | International men's basketball (`basketball_intl_men`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| basketball | International women's basketball (`basketball_intl_women`) | winner | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| golf | PGA Tour (`pga_tour`) | win | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| golf | PGA Tour (`pga_tour`) | top3 | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | Formula 1 (`f1`) | win | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | Formula 1 (`f1`) | top3 | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | NASCAR Cup Series (`nascar_cup`) | win | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | NASCAR Cup Series (`nascar_cup`) | top3 | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | IndyCar (`indycar`) | win | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | IndyCar (`indycar`) | top3 | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | MotoGP (`motogp`) | win | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| motorsport | MotoGP (`motogp`) | top3 | **unavailable** | no model implemented for this market |  |  |  |  | CSV import only |
| hockey | NHL (`nhl`) | winner | **validated** | all validation gates passed | 3090 | 57.0% | 0.6719 | 2026-11-13 | sportsdataverse-data release files (NHL schedules, goalie box scores, play-by-play shots) |
| hockey | NHL (`nhl`) | winner_3way | **validated** | all validation gates passed | 3090 | 45.0% | 1.0499 | 2026-11-13 | sportsdataverse-data release files (NHL schedules, goalie box scores, play-by-play shots) |
| hockey | NHL (`nhl`) | total | **unavailable** | no model implemented for this market |  |  |  | 2026-11-13 | sportsdataverse-data release files (NHL schedules, goalie box scores, play-by-play shots) |
| soccer | Men's international football (`soccer_intl_men`) | winner_3way | **experimental** | gates not met: calibration | 1803 | 61.6% | 0.8430 | 2026-10-06 | martj42/international_results results.csv + shootouts.csv (GitHub raw) |
| soccer | Women's international football (`soccer_intl_women`) | winner_3way | **validated** | all validation gates passed | 1350 | 67.3% | 0.7719 | 2026-10-13 | martj42/womens-international-results results.csv + shootouts.csv (GitHub raw) |
| soccer | English Premier League (`epl`) | winner_3way | **experimental** | gates not met: calibration | 810 | 49.6% | 1.0151 | 2026-09-20 | xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + openfootball/football.json current-season fixtures (GitHub raw) |
| soccer | Spanish La Liga (`laliga`) | winner_3way | **experimental** | gates not met: calibration | 829 | 52.6% | 0.9793 | 2026-09-20 | xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + openfootball/football.json current-season fixtures (GitHub raw) |
| soccer | German Bundesliga (`bundesliga`) | winner_3way | **experimental** | gates not met: calibration | 648 | 51.7% | 1.0001 | 2026-09-20 | xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + openfootball/football.json current-season fixtures (GitHub raw) |
| soccer | Italian Serie A (`seriea`) | winner_3way | **experimental** | gates not met: calibration | 810 | 52.6% | 0.9805 | 2026-09-20 | xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + openfootball/football.json current-season fixtures (GitHub raw) |
| soccer | French Ligue 1 (`ligue1`) | winner_3way | **validated** | all validation gates passed | 656 | 52.4% | 0.9924 | 2026-09-20 | xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + openfootball/football.json current-season fixtures (GitHub raw) |
| soccer | Major League Soccer (`mls`) | winner_3way | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| soccer | National Women's Soccer League (`nwsl`) | winner_3way | **adapter-only** | no real data ingested in this installation |  |  |  |  | CSV import only |
| tennis | ATP Tour main draw singles (`atp`) | winner | **validated** | all validation gates passed | 3898 | 66.0% | 0.6099 | 2026-05-25 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | ATP Tour qualifying singles (`atp_qual`) | winner | **validated** | all validation gates passed | 1799 | 65.0% | 0.6176 | 2026-05-17 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | ATP Challenger main draw singles (`atp_challenger`) | winner | **validated** | all validation gates passed | 9950 | 63.5% | 0.6382 | 2026-06-01 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | ITF men's singles (`itf_men`) | winner | **experimental** | gates not met: development_folds | 43092 | 71.1% | 0.5613 | 2026-06-01 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | WTA Tour main draw singles (`wta`) | winner | **validated** | all validation gates passed | 3673 | 66.7% | 0.6036 | 2026-05-25 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | WTA Tour qualifying singles (`wta_qual`) | winner | **validated** | all validation gates passed | 3278 | 67.5% | 0.5904 | 2026-05-18 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | WTA 125 main draw singles (`wta_125`) | winner | **validated** | all validation gates passed | 2341 | 65.6% | 0.6114 | 2026-06-02 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
| tennis | ITF women's singles (`itf_women`) | winner | **validated** | all validation gates passed | 25094 | 70.0% | 0.5700 | 2026-06-01 | mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026) |
