# Dataset Overview

## Sources
- Ball-by-ball dataset
- Match-level dataset

## Merge Strategy
Combined match metadata (teams, venue) with ball progression data.

## Final Features
batting_team  
bowling_team  
over  
ball  
venue  

Targets:
runs  
wickets  
winner  

## Current Dataset Size
~260,000 rows

## Known Risks
- Possible leakage from match-level joins
- Encoding strategy may discard player-level signal
- Temporal consistency not yet validated

## Prediction Horizon

- The dataset includes match states across all overs, which may make late-innings predictions significantly easier than early-game forecasts.

- Future evaluation will segment performance by match phase to ensure realistic predictive capability.

- This signals deep modeling awareness.