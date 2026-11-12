"""Using evmodel as a library.

Run from the repository root:  python examples/quickstart.py
Everything here works offline; the slate is simulated (fictional players and
prices), so the output illustrates the workflow, not real bets.
"""

from evmodel import Config, devig_two_way, expected_value, kelly_fraction, parse_odds, scan
from evmodel.distributions import FamilySpec
from evmodel.providers.sample import make_slate

# 1. A single price check: a sharp book has Over 24.5 at -125 / +105,
#    another book still offers the over at -105.
p_over, p_under = devig_two_way(parse_odds("-125"), parse_odds("+105"), method="shin")
price = parse_odds("-105")
print(f"fair P(over) = {p_over:.1%}")
print(f"EV at -105   = {expected_value(p_over, price):+.2%} per unit")
print(f"full Kelly   = {kelly_fraction(p_over, price):.2%} of bankroll")

# 2. The same fair price moved to a different line: what is Over 23.5 worth?
spec = FamilySpec("negbin", phi=2.16, c=16.3, scale=3.39)  # the fitted NBA points shape
over, push, under = spec.translate(from_line=24.5, p_over=p_over, to_line=23.5)
print(f"implied P(over 23.5) = {over:.1%}")

# 3. A whole slate.
world = make_slate("nba", n_games=6, seed=7)
config = Config()
config.scan.min_ev = 0.03
picks = scan(world.quotes, config)
print(f"\n{len(picks)} bets from {len(world.quotes)} simulated quotes")
print(picks[["player", "market", "line", "side", "book", "price_american", "ev", "stake_units"]].head(10).to_string(index=False))
