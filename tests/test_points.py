from dynasty_agent import points


def trade(value, win_now, **extra):
    return {"net_market_value": value, "net_win_now": win_now, **extra}


def test_trade_lean_follows_the_signs():
    assert "favor you on both" in points.build("evaluate_trade", trade("+680", "+6.0"), None)[0]
    assert "favor the other team" in points.build("evaluate_trade", trade("-300", "-1.0"), None)[0]
    assert "mixed: you win on trade value" in points.build("evaluate_trade", trade("+300", "-1.0"), None)[0]


def test_a_trade_warning_and_a_consolidation_note_come_through_and_the_headline_leads():
    pts = points.build("evaluate_trade", trade("+680", "+6.0", headline="X is a free agent", consolidation="deconsolidation: bad"), None)
    assert pts[0] == "X is a free agent" and any("deconsolidation: bad" in p for p in pts)


def test_pick_sell_is_spelled_out_so_the_model_cannot_flip_it():
    rows = [{"season": 2027, "projected_slot": "3.05", "ratio": 3.0, "advice": "SELL", "fantasycalc_price": 1039.0, "comparable_value": 345.0, "round": 3}]
    pts = points.build("pick_advice", {"next_draft": "2027", "posture": "unclear"}, {"rows": rows})
    assert any("SELL (worth trading away now)" in p and "1039" in p and "345" in p for p in pts)


def test_lineup_top_starter_is_named_without_its_number():
    c = {"warnings": [], "win_probability": "48.8%", "opponent_projected": "124.3", "starters": ["Matthew Stafford QB 25.3 (Questionable)"]}
    assert "Your top projected starter is Matthew Stafford." in points.build("set_lineup", c, None)


def test_waiver_list_leaves_the_budget_out_and_names_the_top_pickup():
    c = {"headline": "best pickup: Jake Tonges TE, bid $33", "targets": ["Jake Tonges TE: +0.1 to lineup, bid $33"], "remaining_budget": "$100"}
    pts = points.build("waiver_targets", c, None)
    assert pts[0].startswith("Best pickup: Jake Tonges") and not any("$100" in p for p in pts)
