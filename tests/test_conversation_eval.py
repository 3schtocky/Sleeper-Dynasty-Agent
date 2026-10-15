from dynasty_agent import chat, conversation_eval as ce, tools


def turn(reply="", tool=None, clarification=False, compact=None, arguments=None, result=None):
    return chat.Turn("q", tool=tool, arguments=arguments or {}, compact=compact or {}, reply=reply,
                     clarification=clarification, result=result)


def run(check, t):
    return check(t, "", None)


def test_tool_and_clarification_checks():
    assert run(ce.tool_is("my_team"), turn(tool="my_team")) is None
    assert "wanted my_team" in run(ce.tool_is("my_team"), turn(tool="set_lineup"))
    assert run(ce.asks_back("Coleman"), turn("Coleman is on DannyBoss", clarification=True)) is None
    assert run(ce.asks_back("Coleman"), turn("Make the trade")) == "gave an answer instead of asking"


def test_redirect_checks():
    assert run(ce.redirects(), turn(chat.OFF_TOPIC_REPLY)) is None
    assert run(ce.redirects(), turn("Paris")) == "did not redirect"
    assert run(ce.does_not_redirect(), turn(chat.OFF_TOPIC_REPLY))


def test_plain_and_short_flags_markdown_lists_emoji_and_walls():
    assert run(ce.plain_and_short(), turn("Sell him. It is close.")) is None
    assert run(ce.plain_and_short(), turn("**Sell** him."))
    assert run(ce.plain_and_short(), turn("Sell him 😊."))
    assert run(ce.plain_and_short(2), turn("One. Two. Three."))
    assert run(ce.plain_and_short(), turn(""))


def test_grounded_catches_a_stat_in_talk_and_a_foreign_number_in_a_take():
    assert run(ce.grounded(), turn("Backs fade after age 27.")) is None
    assert "20" in run(ce.grounded(), turn("He scores 20 points a game."))
    res = tools.ToolResult("my_team", numbers="Win now 55th", compact={"x": "55th"})
    assert run(ce.grounded(), turn("You are 55th.", tool="my_team", result=res)) is None
    assert "17" in run(ce.grounded(), turn("That is 17 better.", tool="my_team", result=res))


def test_agrees_with_facts_catches_a_flipped_call():
    sell = {"facts": ["Your 2027 3.05 pick is a SELL (worth trading away now): x"]}
    assert run(ce.agrees_with_facts(), turn("Hold that pick.", compact=sell))
    assert run(ce.agrees_with_facts(), turn("Sell that pick.", compact=sell)) is None
    other = {"facts": ["The numbers favor the other team on both trade value and win-now points."]}
    assert run(ce.agrees_with_facts(), turn("This is a good deal.", compact=other))


def test_every_scenario_step_has_a_question_and_checks():
    assert ce.SCENARIOS
    for sc in ce.SCENARIOS:
        assert sc.steps and all(s.question and s.checks for s in sc.steps), sc.name
