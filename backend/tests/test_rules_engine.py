from app.rules_engine import Detection, evaluate, point_in_polygon

PERSON = Detection("person", 0.9, (100, 100, 200, 400))


def test_helmet_present_no_violation():
    helmet = Detection("helmet", 0.8, (130, 105, 170, 140))
    vest = Detection("vest", 0.8, (110, 180, 190, 280))
    assert evaluate([PERSON, helmet, vest]) == []


def test_missing_helmet_flagged():
    vest = Detection("vest", 0.8, (110, 180, 190, 280))
    v = evaluate([PERSON, vest])
    assert [x.category for x in v] == ["no_helmet"]
    assert v[0].certainty == "likely"


def test_small_person_is_uncertain():
    tiny = Detection("person", 0.9, (10, 10, 40, 70))
    assert all(x.certainty == "uncertain" for x in evaluate([tiny]))


def test_unsupported_class_never_flagged():
    assert evaluate([PERSON], supports=()) == []


def test_restricted_zone():
    zone = [(0, 300), (300, 300), (300, 500), (0, 500)]
    cats = [x.category for x in evaluate([PERSON], [zone], supports=())]
    assert cats == ["restricted_zone"]


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)
