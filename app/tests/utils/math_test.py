from math import pi, sqrt


from app.utils.math import Vec2, clip, normalise_angle, raytrace


class TestVec2:
    def test_init(self):
        v = Vec2(1, 2)
        assert v.x == 1
        assert v.y == 2

    def test_add(self):
        assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)

    def test_subtract(self):
        assert Vec2(1, 2) - Vec2(3, 4) == Vec2(-2, -2)

    def test_multiply(self):
        assert Vec2(1, 2) * 2 == Vec2(2, 4)
        assert Vec2(1, 2) * Vec2(3, 4) == 11

    def test_angle(self):
        assert Vec2(1, 0).angle() == 0
        assert Vec2(0, 1).angle() == pi / 2
        assert Vec2(-1, 0).angle() == pi
        assert Vec2(0, -1).angle() == -pi / 2

    def test_clip(self):
        assert Vec2(2, 0).clip(1) == Vec2(1, 0)
        assert Vec2(1, 0).clip(1) == Vec2(1, 0)
        assert Vec2(0.5, 0).clip(1) == Vec2(0.5, 0)
        assert Vec2(0, 2).clip(1) == Vec2(0, 1)

        assert Vec2(2, 3).clip(1) == Vec2(1, 1)
        assert Vec2(-3, 3).clip(2) == Vec2(-2, 2)
        assert Vec2(-4, 4).clip(-3, 2) == Vec2(-3, 2)

    def test_limit_mag(self):
        assert Vec2(2, 0).limit(1) == Vec2(1, 0)
        assert Vec2(1, 0).limit(1) == Vec2(1, 0)
        assert Vec2(0.5, 0).limit(1) == Vec2(0.5, 0)
        assert Vec2(0, 2).limit(1) == Vec2(0, 1)

        assert Vec2(2, 2).limit(1) == Vec2(sqrt(2) / 2, sqrt(2) / 2)

    def test_mag(self):
        assert Vec2(3, 4).abs() == 5
        assert Vec2(1, 1).abs() == sqrt(2)

    def test_mag2(self):
        assert Vec2(3, 4).mag2() == 25
        assert Vec2(1, 1).mag2() == 2

    def test_rotate(self):
        assert Vec2(1, 0).rotate(pi / 2) == Vec2(0, 1)
        assert Vec2(0, 1).rotate(pi / 2) == Vec2(-1, 0)
        assert Vec2(-1, 0).rotate(pi / 2) == Vec2(0, -1)
        assert Vec2(0, -1).rotate(pi / 2) == Vec2(1, 0)
        assert Vec2(1, 1).rotate(pi / 2) == Vec2(-1, 1)

    def test_set_mag(self):
        assert Vec2(1, 0).set_mag(1) == Vec2(1, 0)
        assert Vec2(1, 0).set_mag(2) == Vec2(2, 0)
        assert Vec2(1, 1).set_mag(1) == Vec2(sqrt(2) / 2, sqrt(2) / 2)
        assert Vec2(1, 1).set_mag(2) == Vec2(sqrt(2), sqrt(2))
        assert Vec2(1, 1).set_mag(0) == Vec2(0, 0)
        assert Vec2(0, 0).set_mag(1) == Vec2(0, 0)
        assert Vec2(0, 0).set_mag(0) == Vec2(0, 0)


def test_clip():
    assert clip(0.5, 0, 1) == 0.5
    assert clip(1.5, 0, 1) == 1
    assert clip(0.5, 1, 2) == 1
    assert clip(1.5, 1, 2) == 1.5
    assert clip(2.5, 1, 2) == 2
    assert clip(0, 1, 2) == 1
    assert clip(1, 1, 2) == 1
    assert clip(2, 1, 2) == 2
    assert clip(3, 1, 2) == 2


def test_normalise_angle():
    assert normalise_angle(0) == 0
    assert normalise_angle(pi) == pi
    assert normalise_angle(-pi) == pi
    assert normalise_angle(2 * pi) == 0
    assert normalise_angle(-2 * pi) == 0
    assert normalise_angle(3 * pi) == pi
    assert normalise_angle(-3 * pi) == pi


def test_raytrace():
    assert list(raytrace((0, 0), (3, 0))) == [(0, 0), (1, 0), (2, 0), (3, 0)]
