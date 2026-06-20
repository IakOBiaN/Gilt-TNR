"""Unit tests for the small pure helper functions in the algorithm modules."""
import GiltTNR2D
import GiltTNR3D


# - update_pars (identical in both modules) -

def test_update_pars_returns_same_object_without_kwargs():
    pars = {"a": 1, "b": 2}
    assert GiltTNR2D.update_pars(pars) is pars
    assert GiltTNR3D.update_pars(pars) is pars


def test_update_pars_copies_and_overrides():
    pars = {"a": 1, "b": 2}
    new = GiltTNR2D.update_pars(pars, b=3, c=4)
    assert new == {"a": 1, "b": 3, "c": 4}
    # The original must not be mutated.
    assert pars == {"a": 1, "b": 2}
    assert new is not pars


# - permutation helpers (GiltTNR3D) -

def test_invert_permutation_known_value():
    assert GiltTNR3D.invert_permutation((2, 0, 1)) == [1, 2, 0]


def test_invert_permutation_is_inverse():
    p = (3, 0, 2, 1, 4)
    inv = tuple(GiltTNR3D.invert_permutation(p))
    identity = tuple(range(len(p)))
    assert GiltTNR3D.combine_permutations(p, inv) == identity
    assert GiltTNR3D.combine_permutations(inv, p) == identity


def test_combine_permutations_with_identity():
    identity = (0, 1, 2, 3)
    p = (2, 3, 0, 1)
    assert GiltTNR3D.combine_permutations(p, identity) == p
    assert GiltTNR3D.combine_permutations(identity, p) == p


def test_combine_permutations_composition():
    # By construction result[i] == p2[p1[i]].
    p1 = (0, 2, 1)
    p2 = (1, 0, 2)
    expected = tuple(p2[p1[i]] for i in range(len(p1)))
    assert GiltTNR3D.combine_permutations(p1, p2) == expected
    assert expected == (1, 2, 0)
