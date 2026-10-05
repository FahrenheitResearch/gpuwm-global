"""The arithmetic pin is per gravity-wave arithmetic and per tracer era.

Two [semi_implicit] schemes under two integrator families ship: the
vertical-mode operator and the barotropic proxy, each under the split-era
steppers (ssprk3 / rk4) and under the IMEX pair.  Four arithmetics, four
pins, and they must pin apart or a checkpoint of one could resume under
another.

The digests every archive carried BEFORE the grid tracers (2026-09-02:
the condensate species and number moments left the spectral basis) are
kept as the spectral-tracer era: a reader without a scheme still inspects
such an archive, and no restart resumes under it, because its condensate
transport is the defect the grid tracers retire.
"""
from __future__ import annotations

import pytest

from arwen_global import pins
from arwen_global.semi_implicit import SEMI_IMPLICIT_SCHEMES

#: arwen_global.pins.PINS_HASH as it stood on the
#: last tree with the single pin document, whose "semi_implicit" entry was
#: "barotropic-crank-nicolson-helmholtz-mode-v1".  Every checkpoint,
#: receipt and export of that era carries this digest.
PRE_VERTICAL_MODE_PINS_HASH = (
    "d5afac4ed5f20778197202390e1fb3a0c47d495b70a913175b861a7c23b842f1"
)
#: The vertical-mode split document's digest as first committed
#: carried by every vertical-mode split-era checkpoint
#: written before the two IMEX documents were split out.
VERTICAL_MODE_PINS_HASH = (
    "d05af5c097381e39c0e932992e6afed1722e2e94d508fe4c2ef634038045a0be"
)
#: The two IMEX documents as they stood when they were first recorded.
IMEX_VERTICAL_MODE_PINS_HASH_AS_RECORDED = (
    "26533ea4cd81faab809ef55374907de39ca313025fa92f6e1f2807d502982a6b"
)
IMEX_EXTERNAL_PINS_HASH_AS_RECORDED = (
    "15b5d7012e190e33fa49384cabd0ec57263cb905570bc5f963f6221ab4cd934d"
)
#: The four digests above are the SPECTRAL-TRACER ERA.  The current
#: documents' digests as first committed with the grid tracers; every
#: checkpoint written since carries one of these four.
GRID_TRACER_SPLIT_VERTICAL_MODE_PINS_HASH = (
    "4c4340945258b8c3e5648d0350af6e84ceb1e3d69d17c8b7d4603d361cb92aa6"
)
GRID_TRACER_SPLIT_EXTERNAL_PINS_HASH = (
    "f536e10061499732a08bd6e32cb45160820bb55519df5c0721be4c33fbf573a0"
)
GRID_TRACER_IMEX_VERTICAL_MODE_PINS_HASH = (
    "c5d0545d71c6b3fefd2aa5899e720161b0f0d38ff4522a50f66cdc44a52a992b"
)
GRID_TRACER_IMEX_EXTERNAL_PINS_HASH = (
    "6066104ed4e0004ea7e17a5e99fd014968ec97241a93dc112400fd074f909546"
)
#: The fifth arithmetic: the two-time-level semi-Lagrangian semi-implicit
#: core, which changes the gravity-wave composition, the momentum, the
#: scalar transport and the checkpoint, and pairs with the vertical-mode
#: operator alone (the barotropic proxy leaves every internal mode
#: explicit, which a 300 s step has no budget for, and the config door
#: refuses that pairing by name).  Moved twice, both on 2026-09-06: when
#: the tracer mass fixer gained the clip-deficit stage the scalar-transport
#: pin went to v2, and when theta stopped being advected as a deviation
#: from the reference profile (the reference's grid tendency warmed the lid
#: by 32 K a day where the trajectory is clamped, semilag.rhs) it went to
#: v3, this hash moving with it each time.  On 2026-10-05 two lines each
#: took the pin to a v4 of its own: the default mass fixer became the
#: published Bermejo-Conde weighting (this digest), and the departure level
#: was clamped at the boundary interfaces with the geometric level rate
#: (LID_V4_PINS_HASH).  Merged, the string carries both and reads v5; v3
#: (the WOOF Global 1.0.x arithmetic) and both v4 digests are retired and
#: inspectable.  The other four are untouched.
SEMILAG_VERTICAL_MODE_PINS_HASH = (
    "5344d698e0ebdcf62080871b37b55516691fb3b81f79a65c542857ab8f19c3f2"
)
LID_V4_PINS_HASH = (
    "9f2c83af1c759b5be69cf6b55936e2deb0a9097431f05ffa7aab88927b33f5de"
)
SEMILAG_V3_PINS_HASH = (
    "d82dc8ae4b0b5ea75aadccbf8b2ef5f8b3330d5b36670f00234f28bfe6e9d5d8"
)

#: The five CURRENT digests (DYC-1, DYC-3).  The four Eulerian documents
#: moved when the top and bottom layers took the one-sided gradient to
#: their neighbour instead of zero (the zero gradient carried the top
#: layer's own theta out under descent and cooled the lid), and the
#: semi-Lagrangian one when its departure level was clamped at the
#: boundary interfaces and its level rate took the geometric full-level
#: spacing, then again (v5) when that string and the Bermejo-Conde fixer's
#: were merged.  Every digest above that was current until then is
#: retired: inspectable, never resumable.
CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH = (
    "eb45fdb232e2af9e53b841a712114fdca59e34638a7da7b161f1ed3f65750fcc"
)
CURRENT_SPLIT_EXTERNAL_PINS_HASH = (
    "76b4f67b5e4270cdab3bff8482191308234772081cf86cef82de7f6e9186fc1a"
)
CURRENT_IMEX_VERTICAL_MODE_PINS_HASH = (
    "84a38ed0c4ca28255497b27fca5c068e64deefc7d570a5e9d20225184c148c98"
)
CURRENT_IMEX_EXTERNAL_PINS_HASH = (
    "37a75cb316831c821a7961ad6f9c746f1bd6556234ae6a7fca95de2bb38f68bf"
)
CURRENT_SEMILAG_VERTICAL_MODE_PINS_HASH = (
    "04527ce96c45834329cd131c8f5428f623ec0d0216ac085f10b2b15da3e0c23d"
)


def test_the_lid_clamp_era_pins_are_inspectable_and_never_current():
    retired = {
        GRID_TRACER_SPLIT_VERTICAL_MODE_PINS_HASH,
        GRID_TRACER_SPLIT_EXTERNAL_PINS_HASH,
        GRID_TRACER_IMEX_VERTICAL_MODE_PINS_HASH,
        GRID_TRACER_IMEX_EXTERNAL_PINS_HASH,
    }
    assert pins.RETIRED_BOUNDARY_FACE_PINS_HASHES == frozenset(retired)
    assert SEMILAG_VERTICAL_MODE_PINS_HASH in pins.RETIRED_SEMILAG_PINS_HASHES
    assert LID_V4_PINS_HASH in pins.RETIRED_SEMILAG_PINS_HASHES
    for digest in (*retired, SEMILAG_VERTICAL_MODE_PINS_HASH, LID_V4_PINS_HASH):
        assert digest in pins.INSPECTABLE_RETIRED_PINS_HASHES
        assert digest not in pins.KNOWN_PINS_HASHES
        assert pins.scheme_of_pins_hash(digest) is None


def test_the_v3_semilag_pin_is_retired_inspectable_and_never_current():
    assert SEMILAG_V3_PINS_HASH in pins.RETIRED_SEMILAG_PINS_HASHES
    assert SEMILAG_V3_PINS_HASH in pins.INSPECTABLE_RETIRED_PINS_HASHES
    assert SEMILAG_V3_PINS_HASH not in pins.KNOWN_PINS_HASHES
    assert pins.scheme_of_pins_hash(SEMILAG_V3_PINS_HASH) is None


def test_the_external_scheme_keeps_its_proxy_pin_string_under_the_grid_tracers():
    assert pins.pins_hash("external", "ssprk3") == CURRENT_SPLIT_EXTERNAL_PINS_HASH
    assert pins.PINS_HASH_BY_SCHEME["external"] == CURRENT_SPLIT_EXTERNAL_PINS_HASH
    assert (
        pins.pin_document("external", "ssprk3")["semi_implicit"]
        == "barotropic-crank-nicolson-helmholtz-mode-v1"
    )
    assert pins.pins_hash("external") == CURRENT_IMEX_EXTERNAL_PINS_HASH
    assert pins.pins_hash("external") != CURRENT_SPLIT_EXTERNAL_PINS_HASH


def test_the_spectral_tracer_era_pins_are_inspectable_and_never_current():
    assert pins.SPECTRAL_TRACER_ERA_PINS_HASHES == frozenset({
        PRE_VERTICAL_MODE_PINS_HASH,
        VERTICAL_MODE_PINS_HASH,
        IMEX_VERTICAL_MODE_PINS_HASH_AS_RECORDED,
        IMEX_EXTERNAL_PINS_HASH_AS_RECORDED,
    })
    assert not (pins.SPECTRAL_TRACER_ERA_PINS_HASHES & pins.KNOWN_PINS_HASHES)
    for digest in pins.SPECTRAL_TRACER_ERA_PINS_HASHES:
        assert pins.scheme_of_pins_hash(digest) is None
    document = pins.pin_document()
    assert document["representation"]["grid"] == [
        "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
    ]
    assert document["representation"]["spectral"][-1] == "qv"


def test_the_vertical_mode_scheme_is_the_default_pin_and_its_committed_literal():
    assert pins.DEFAULT_SEMI_IMPLICIT_SCHEME == "vertical_modes"
    # The split-era vertical-mode pin under the split steppers; the
    # default arithmetic is the IMEX pair.
    assert pins.pins_hash("vertical_modes", "ssprk3") == CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH
    assert pins.pins_hash("vertical_modes", "rk4") == CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH
    assert pins.PINS_HASH_BY_SCHEME["vertical_modes"] == CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH
    assert pins.DEFAULT_INTEGRATOR == "imex_ssp3"
    assert pins.pins_hash() == pins.PINS_HASH == pins.pins_hash("vertical_modes", "imex_ssp3")
    assert pins.PINS_HASH == CURRENT_IMEX_VERTICAL_MODE_PINS_HASH
    assert pins.PINS_HASH != CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH
    assert pins.PIN_DOCUMENT == pins.pin_document("vertical_modes")
    assert (
        pins.PIN_DOCUMENT["semi_implicit"]
        == pins.IMEX_SEMI_IMPLICIT_PINS["vertical_modes"]
    )
    assert (
        pins.pin_document("vertical_modes", "ssprk3")["semi_implicit"]
        == pins.SEMI_IMPLICIT_PINS["vertical_modes"]
    )


def test_the_two_schemes_pin_apart_and_nothing_else_differs():
    external = pins.pin_document("external", "ssprk3")
    modes = pins.pin_document("vertical_modes", "ssprk3")
    assert pins.pins_hash("external") != pins.pins_hash("vertical_modes")
    assert pins.pins_hash("external", "ssprk3") != pins.pins_hash("vertical_modes", "ssprk3")
    assert set(external) == set(modes)
    differing = {key for key in external if external[key] != modes[key]}
    assert differing == {"semi_implicit"}
    # Two schemes under the split-era steppers, two under the IMEX
    # integrator, and one under the semi-Lagrangian core, which pairs
    # with the vertical-mode operator alone: five arithmetics, five pins.
    assert len(pins.KNOWN_PINS_HASHES) == 5
    assert pins.KNOWN_PINS_HASHES == frozenset({
        CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH,
        CURRENT_SPLIT_EXTERNAL_PINS_HASH,
        CURRENT_IMEX_VERTICAL_MODE_PINS_HASH,
        CURRENT_IMEX_EXTERNAL_PINS_HASH,
        CURRENT_SEMILAG_VERTICAL_MODE_PINS_HASH,
    })
    assert set(pins.PINS_HASH_BY_SCHEME.values()) < pins.KNOWN_PINS_HASHES
    assert pins.scheme_of_pins_hash(pins.pins_hash("external", "ssprk3")) == "external"
    assert pins.scheme_of_pins_hash(CURRENT_SPLIT_VERTICAL_MODE_PINS_HASH) == "vertical_modes"
    assert pins.scheme_of_pins_hash(pins.PINS_HASH) == "vertical_modes/imex_ssp3"
    assert pins.scheme_of_pins_hash("f" * 64) is None
    assert pins.scheme_of_pins_hash(None) is None


def test_every_config_selectable_scheme_carries_a_pin():
    assert set(pins.SEMI_IMPLICIT_PINS) == set(SEMI_IMPLICIT_SCHEMES)
    with pytest.raises(ValueError, match="unknown semi-implicit scheme 'proxy'"):
        pins.pins_hash("proxy")


def test_pins_receipt_carries_the_scheme_document_and_its_digest():
    receipt = pins.pins_receipt("external", "ssprk3")
    assert receipt["sha256"] == CURRENT_SPLIT_EXTERNAL_PINS_HASH
    assert receipt["pins"] == pins.pin_document("external", "ssprk3")
    default = pins.pins_receipt()
    assert default["sha256"] == pins.PINS_HASH
    assert default["pins"] == pins.PIN_DOCUMENT


def test_pin_documents_are_fresh_copies():
    document = pins.pin_document()
    document["native_physics"]["order"].append("tampered")
    document["semi_implicit"] = "tampered"
    assert pins.PIN_DOCUMENT["semi_implicit"] != "tampered"
    assert "tampered" not in pins.PIN_DOCUMENT["native_physics"]["order"]
    assert pins.pins_hash() == pins.PINS_HASH


def test_a_receipt_records_the_libraries_its_numbers_rode_on():
    """A receipt that cannot say which numpy wrote it cannot answer the first
    question two differing runs raise.

    THE BREAKAGE THIS NAMES, measured in this repository rather than imagined:
    `numpy.polynomial.legendre.leggauss` is a LAPACK eigensolve and its bits
    moved between numpy 2.2.6 and 2.3.0, so a spectral table's recorded digest
    reproduces under one and not the other.  The receipt recorded the
    configuration, the arithmetic pins, the physics identity and the machine,
    and nothing at all about the libraries, and that answer is not recoverable
    afterwards: by the time anybody asks, the environment has moved.

    The block is beside `pins`, never inside it.  `pins_hash` is the identity
    of the ARITHMETIC and two runs under different numpys share it on purpose;
    the libraries are what says why they might still differ.
    """

    import sys
    from importlib.metadata import version

    from arwen_global.receipt import finalize_receipt, library_versions

    libraries = library_versions()
    assert libraries["python"] == ".".join(str(n) for n in sys.version_info[:3])
    assert libraries["numpy"] == version("numpy")
    assert libraries["gpuwm"] == version("gpuwm")
    assert libraries["gpuwm-global"] == version("gpuwm-global")
    # A distribution that is not installed is absent rather than null: a key
    # whose value is None reads as "asked and got nothing", which is a
    # different claim from "not installed".
    assert all(value for value in libraries.values())

    stamped = finalize_receipt({"name": "x", "status": "pass"})
    assert stamped["libraries"] == libraries
    # The self-hash covers it, so a receipt cannot be edited to claim a
    # different environment and still validate.
    import hashlib

    from arwen_global.receipt import canonical

    body = {k: v for k, v in stamped.items() if k != "self_sha256"}
    assert stamped["self_sha256"] == hashlib.sha256(canonical(body)).hexdigest()

    # A caller that has already recorded them keeps its own.
    mine = {"python": "0.0.0"}
    assert finalize_receipt({"name": "x", "libraries": mine})["libraries"] == mine
