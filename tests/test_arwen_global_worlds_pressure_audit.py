"""The thin-air kernel audit page is held to the kernels it describes.

`docs/arwen-global-worlds-pressure-audit.md` lists every place the two
convection schemes this package carries (Grell-Freitas in
`core/kernels/gf.cu`, New Tiedtke in `core/kernels/ntiedtke.cu`), its own
mass-flux scheme (`physics/arwen_massflux.py`) and the Morrison
microphysics (`core/kernels/morrison.cu`) read an absolute pressure, and
says for each what it decides and what it would have to scale with on a
world whose air is thinner than Earth's.  The page decides whether thin
air can ever leave "not supported", so a row that no longer matches the
source is a wrong answer to that question, not a stale sentence.

Three things are checked, all by reading text, with no device and no run:

1. every row's source line is still in the file the row names, exactly as
   many times as the page says (a re-cut of the carried kernels that moves
   a threshold fails here and points at the row);
2. no NEW absolute-pressure literal has arrived in the New Tiedtke kernel
   and no NEW use of a Grell-Freitas pressure constant, so the inventory
   stays complete and not merely correct;
3. the page's share column is the arithmetic it claims, the threshold over
   a 1013.25 hPa column and over a 500 hPa column, to one decimal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import arwen_global

PACKAGE = Path(arwen_global.__file__).resolve().parent
KERNELS = PACKAGE / "core" / "kernels"
PAGE = Path(__file__).resolve().parents[1] / "docs" / "arwen-global-worlds-pressure-audit.md"

#: Each audited row: its id on the page, the file, and the source lines that
#: carry it with how often each occurs in that file.
ROWS: dict[str, tuple[Path, tuple[tuple[str, int], ...]]] = {
    # Grell-Freitas, deep and shallow (cumulus = "gf", every shipped config).
    "GF-1": (KERNELS / "gf.cu", (
        ("0x42960000u,  // [ 46] CAP_MAXS = cap_maxs = 75.", 1),
        ("float cap_maxs = K_CAP_MAXS;", 1),
        ("if (pbcdif <= plus) break;", 1),
    )),
    "GF-2": (KERNELS / "gf.cu", (
        ("if (ztexec > K_ZERO) cap_max = FADD(cap_max, K_TWENTYFIVE);", 1),
        ("if (ztexec < K_ZERO) cap_max = FSUB(cap_max, K_TWENTYFIVE);", 1),
    )),
    "GF-3": (KERNELS / "gf.cu", (
        ("0x41A00000u,  // [ 60] CAP_INC_DEEP = cap_max_increment = 20.", 1),
        ("gfd_cup_kbcon(cap_max_increment, 2, &k22x2,", 1),
        ("gfd_cup_kbcon(cap_max_increment, 3, &k22x_final,", 1),
        ("if (ierr2 > 0 || ierr3 > 0) {", 1),
        ("float ens_adj = K_ONE;", 1),
        ("ens_adj", 4),
    )),
    "GF-4": (KERNELS / "gf.cu", (
        ("float plus = GMAX(K_TWENTYFIVE,", 1),
        ("FSUB(cap_max, FMUL((float)(iloop - 1), cap_inc)));", 1),
    )),
    "GF-5": (KERNELS / "gf.cu", (
        ("plus = K_P150;", 1),
        ("0x43160000u,  // [ 61] P150 = 150.", 1),
    )),
    "GF-6": (KERNELS / "gf.cu", (
        ("if (cap_max > K_P200) pbcdif = FADD(-p_cup[kbcon], cap_max);", 1),
        ("if (kpbl > 3) cap_max = po_cup[kpbl];", 1),
    )),
    "GF-7": (KERNELS / "gf.cu", (
        ("0x42FA0000u,  // [125] CAP_MAXS_SH = cap_maxs = 125. (shallow)", 1),
        ("float cap_max_increment = K_TWENTYFIVE;", 1),
    )),
    "GF-8": (KERNELS / "gf.cu", (
        ("&& FSUB(po_cup[kbcon], po_cup[k_inv[1]]) < K_P200) {", 1),
        ("if (FSUB(po_cup[kbcon], po_cup[k]) > K_P200) {", 1),
        ("0x43480000u,  // [ 62] P200 = 200.", 1),
    )),
    "GF-9": (KERNELS / "gf.cu", (
        ("sd[kx] = FSUB(fabsf(dp), K_P100);", 1),
        ("0x42C80000u,  // [117] P100 = 100. (800 hPa slot)", 1),
    )),
    "GF-10": (KERNELS / "gf.cu", (
        ("sd[kx] = FSUB(fabsf(dp), K_P300);", 1),
        ("0x43960000u,  // [118] P300 = 300. (550 hPa slot)", 1),
        ("gfd_get_inversion_layers(ierr6_saved, p_cup, t_cup, z_cup, kbcon,", 1),
    )),
    "GF-11": (KERNELS / "gf.cu", (
        ("if (FSUB(po[kbcon], po[k]) > K_P150) {", 1),
        ("ISCB[ISCA_pmin_lev] = pmin_lev;", 1),
    )),
    "GF-12": (KERNELS / "gf.cu", (
        ("xmb_ave = GMIN(xmb_ave, K_HUNDRED);", 1),
        ("xmbmax = K_ONE;", 1),
    )),
    # New Tiedtke (cumulus = "ntiedtke").
    "NT-1": (KERNELS / "ntiedtke.cu", (
        ("#define NT_ZDNOPRC 2.0e4f", 2),
        ("if (__fadd_rn(NTC(paph, kcbot), -NTC(paph, kctop)) > NT_ZDNOPRC)", 1),
        ("if (__fadd_rn(NTC(paph, kcbot), -NTC(paph, kctop)) < NT_ZDNOPRC)", 1),
        ("if (ktype[i] == 1 && zpbmpt < NT_ZDNOPRC) ktype[i] = 2;", 1),
        ("if (ktype[i] == 2 && zpbmpt >= NT_ZDNOPRC) ktype[i] = 1;", 1),
    )),
    "NT-2": (KERNELS / "ntiedtke.cu", (
        ("if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 350.0e2f)", 1),
    )),
    "NT-3": (KERNELS / "ntiedtke.cu", (
        ("if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f) {", 1),
    )),
    "NT-4": (KERNELS / "ntiedtke.cu", (
        ("if (pmix < 50.0e2f) {", 1),
    )),
    "NT-5": (KERNELS / "ntiedtke.cu", (
        ("if (__fadd_rn(NTK(paph, klev + 1), -NTK(paph, kdpl[i]))", 1),
    )),
    "NT-6": (KERNELS / "ntiedtke.cu", (
        ("if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f)\n"
         "            itopde = jk;", 1),
    )),
    "NT-7": (KERNELS / "ntiedtke.cu", (
        ("if (!LD && NTC(paph, jk) < 4.0e4f) kt0 = jk;", 1),
        ("kctop0[i] = kt0;", 1),
    )),
    "NT-8": (KERNELS / "ntiedtke.cu", (
        ("zcape[i] = fmaxf(0.0f, fminf(__fadd_rn(zcape1[i], -zcape2[i]),", 1),
        ("const float zdp = __fadd_rn(NTK(pap, jk), -NTK(pap, jk - 1));", 1),
    )),
    "NT-9": (KERNELS / "ntiedtke.cu", (
        ("__fdiv_rn(__fsqrt_rn(__fdiv_rn(NTC(paph, jk),", 1),
        ("5.09e-3f),", 1),
    )),
    "NT-10": (KERNELS / "ntiedtke.cu", (
        ("#define NT_CMFCMAX 1.0f", 1),
        ("mfub = 0.01f;", 1),
    )),
    # This package's own scheme (cumulus = "own").
    "OWN-1": (PACKAGE / "physics" / "arwen_massflux.py", (
        ("evaporation_coefficient: float = 0.5       # per 100 hPa at zero RH", 1),
        ("fraction = xp.where(below_base, xp.minimum(k_e * rh_gap * dp[k], 1.0), 0.0)", 1),
    )),
    # Morrison microphysics.
    "M-1": (KERNELS / "morrison.cu", (
        ("#define MRHOSU (85000.0f / (287.15f * 273.15f))", 1),
        ("real dens54 = powf(MRHOSU / rhoa, 0.54f);", 1),
    )),
    "M-2": (KERNELS / "morrison.cu", (
        ("real dv = 8.794e-5f * powf(*temp, 1.81f) / pressure;", 1),
    )),
    "M-3": (KERNELS / "morrison.cu", (
        ("const real slip = 7.37f * temp / (288.0f * 10.0f * pressure) / 100.0f;", 1),
    )),
    "M-4": (KERNELS / "morrison.cu", (
        ("real ew = fminf(0.99f * pressure, morr_polysvp(tpred, false));", 2),
        ("real ew = fminf(0.99f * p, morr_polysvp(temp, false));", 1),
    )),
    "M-5": (KERNELS / "morrison.cu", (
        ("if (reset_cloud_number) *nc = nc_fixed / rhoa;", 1),
        ("real nc_fixed = 250.0e6f;", 2),
        ("* 1000.0f, 500.0e3f) / rhoa;", 1),
    )),
    # The vertical coordinate every scheme sits on.
    "V-1": (PACKAGE / "vertical.py", (
        ("for ps in (50_000.0, 80_000.0, 105_000.0, 120_000.0):", 1),
    )),
}

#: Every line of the New Tiedtke kernel that spells a literal of order 1e2 to
#: 1e4 in exponent form, which is how the port writes a pressure in Pa.  A
#: line outside this set is a threshold the page does not cover yet.
NTIEDTKE_EXPONENT_LITERAL_LINES = {
    "#define NT_ZDNOPRC 2.0e4f",                                      # NT-1, twice
    "if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 350.0e2f)",  # NT-2
    "if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f) {",  # NT-3
    "if (pmix < 50.0e2f) {",                                          # NT-4
    "< 50.0e2f) {",                                                   # NT-5
    "if (__fadd_rn(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f)",  # NT-6
    "if (!LD && NTC(paph, jk) < 4.0e4f) kt0 = jk;",                   # NT-7
    "&& hgt > 5.0e2f && hgt < 1.0e4f)) return;",                      # metres, not Pa
}

#: Every use of a Grell-Freitas pressure-depth word, outside its #define.
GF_PRESSURE_WORD_USES = {
    "K_CAP_MAXS": 1, "K_CAP_MAXS_SH": 1, "K_CAP_INC_DEEP": 2,
    "K_P100": 1, "K_P150": 2, "K_P200": 3, "K_P300": 1,
}

SHARE_ROW = re.compile(
    r"^\| ((?:GF|NT)-\d+) \| ([0-9.]+) hPa \| ([0-9.]+) % \| ([0-9.]+) % \|$",
    re.MULTILINE)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.mark.parametrize("row", sorted(ROWS))
def test_every_row_still_names_the_source_it_describes(row: str) -> None:
    path, snippets = ROWS[row]
    text = path.read_text(encoding="utf-8")
    for snippet, count in snippets:
        assert text.count(snippet) == count, (
            f"{row}: {path.name} carries {text.count(snippet)} of {snippet!r}, "
            f"the audit page was written against {count}; re-read the row in "
            f"{PAGE.name} before changing either")


def test_the_page_lists_exactly_the_rows_this_test_holds() -> None:
    on_page = set(re.findall(r"^\| ((?:GF|NT|OWN|M|V)-\d+) \|", _page(), re.MULTILINE))
    assert on_page == set(ROWS)


def test_no_new_pressure_literal_reached_the_new_tiedtke_kernel() -> None:
    text = (KERNELS / "ntiedtke.cu").read_text(encoding="utf-8")
    found = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("//", "/*", "*")):
            continue
        if re.search(r"\b\d+\.\d*e[234]f\b", stripped):
            found.add(stripped)
    assert found == NTIEDTKE_EXPONENT_LITERAL_LINES


def test_no_new_use_of_a_grell_freitas_pressure_word() -> None:
    text = (KERNELS / "gf.cu").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#define"))
    uses = {word: len(re.findall(rf"\b{word}\b", body)) for word in GF_PRESSURE_WORD_USES}
    assert uses == GF_PRESSURE_WORD_USES


def test_morrison_compares_no_pressure_with_a_threshold() -> None:
    text = (KERNELS / "morrison.cu").read_text(encoding="utf-8")
    assert not re.search(r"\b(pressure|pres|p)\s*[<>]=?\s*[0-9]", text)
    assert not re.search(r"[0-9]\s*[<>]=?\s*(pressure|pres|p)\b", text)


def test_the_share_columns_are_the_arithmetic_the_page_states() -> None:
    rows = SHARE_ROW.findall(_page())
    assert len(rows) == 11
    for row, hpa, earth, thin in rows:
        value = float(hpa)
        assert f"{100.0 * value / 1013.25:.1f}" == earth, row
        assert f"{100.0 * value / 500.0:.1f}" == thin, row
