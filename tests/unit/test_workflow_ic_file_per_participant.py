"""End-to-end workflow test: the typical multi-file lab setup a per-file/
per-group cross-file IC reference is actually built for -- each of two participants
contributes three ordinary stage recordings plus one dedicated, reference-only IC
recording, ALL EIGHT files matched by one glob and run through ``run_batch`` +
``write_batch`` as a single batch, with the IC value resolved per PARTICIPANT via a
``reference_defaults`` GROUP entry (not a per-file ``processing.references`` entry,
which the ``typed_ic_crossfile`` golden scenario already covers for the single-file
case). This closes a real gap: no earlier test ran a genuinely
multi-file-per-participant batch through the real pipeline end to end.

Two participants (not one) so "one row per group" in the cohort summary and
"resolves per participant, not globally" in the reference link both have something
real to prove -- a single-group batch can't distinguish "grouped correctly" from
"there was only one thing to group" (``core.summary.build_cohort_summary`` itself
only ever writes a "By group" breakdown "when every file lands in a single group";
see its own docstring), and a single IC source can't rule out one participant's
files silently borrowing anOTHER participant's reference.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "..", "golden")
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)

import generate_data as gd  # noqa: E402  (tests/golden's own synthetic-data generator;
# reused here, not duplicated, for the exact same proven trapezoid/plateau breath
# shapes and lead-in convention -- see make_ic_reference_file's own docstring)

from respmech.cli.__main__ import main as cli_main  # noqa: E402
from respmech.core.analysis.references import check_links  # noqa: E402
from respmech.core.pipeline import run_batch  # noqa: E402
from respmech.core.io.writers import write_batch  # noqa: E402
from respmech.core.settings import BreathRef, BreathTypeEntry, GroupReferenceEntry  # noqa: E402
from respmech.core.summary import build_cohort_summary, group_key  # noqa: E402
from respmech.settingsio.migrate import migrate_dict  # noqa: E402
from respmech.settingsio.toml_io import save_toml  # noqa: E402
from respmech.ui.manifest import group_readout  # noqa: E402

#: Same channel mapping as tests/golden's own fixtures (time,EMG1-3,flow,volume,
#: poes,pgas,pdi,ENT1-3) -- what generate_data.make_file/make_ic_reference_file
#: actually write.
_CHANNELS = {"column_flow": 5, "column_volume": 6, "column_poes": 7, "column_pgas": 8,
            "column_pdi": 9, "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}

#: Two participants, each with a dedicated seed pair -- distinct from every seed
#: already used in generate_data.py's own __main__ block, and from each other.
_PARTICIPANTS = {"P01": (70101, 70102), "P02": (70201, 70202)}


def _build_participant_folder(root):
    """Writes 3 stage recordings + 1 reference-only IC recording per participant in
    ``_PARTICIPANTS`` into ``root`` -- 8 files total, one shared folder (the
    realistic "one study folder, several participants" layout)."""
    for participant, (stage_seed, ic_seed) in _PARTICIPANTS.items():
        for i in range(3):
            gd.make_file(os.path.join(root, f"{participant}_stage{i + 1}.csv"),
                        seed=stage_seed + i, n_breaths=4)
        gd.make_ic_reference_file(os.path.join(root, f"{participant}_IC.csv"), seed=ic_seed)


def _settings(root, outdir):
    d = {
        "input": {"inputfolder": str(root), "files": "P0[12]_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": dict(_CHANNELS)},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow"}},
        "output": {"outputfolder": str(outdir), "data": {
            "saveaveragedata": True, "savebreathbybreathdata": True}},
    }
    settings, _ = migrate_dict(d)
    for participant in _PARTICIPANTS:
        ic_file = f"{participant}_IC.csv"
        settings.processing.breath_types.append(
            BreathTypeEntry(file=ic_file, breath=gd.MANOEUVRE_CROSSFILE_IC_BREATH_NO, kind="ic"))
        settings.processing.reference_defaults.append(GroupReferenceEntry(
            group=participant, ic=BreathRef(file=ic_file, breaths=[gd.MANOEUVRE_CROSSFILE_IC_BREATH_NO])))
    return settings


def test_workflow_ic_file_per_participant(tmp_path):
    root = tmp_path / "study"
    outdir = tmp_path / "out"
    os.makedirs(root)
    _build_participant_folder(str(root))
    settings = _settings(str(root), str(outdir))
    settings.validate()

    matched = sorted(f for f in os.listdir(root) if f.endswith(".csv"))
    assert matched == sorted(
        f"{p}_{suffix}.csv" for p in _PARTICIPANTS
        for suffix in ("stage1", "stage2", "stage3", "IC"))

    # group_readout (the live Setup read-out): both IC files are
    # correctly predicted reference-only and excluded from the "N files -> M
    # groups" count; the 6 stage files collapse into exactly the 2 participant
    # groups, 3 files each.
    status, text = group_readout(matched, settings)
    assert status == "info"
    assert text == "6 files (2 reference-only) → 2 groups · P01 (3) · P02 (3)"

    # check_links: every reference_defaults group matches a real analysed file and
    # every linked breath is typed correctly -- no cautions at all.
    assert check_links(settings, matched) == []

    result = run_batch(settings)
    assert not result.failed_files

    for participant in _PARTICIPANTS:
        ic_file = f"{participant}_IC.csv"
        ic_fr = result.files[ic_file]
        # A reference-only file has no tidal breath table/average row of its
        # own -- it never contributes a row to the written cohort/average sheets.
        assert ic_fr.role == "reference"
        assert ic_fr.average_row is None
        assert ic_fr.breaths_table is None
        assert ic_fr.manoeuvres[gd.MANOEUVRE_CROSSFILE_IC_BREATH_NO]["vol_ic"] == pytest.approx(3.0, abs=1e-9)

        for i in range(3):
            stage_file = f"{participant}_stage{i + 1}.csv"
            fr = result.files[stage_file]
            assert fr.error is None
            # Each stage file gets vol_ic_ref FROM ITS OWN PARTICIPANT'S IC file --
            # the group-default resolution (core.analysis.references.
            # resolve_reference) never lets one participant's stage file borrow
            # another participant's reference, even though every IC file in this
            # batch happens to report the SAME vol_ic (3.0, by generator design):
            # this is what ic_ref_source (not just the numeric value) is asserted
            # against below.
            assert fr.average_row["vol_ic_ref"].iloc[0] == pytest.approx(3.0, abs=1e-9)
            assert fr.average_row["ic_ref_source"].iloc[0] == ic_file
            assert fr.references_used["ic"]["source"] == ic_file

    # average_table: exactly the 6 stage files, never the 2 reference-only ones --
    # a reference-only file's average_row is always None and is therefore
    # never appended to the cross-file "Average breathdata" concat.
    assert len(result.average_table) == 6
    assert set(result.average_table["file"]) == {
        f"{p}_stage{i}.csv" for p in _PARTICIPANTS for i in (1, 2, 3)}

    # Cohort summary (P8/P15): "one row per group" resolves here to one N_FILES
    # value per group in the "By group" breakdown (build_cohort_summary stacks one
    # row PER VARIABLE per group, tagged with a leading `group` column -- there is
    # no literal single-row-per-group shape in the written sheet) -- with 2 groups,
    # by_group is populated at all (a single-group batch would leave it None, see
    # this test module's own docstring), and each group's own block reports
    # n_files == 3, never 4: the reference-only IC file is excluded from the
    # grouping the same way it is excluded from average_table above.
    agg = build_cohort_summary(result, settings)
    assert agg["by_group"] is not None
    for participant in _PARTICIPANTS:
        rows = agg["by_group"][agg["by_group"]["group"] == participant]
        assert len(rows) > 0
        assert (rows["n_files"] == 3).all()

    written = write_batch(result, settings, str(outdir))
    assert os.path.join(str(outdir), "data", "Cohort summary.xlsx") in written
    import pandas as pd
    for participant in _PARTICIPANTS:
        # write_batch (unlike the golden harness's own narrower run_scenario
        # serialisation) DOES write a per-file workbook for a reference-only file
        # too -- its "Data" sheet is the file's Manoeuvres table (its only content)
        # rather than a tidal breaths_table, per writers.py's own reference-only
        # handling ("no tidal breathdata to be the Data sheet -- the Manoeuvres table ...
        # takes its place").
        ic_path = os.path.join(str(outdir), "data", f"{participant}_IC.csv.breathdata.xlsx")
        assert ic_path in written
        ic_data = pd.read_excel(ic_path, sheet_name="Data")
        assert ic_data["vol_ic"].iloc[0] == pytest.approx(3.0, abs=1e-9)
        for i in range(1, 4):
            assert os.path.join(
                str(outdir), "data", f"{participant}_stage{i}.csv.breathdata.xlsx") in written

    toml = tmp_path / "settings.toml"
    save_toml(settings, toml)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0


def test_workflow_group_keys_are_exactly_the_two_participants():
    """Sanity-check on this test's own assumption, isolated from the full pipeline
    above: core.summary.group_key's default leading-token rule really does split
    "P01_stage1.csv"/"P01_IC.csv" to "P01" (not, say, "P01_stage1" whole) -- if this
    ever stopped holding, every assertion above would fail for a confusing reason
    (a "0 groups"/"8 groups" mismatch) rather than this direct one."""
    for participant in _PARTICIPANTS:
        assert group_key(f"{participant}_stage1.csv") == participant
        assert group_key(f"{participant}_IC.csv") == participant
