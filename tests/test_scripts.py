import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freenerf_ns import datasets  # noqa: E402
from tests.helpers import make_fake_blender  # noqa: E402


def run(args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)


def test_run_all_list_reports_the_proposal_totals():
    r = run(["scripts/run_all.py", "--list"])
    assert r.returncode == 0, r.stderr
    assert "66 runs" in r.stdout and "46.0" not in r.stdout   # 66 evaluated runs; the 36 short tuning runs are E0
    only = run(["scripts/run_all.py", "--list", "--only", "E4"])
    assert "3 runs" in only.stdout and "Aprime" in only.stdout


def test_run_arm_dry_run_prints_the_three_stages_with_correct_paths():
    import run_arm
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "b", ("lego",), size=16)
        datasets.synthetic_manifests(t / "b", t / "m", scenes=("lego",))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_arm.execute("lego", "D", 8, 1, 5000, manifests=str(t / "m"), data_root=str(t / "b"), results_dir=str(t / "r"),
                            out_dir="outputs", orbit=30, dry_run=True)
        out = buf.getvalue()
        assert out.count("$") == 3
        assert "ns-train freenerf" in out and "--experiment-name lego_D_n8_s1" in out
        assert "freenerf_ns.export" in out and "outputs/lego_D_n8_s1/freenerf/main/config.yml" in out and "--orbit 30" in out
        assert "evalkit.run_eval" in out and "--n 8" in out and "--no-train" not in out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_arm.execute("lego", "A", "dense", 0, 5000, manifests=str(t / "m"), data_root=str(t / "b"), results_dir=str(t / "r"),
                            dry_run=True)
        assert "--no-train" in buf.getvalue() and "--n 100" in buf.getvalue()           # dense runs skip train-view export


def test_run_arm_skips_finished_runs():
    import run_arm
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "r").mkdir()
        (t / "r" / "lego_A_n8_s0.json").write_text("{}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_arm.execute("lego", "A", 8, 0, 10, manifests=str(t), results_dir=str(t / "r"), dry_run=True)
        assert "[skip]" in buf.getvalue() and "$" not in buf.getvalue()


def test_make_manifests_check_flag():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "b", ("lego",), size=16)
        datasets.synthetic_manifests(t / "b", t / "m", scenes=("lego",))
        ok = run(["scripts/make_manifests.py", "--check", "--manifests", str(t / "m")])
        assert ok.returncode == 0 and "manifests OK" in ok.stdout
        p = t / "m" / "lego_n4.json"; d = json.loads(p.read_text()); d["train"][0] += 1; p.write_text(json.dumps(d))
        bad = run(["scripts/make_manifests.py", "--check", "--manifests", str(t / "m")])
        assert bad.returncode == 1 and "changed" in bad.stdout


def test_freeze_config_requires_equal_budgets_and_tuned_values():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "tuning").mkdir()
        cfg = t / "frozen.json"
        cfg.write_text(json.dumps({"A": {"lr": 3e-4}, "D": {"freq_end_frac": 0.4}, "E": {"ds_weight": 0.1}}))
        for arm, n in (("A", 6), ("D", 6), ("E", 5)):
            for k in range(n):
                (t / "tuning" / f"{arm}_t{k}.json").write_text("{}")
        r = run(["scripts/freeze_config.py", "--frozen", str(cfg), "--tuning", str(t / "tuning")])
        assert r.returncode == 1 and "unequal tuning budgets" in r.stdout
        (t / "tuning" / "E_t5.json").write_text("{}")
        r = run(["scripts/freeze_config.py", "--frozen", str(cfg), "--tuning", str(t / "tuning")])
        assert r.returncode == 0 and "git tag config-frozen" in r.stdout
        assert json.loads(cfg.read_text())["_meta"]["frozen"] is True


def test_run_controls_on_a_prepared_workspace():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "b", ("lego",), size=16)
        datasets.synthetic_manifests(t / "b", t / "m", scenes=("lego",))
        datasets.make_sparse(t / "b", t / "m", "lego", 8, t / "s")
        r = run(["scripts/run_controls.py", "--manifests", str(t / "m"), "--sparse", str(t / "s"), "--results", str(t / "res")])
        assert r.returncode == 0, r.stdout + r.stderr
        assert "C5 isolation lego_n8" in r.stdout and "FAIL" not in r.stdout
        (t / "s" / "lego_n8" / "test").mkdir()                                              # smuggle in a test folder
        bad = run(["scripts/run_controls.py", "--manifests", str(t / "m"), "--sparse", str(t / "s"), "--results", str(t / "res")])
        assert bad.returncode == 1 and "FAIL" in bad.stdout


def test_permuted_pose_control_run_is_labelled_separately_from_arm_A():
    import run_arm
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "b", ("lego",), size=16)
        datasets.synthetic_manifests(t / "b", t / "m", scenes=("lego",))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_arm.execute("lego", "A", 8, 0, 10, manifests=str(t / "m"), data_root=str(t / "b"), results_dir=str(t / "r"),
                            permute=True, dry_run=True)
        out = buf.getvalue()
        assert "--arm A_perm" in out and "--run-id lego_A_n8_s0_perm" in out


def test_lite_profile_produces_laptop_sized_commands():
    import run_arm
    kw = run_arm.lite_kwargs("mps")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_fake_blender(t / "b", ("lego",), size=16)
        datasets.synthetic_manifests(t / "b", t / "m", scenes=("lego",))
        kw.update(data_root=str(t / "b"), results_dir=str(t / "r"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_arm.execute("lego", "D", 8, 0, 3000, manifests=str(t / "m"), dry_run=True, **kw)
        out = buf.getvalue()
        assert "--pipeline.datamanager.train-num-rays-per-batch 512" in out
        assert "--machine.device-type mps" in out and "--pipeline.model.eval-num-rays-per-chunk 4096" in out
        assert "--width 200 --height 200" in out and "outputs_lite/lego_D_n8_s0" in out
    full = " ".join(__import__("freenerf_ns.arms", fromlist=["x"]).train_command(
        __import__("freenerf_ns.arms", fromlist=["x"]).arm_spec("D"), data_dir="d", out_dir="o", run_id="r", iters=1, seed=0))
    assert "device-type" not in full and "train-num-rays-per-batch" not in full          # full profile unchanged


def test_shards_partition_the_plan():
    a = run(["scripts/run_all.py", "--plan", "submission", "--list", "--shard", "0/2"]).stdout
    b = run(["scripts/run_all.py", "--plan", "submission", "--list", "--shard", "1/2"]).stdout
    ids = lambda out: {l.split()[1] for l in out.splitlines() if l.startswith("S-")}
    assert ids(a) and ids(b) and not ids(a) & ids(b) and len(ids(a) | ids(b)) == 30
