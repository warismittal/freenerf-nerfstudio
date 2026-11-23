from freenerf_ns import arms, plan


def test_plan_totals_match_the_proposal_table():
    core = plan.core_runs()
    tuning = plan.tuning_runs()
    assert len(core) == 66 and len(tuning) == 36
    assert len(core) + len(tuning) == 102                                   # "Core plan: 102 runs"
    assert abs(plan.total_cost(core) + plan.total_cost(tuning) - 82) < 1e-9  # "82 standard-run equivalents"
    assert abs(plan.total_cost(plan.core_runs(True)) - plan.total_cost(core) - 13.67) < 0.01   # extensions +13.7 SRE


def test_plan_breakdown_per_experiment():
    core = plan.core_runs()
    cnt = lambda e: sum(r.exp == e for r in core)
    assert (cnt("E1"), cnt("E2a"), cnt("E2b"), cnt("E3"), cnt("E4")) == (12, 18, 18, 15, 3)
    assert abs(sum(r.cost for r in core if r.exp == "E2b") - 22) < 1e-9
    assert len({r.run_id for r in core}) == len(core)                      # run ids are unique
    assert {r.scene for r in core if r.exp == "E1"} == {"drums", "ficus"}
    assert plan.select(core, "E4") == [r for r in core if r.exp == "E4"]


def test_arms_B_and_C_inherit_D_hyperparameters():
    frozen = {"D": {"freq_end_frac": 0.4, "occ_weight": 0.03}, "E": {"ds_weight": 1.0, "ds_patch_size": 4}, "A": {"lr": 2e-4}}
    d, b, c = (arms.arm_spec(a, frozen) for a in "DBC")
    assert d.model_flags["freq_end_frac"] == b.model_flags["freq_end_frac"] == 0.4
    assert d.model_flags["occ_weight"] == c.model_flags["occ_weight"] == 0.03
    assert set(b.model_flags) == {"freq_mask", "freq_end_frac"} and set(c.model_flags) == {"occ_weight", "occ_num_samples"}
    assert all(arms.arm_spec(a, frozen).lr == 2e-4 for a in "ABCDE")        # one tuned learning rate for every arm
    assert arms.arm_spec("Aprime", frozen).lr is None and arms.arm_spec("Aprime").method == "sparse-vanilla-default"


def test_identity_control_arm_activates_mask_path_with_unit_mask():
    s = arms.arm_spec("C2")
    assert s.model_flags == {"freq_mask": "ramp", "freq_end_frac": 0.0, "occ_weight": 0.0}


def test_train_command_flags():
    spec = arms.arm_spec("D", {"D": {"freq_end_frac": 0.4}})
    cmd = arms.train_command(spec, data_dir="data/sparse/lego_n8", out_dir="outputs", run_id="lego_D_n8_s1", iters=20000, seed=1)
    s = " ".join(cmd)
    assert cmd[:2] == ["ns-train", "freenerf"] and cmd[-1] == "blender-data"
    assert "--pipeline.model.freq-mask ramp" in s and "--pipeline.model.freq-end-frac 0.4" in s
    assert "--pipeline.model.occ-weight 0.01" in s and "--machine.seed 1" in s
    assert "--optimizers.fields.scheduler.max-steps 20000" in s and "--timestamp main" in s
    vanilla = " ".join(arms.train_command(arms.arm_spec("Aprime"), data_dir="d", out_dir="o", run_id="r", iters=100, seed=0))
    assert "scheduler" not in vanilla and "--optimizers.fields.optimizer.lr" not in vanilla
    real = " ".join(arms.train_command(arms.arm_spec("E", scene_kind="real"), data_dir="d", out_dir="o", run_id="r", iters=100, seed=0, scene_kind="real"))
    assert "--pipeline.model.ds-pose-source interp" in real and "background-color last_sample" in real
    assert "ds-alpha-weight" not in s


def test_unknown_arm_rejected():
    try:
        arms.arm_spec("Z")
    except ValueError:
        return
    raise AssertionError


def test_submission_plan_is_a_subset_of_the_core_design():
    sub = plan.submission_runs()
    assert len(sub) == 30 and len({r.run_id for r in sub}) == 30
    assert {r.arm for r in sub} <= set(arms.ARMS)
    main = [r for r in sub if r.exp == "S-main" and r.n == 8]
    assert {(r.scene, r.arm) for r in main} == {(s, a) for s in ("lego", "chair") for a in ("A", "D")}
    assert all(len([r for r in main if (r.scene, r.arm) == k]) == 3 for k in {(r.scene, r.arm) for r in main})
