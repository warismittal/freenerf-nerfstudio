"""Results dashboard (CPU only): reads results/ written by evalkit.aggregate.   streamlit run app/streamlit_app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path("results")
st.set_page_config(page_title="FreeNeRF gap recovery", layout="wide")
st.title("Sparse-view NeRF: how much of the 100-to-8-view gap does a prior recover?")

if not (R / "tables").exists():
    st.warning("No results yet. Run `python -m evalkit.aggregate` first.")
    st.stop()

hyp = json.loads((R / "hypotheses.json").read_text()) if (R / "hypotheses.json").exists() else None
if hyp:
    st.subheader("Reproduction verdict: " + hyp["verdict"]["verdict"])
    st.dataframe(pd.DataFrame(hyp["hypotheses"])[["id", "status", "text"]], use_container_width=True, hide_index=True)

t = {p.stem: pd.read_csv(p) for p in sorted((R / "tables").glob("*.csv"))}
tab1, tab2, tab3, tab4 = st.tabs(["Gap recovery", "All metrics", "Figures", "Videos"])
with tab1:
    if "gap_recovery" in t:
        st.dataframe(t["gap_recovery"].round(3), use_container_width=True, hide_index=True)
    if "view_equivalence" in t:
        st.caption("N_eq: the number of views vanilla NeRF needs to match the prior's 8-view PSNR.")
        st.dataframe(t["view_equivalence"].round(2), use_container_width=True, hide_index=True)
with tab2:
    if "summary" in t:
        scene = st.selectbox("Scene", sorted(t["summary"].scene.unique()))
        st.dataframe(t["summary"][t["summary"].scene == scene].round(3), use_container_width=True, hide_index=True)
with tab3:
    for p in sorted((R / "figures").glob("*.png")):
        st.image(str(p), caption=p.stem, use_container_width=True)
with tab4:
    vids = sorted((R / "videos").glob("*.mp4")) if (R / "videos").exists() else []
    for v in vids:
        st.video(str(v))
    if not vids:
        st.info("Run scripts/make_videos.py to create the vanilla | FreeNeRF | depth-smoothness videos.")
