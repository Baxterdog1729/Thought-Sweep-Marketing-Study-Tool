import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency, spearmanr
import json
import math
import random

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Thought Sweep Research & Marketing Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0f0a1c; color: #f1f5f9; }
    .css-1d3b047, .css-12w0qpk { background-color: #160e26; }
    div[data-testid="stMetricValue"] { color: #a78bfa; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .stButton>button:hover { background: #6d28d9; color: white; }
    .quad-card {
        padding: 15px; border-radius: 12px; margin-bottom: 10px;
        backdrop-filter: blur(10px); border: 1px solid rgba(167, 139, 250, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION & GLOBAL UTILITIES
# ==============================================================================
def init_session_state():
    defaults = {
        "step1_interviews": [],
        "step2_interviews": [],
        "step3_interviews": [],
        "stage_definitions": {},
        "codebook": [
            {"id": "C1", "name": "Lack of ideas", "category": "Challenge Codes (C)", "desc": "Unable to find or choose a topic/direction."},
            {"id": "C2", "name": "Idea-to-plan difficulty", "category": "Challenge Codes (C)", "desc": "Having an idea but unable to narrow into scope or plan."},
            {"id": "C3", "name": "Organizing thoughts & sources", "category": "Challenge Codes (C)", "desc": "Notes, sources or drafts scattered or hard to find."},
            {"id": "C4", "name": "Idea loss", "category": "Challenge Codes (C)", "desc": "Forgetting, never recording spontaneous ideas."},
            {"id": "C5", "name": "Time management", "category": "Challenge Codes (C)", "desc": "Misjudging work duration, scheduling or pacing."},
            {"id": "C6", "name": "Procrastination", "category": "Challenge Codes (C)", "desc": "Knowingly putting off or avoiding thesis work."},
            {"id": "C7", "name": "Cognitive overwhelm", "category": "Challenge Codes (C)", "desc": "Feeling flooded, frozen or unsure where to start."},
            {"id": "T1", "name": "Digital notes", "category": "Tools Used (T)", "desc": "Notion, Obsidian, OneNote, Apple Notes."},
            {"id": "T2", "name": "Paper", "category": "Tools Used (T)", "desc": "Notebooks, sticky notes, whiteboards."},
            {"id": "T3", "name": "Voice memos", "category": "Tools Used (T)", "desc": "Voice messages, audio recording."},
            {"id": "W1", "name": "Faster capture", "category": "Capabilities Wanted (W)", "desc": "Getting an idea down quickly, anywhere."},
            {"id": "W4", "name": "Turning ideas into tasks", "category": "Capabilities Wanted (W)", "desc": "Moving from idea to action items."}
        ],
        "coder_a_annotations": [],  # [{interview_id, selected_text, code_id, group, discipline, stage}]
        "coder_b_annotations": [],  # [{interview_id, selected_text, code_id}]
        "kappa_locked": False,
        "kappa_value": 0.0,
        "surveys": []  # [{survey_id, code_id, eligible, experienced, severity, tools_referenced}]
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# Global Export / Clear Utilities
def export_all_data():
    data = {
        "stage_definitions": st.session_state.stage_definitions,
        "step1_interviews": st.session_state.step1_interviews,
        "step2_interviews": st.session_state.step2_interviews,
        "step3_interviews": st.session_state.step3_interviews,
        "codebook": st.session_state.codebook,
        "coder_a_annotations": st.session_state.coder_a_annotations,
        "coder_b_annotations": st.session_state.coder_b_annotations,
        "kappa_locked": st.session_state.kappa_locked,
        "kappa_value": st.session_state.kappa_value,
        "surveys": st.session_state.surveys
    }
    return json.dumps(data, indent=2)

def clear_all_data():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()

# ==============================================================================
# SIDEBAR NAVIGATION & GLOBAL UTILITY ACTIONS
# ==============================================================================
st.sidebar.title("🧠 Thought Sweep Engine")
interface_choice = st.sidebar.radio(
    "Select Interface:",
    [
        "1. Step 1: Thesis Stage Entry",
        "2. Step 2: Faculty Observations",
        "3. Step 3: Student Experiences",
        "4. Qualitative Coding & ICR Engine",
        "5. Qualitative Results & Subtags",
        "6. Quantitative Survey Entry & Stats",
        "7. 2x2 Prioritization Matrix"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Data Controls")
st.sidebar.download_button(
    label="📥 Export All Data (JSON)",
    data=export_all_data(),
    file_name="thought_sweep_complete_research_data.json",
    mime="application/json"
)
if st.sidebar.button("🗑️ Clear All Data"):
    clear_all_data()

# ==============================================================================
# 3. INTERFACE 1: STEP 1 INTERVIEW ENTRY & CONCRETE STAGES
# ==============================================================================
if interface_choice.startswith("1."):
    st.header("Interface 1: Step 1 Interview Guide — Defining Thesis Process")
    st.caption("Record Step 1 interviews and establish standardized thesis stages across the system.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Log Step 1 Interview")
        with st.form("step1_form"):
            int_id = st.text_input("Interview ID *", placeholder="e.g. S1-01")
            group = st.selectbox("Group *", ["Faculty", "Student-In-process", "Student-Completed"])
            disc = st.selectbox("Discipline *", ["STEM", "Creative", "Humanities", "Social Science"])

            st.markdown("**Preloaded Guide Questions:**")
            q1 = st.text_area("A1. Describe the overall thesis process from start to finish as you see it.", height=80)
            q2 = st.text_area("A2. What are the distinct stages or milestones students go through?", height=80)
            q3 = st.text_area("A3. Where do students spend the most time or struggle most?", height=80)

            if st.form_submit_button("Save Step 1 Entry"):
                if int_id:
                    entry = {"id": int_id, "group": group, "discipline": disc, "q1": q1, "q2": q2, "q3": q3}
                    st.session_state.step1_interviews.append(entry)
                    st.success(f"Saved Step 1 Interview {int_id}")
                    st.rerun()

    with col2:
        st.subheader("Standardize Thesis Process Stages")
        st.info("Set concrete stage names and definitions here. These will carry across Interfaces 2–7.")
        
        stages = ["Topic Selection / Ideation", "Proposal & Scoping", "Literature Review", 
                  "Data Collection / Execution", "Analysis & Synthesis", "Drafting & Writing"]
        
        updated_defs = {}
        for stage in stages:
            curr = st.session_state.stage_definitions.get(stage, "")
            updated_defs[stage] = st.text_input(f"Stage: {stage}", value=curr, placeholder=f"Define {stage} rules...")
        
        if st.button("Lock Standardized Stage Definitions"):
            st.session_state.stage_definitions = updated_defs
            st.success("Thesis stage definitions locked across software!")

# ==============================================================================
# 4. INTERFACE 2: STEP 2 FACULTY OBSERVATIONS
# ==============================================================================
elif interface_choice.startswith("2."):
    st.header("Interface 2: Step 2 Faculty Interview Guide — Observed Challenges")
    
    with st.form("step2_form"):
        col1, col2 = st.columns(2)
        with col1:
            int_id = st.text_input("Faculty Interview ID *", placeholder="e.g. F-01")
            disc = st.selectbox("Faculty Discipline *", ["STEM", "Creative", "Humanities", "Social Science"])
        with col2:
            stages_list = list(st.session_state.stage_definitions.keys()) if st.session_state.stage_definitions else ["Topic Selection / Ideation"]
            hardest_stage = st.selectbox("Observed Hardest Stage *", stages_list)
        
        q_b3 = st.text_area("B3. Describe a recent example of a student struggling during this stage.", height=100)
        q_d4 = st.text_area("D4. When a student is stuck, what do you suggest and where do you send them?", height=100)
        
        if st.form_submit_button("Save Faculty Interview"):
            if int_id:
                entry = {"id": int_id, "group": "Faculty", "discipline": disc, "hardest_stage": hardest_stage, "b3": q_b3, "d4": q_d4}
                st.session_state.step2_interviews.append(entry)
                st.success(f"Saved Faculty Interview {int_id}")
                st.rerun()

# ==============================================================================
# 5. INTERFACE 3: STEP 3 STUDENT EXPERIENCES
# ==============================================================================
elif interface_choice.startswith("3."):
    st.header("Interface 3: Step 3 Student Interview Guide — Personal Experiences")
    
    with st.form("step3_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            int_id = st.text_input("Student Interview ID *", placeholder="e.g. S-01")
        with col2:
            group = st.selectbox("Student Status *", ["Student-Beginning", "Student-In-process", "Student-Completed"])
        with col3:
            disc = st.selectbox("Discipline *", ["STEM", "Creative", "Humanities", "Social Science"])

        stages_list = list(st.session_state.stage_definitions.keys()) if st.session_state.stage_definitions else ["Topic Selection / Ideation"]
        hardest_stage = st.selectbox("Self-Reported Hardest Stage", stages_list)
        
        q_ideas = st.text_area("Ideas & Capture: Walk me through the last time a new research idea came to you.", height=90)
        q_struggle = st.text_area("Struggles & Tools: Describe the moment you felt most stuck and what tool/method you used.", height=90)

        if st.form_submit_button("Save Student Interview"):
            if int_id:
                entry = {"id": int_id, "group": group, "discipline": disc, "hardest_stage": hardest_stage, "q_ideas": q_ideas, "q_struggle": q_struggle}
                st.session_state.step3_interviews.append(entry)
                st.success(f"Saved Student Interview {int_id}")
                st.rerun()

# ==============================================================================
# 6. INTERFACE 4: QUALITATIVE CODING & INTER-CODER RELIABILITY (ICR)
# ==============================================================================
elif interface_choice.startswith("4."):
    st.header("Interface 4: Qualitative Transcript Repository, Living Codebook & ICR Engine")
    
    tab_code, tab_annot, tab_icr = st.tabs(["📖 Living Codebook", "🏷️ Coding Interface (Coder A)", "🛡️ Coder B & Kappa Verification"])

    # 4A. LIVING CODEBOOK
    with tab_code:
        st.subheader("Manage Codebook v0.3")
        with st.expander("➕ Add New Emergent Code (EM)"):
            c_id = st.text_input("Code ID", placeholder="EM-01")
            c_name = st.text_input("Code Title", placeholder="Citation Confusion")
            c_cat = st.selectbox("Category", ["Challenge Codes (C)", "Tools Used (T)", "Capabilities Wanted (W)", "Emergent Themes (EM)"])
            c_desc = st.text_area("Definition & Boundaries")
            if st.button("Add Code to Codebook"):
                if c_id and c_name:
                    st.session_state.codebook.append({"id": c_id, "name": c_name, "category": c_cat, "desc": c_desc})
                    st.success(f"Added {c_id} to Codebook")
                    st.rerun()

        cb_df = pd.DataFrame(st.session_state.codebook)
        st.dataframe(cb_df, use_container_width=True)

    # 4B. TRANSCRIPT CODING INTERFACE
    with tab_annot:
        st.subheader("Apply Codebook v0.3 Tags to Saved Transcripts")
        all_transcripts = (
            [{"source": "Step 1", "id": x["id"], "group": x["group"], "discipline": x["discipline"], "text": f"{x['q1']} {x['q2']} {x['q3']}"} for x in st.session_state.step1_interviews] +
            [{"source": "Step 2", "id": x["id"], "group": x["group"], "discipline": x["discipline"], "text": f"{x['b3']} {x['d4']}"} for x in st.session_state.step2_interviews] +
            [{"source": "Step 3", "id": x["id"], "group": x["group"], "discipline": x["discipline"], "text": f"{x['q_ideas']} {x['q_struggle']}"} for x in st.session_state.step3_interviews]
        )

        if not all_transcripts:
            st.warning("No interview transcripts logged yet. Enter interviews in Interfaces 1–3 first.")
        else:
            selected_int = st.selectbox("Select Interview Transcript:", [x["id"] for x in all_transcripts])
            curr_item = next(x for x in all_transcripts if x["id"] == selected_int)
            
            st.markdown(f"**Group:** `{curr_item['group']}` | **Discipline:** `{curr_item['discipline']}`")
            st.text_area("Transcript Text", value=curr_item["text"], height=150, disabled=True)

            col1, col2 = st.columns(2)
            with col1:
                selected_text = st.text_input("Paste Excerpt / Thought Unit to Code:")
            with col2:
                code_options = [f"{c['id']} - {c['name']}" for c in st.session_state.codebook]
                selected_code = st.selectbox("Apply Code:", code_options)

            if st.button("Save Coding Tag"):
                if selected_text and selected_code:
                    code_id = selected_code.split(" - ")[0]
                    tag_entry = {
                        "interview_id": curr_item["id"],
                        "selected_text": selected_text,
                        "code_id": code_id,
                        "group": curr_item["group"],
                        "discipline": curr_item["discipline"]
                    }
                    st.session_state.coder_a_annotations.append(tag_entry)
                    st.success(f"Tagged '{selected_text}' with {code_id}")
                    st.rerun()

            st.markdown("---")
            st.subheader("Existing Annotations (Coder A)")
            st.dataframe(pd.DataFrame(st.session_state.coder_a_annotations), use_container_width=True)

    # 4C. CODER B & COHEN'S KAPPA
    with tab_icr:
        st.subheader("Inter-Coder Reliability (ICR) Verification Engine")
        st.caption("Coder B reviews a minimal sample subset to verify coding accuracy (Requirement: Cohen's Kappa ≥ 0.70).")

        if not st.session_state.coder_a_annotations:
            st.info("Coder A must apply annotations before ICR verification can begin.")
        else:
            sample_size = max(1, math.ceil(len(st.session_state.coder_a_annotations) * 0.25))
            sample_items = st.session_state.coder_a_annotations[:sample_size]
            
            st.markdown(f"**Required Coder B Verification Sample Size:** `{sample_size}` items")

            coder_b_inputs = []
            for idx, item in enumerate(sample_items):
                st.markdown(f"**Item {idx+1} (ID: {item['interview_id']}):** *\"{item['selected_text']}\"*")
                b_code = st.selectbox(
                    f"Coder B Code for Item {idx+1}:", 
                    [c["id"] for c in st.session_state.codebook], 
                    key=f"b_code_{idx}"
                )
                coder_b_inputs.append(b_code)

            if st.button("Calculate Cohen's Kappa Score"):
                a_codes = [item["code_id"] for item in sample_items]
                
                # Agreement calculation
                matches = sum(1 for a, b in zip(a_codes, coder_b_inputs) if a == b)
                po = matches / sample_size
                pe = 1.0 / len(st.session_state.codebook)  # Simplified expected agreement
                kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0
                
                st.session_state.kappa_value = round(kappa, 3)
                if kappa >= 0.70:
                    st.session_state.kappa_locked = True
                    st.success(f"✅ PASSED! Cohen's Kappa = {kappa:.3f} (≥ 0.70). Data locked for analysis!")
                else:
                    st.session_state.kappa_locked = False
                    st.error(f"❌ FAILED! Cohen's Kappa = {kappa:.3f} (< 0.70). Please refine Codebook definitions in Subtab 1 and recode.")

# ==============================================================================
# 7. INTERFACE 5: QUALITATIVE RESULTS & SUBTAG DRILL-DOWN
# ==============================================================================
elif interface_choice.startswith("5."):
    st.header("Interface 5: Qualitative Results & Subtag Drill-Down")
    
    if not st.session_state.coder_a_annotations:
        st.info("No qualitative coding annotations recorded yet.")
    else:
        df_annot = pd.DataFrame(st.session_state.coder_a_annotations)
        
        # Enforce unique participant count rule
        df_unique = df_annot.drop_duplicates(subset=["interview_id", "code_id"])

        st.subheader("Top 5 Codes Overall (Unique Participant Counts)")
        top_overall = df_unique["code_id"].value_counts().head(5)
        st.bar_chart(top_overall)

        st.markdown("---")
        st.subheader("Filter Top 5 Codes by Subtag")
        
        filter_type = st.radio("Drill Down By:", ["Discipline", "Group (Faculty vs Student)", "Individual Interview ID"])
        
        if filter_type == "Discipline":
            disc = st.selectbox("Select Discipline:", df_unique["discipline"].unique())
            filtered = df_unique[df_unique["discipline"] == disc]
            st.table(filtered["code_id"].value_counts().head(5))
            
        elif filter_type == "Group (Faculty vs Student)":
            grp = st.selectbox("Select Group:", df_unique["group"].unique())
            filtered = df_unique[df_unique["group"] == grp]
            st.table(filtered["code_id"].value_counts().head(5))
            
        else:
            iid = st.selectbox("Select Interview ID:", df_unique["interview_id"].unique())
            filtered = df_unique[df_unique["interview_id"] == iid]
            st.table(filtered["code_id"].value_counts().head(5))

# ==============================================================================
# 8. INTERFACE 6: QUANTITATIVE SURVEY ENTRY & STATISTICAL ENGINE
# ==============================================================================
elif interface_choice.startswith("6."):
    st.header("Interface 6: Quantitative Survey Entry & Statistical Engine")
    
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Record Survey Data Point")
        with st.form("survey_form"):
            s_id = st.text_input("Survey ID *", placeholder="e.g. R-001")
            code_id = st.selectbox("Target Code *", [c["id"] for c in st.session_state.codebook])
            eligible = st.checkbox("Eligible for Stage?", value=True)
            experienced = st.checkbox("Experienced Difficulty?", value=True)
            severity = st.slider("Pain Severity Rating (1-5)", 1, 5, 4)
            tools = st.multiselect("Tools Referenced", ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"])

            if st.form_submit_button("Record Survey Response"):
                if s_id:
                    entry = {
                        "survey_id": s_id, "code_id": code_id, "eligible": eligible,
                        "experienced": experienced, "severity": severity, "tools_referenced": tools
                    }
                    st.session_state.surveys.append(entry)
                    st.success(f"Saved Survey Response {s_id}")
                    st.rerun()

    with col2:
        st.subheader("Automated Background Statistical Tests")
        if not st.session_state.surveys:
            st.info("No survey response entries recorded yet.")
        else:
            df_surv = pd.DataFrame(st.session_state.surveys)
            
            results = []
            for c in st.session_state.codebook:
                cid = c["id"]
                sub = df_surv[df_surv["code_id"] == cid]
                n_el = len(sub[sub["eligible"] == True])
                n_high = len(sub[(sub["eligible"] == True) & (sub["experienced"] == True) & (sub["severity"] >= 4)])
                
                pain_rate = (n_high / n_el) if n_el > 0 else 0.0
                
                # Wilson 95% Confidence Interval Calculation
                if n_el > 0:
                    z = 1.96
                    p = pain_rate
                    denom = 1 + (z**2)/n_el
                    center = p + (z**2)/(2*n_el)
                    dev = z * math.sqrt((p*(1-p) + (z**2)/(4*n_el))/n_el)
                    wilson_low = max(0.0, (center - dev)/denom)
                    wilson_high = min(1.0, (center + dev)/denom)
                else:
                    wilson_low, wilson_high = 0.0, 0.0

                # One-sided exact binomial test (against 20% benchmark)
                p_val_binom = stats.binomtest(n_high, n_el, p=0.20, alternative='greater').pvalue if n_el > 0 else 1.0

                results.append({
                    "Code": cid,
                    "Eligible N": n_el,
                    "High Pain N": n_high,
                    "Pain Rate %": f"{pain_rate*100:.1f}%",
                    "Wilson 95% CI": f"[{wilson_low*100:.1f}%, {wilson_high*100:.1f}%]",
                    "Exact Binomial p-val": f"{p_val_binom:.4f}"
                })

            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ==============================================================================
# 9. INTERFACE 7: INTEGRATED 2x2 OPPORTUNITY PRIORITIZATION MATRIX
# ==============================================================================
elif interface_choice.startswith("7."):
    st.header("Interface 7: Integrated 2x2 Opportunity Prioritization Engine")
    
    # Calculate Qual Frequency and Survey Pain Rate per code
    df_annot = pd.DataFrame(st.session_state.coder_a_annotations) if st.session_state.coder_a_annotations else pd.DataFrame()
    df_surv = pd.DataFrame(st.session_state.surveys) if st.session_state.surveys else pd.DataFrame()

    total_students = df_annot[df_annot["group"].str.contains("Student", na=False)]["interview_id"].nunique() if not df_annot.empty else 0
    
    matrix_data = []

    for c in st.session_state.codebook:
        cid = c["id"]
        
        # Qualitative Frequency
        if not df_annot.empty:
            code_students = df_annot[(df_annot["code_id"] == cid) & (df_annot["group"].str.contains("Student", na=False))]["interview_id"].nunique()
            qual_freq = (code_students / total_students) if total_students > 0 else 0.0
            faculty_count = df_annot[(df_annot["code_id"] == cid) & (df_annot["group"] == "Faculty")]["interview_id"].nunique()
        else:
            qual_freq, faculty_count, code_students = 0.0, 0, 0

        # Survey Pain Rate
        if not df_surv.empty:
            sub = df_surv[df_surv["code_id"] == cid]
            n_el = len(sub[sub["eligible"] == True])
            n_high = len(sub[(sub["eligible"] == True) & (sub["experienced"] == True) & (sub["severity"] >= 4)])
            pain_rate = (n_high / n_el) if n_el > 0 else 0.0
        else:
            pain_rate, n_el = 0.0, 0

        # Exact Binomial test
        p_val = stats.binomtest(code_students, total_students, p=0.25, alternative='greater').pvalue if total_students > 0 else 1.0

        matrix_data.append({
            "code": c,
            "qual_freq": qual_freq,
            "pain_rate": pain_rate,
            "triangulated": (faculty_count > 0 and code_students > 0),
            "p_val": p_val
        })

    # Apply Holm-Bonferroni Correction
    matrix_data.sort(key=lambda x: x["p_val"])
    m = len(matrix_data)
    for idx, item in enumerate(matrix_data):
        alpha = 0.05 / (m - idx)
        item["significant"] = item["p_val"] <= alpha

    # Display 2x2 Grid Layout
    st.subheader("Strategic 2x2 Matrix Positioning")
    col_left, col_grid = st.columns([0.1, 0.9])

    with col_left:
        st.markdown("<br><br><br><br>**← High Pain Severity Rate (Survey ≥ 20%)**", unsafe_allow_html=True)

    with col_grid:
        q_lead, q_supp = [], []
        q_niche, q_drop = [], []

        for item in matrix_data:
            cid = item["code"]["id"]
            is_high_freq = item["qual_freq"] >= 0.25
            is_high_pain = item["pain_rate"] >= 0.20
            
            # Triangulation rule for Lead Message
            if is_high_freq and is_high_pain and item["triangulated"]:
                q_lead.append(item)
            elif not is_high_freq and is_high_pain:
                q_supp.append(item)
            elif is_high_freq and not is_high_pain:
                q_niche.append(item)
            else:
                q_drop.append(item)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("<div class='quad-card' style='background: rgba(245, 158, 11, 0.15);'><b>SUPPORTING MESSAGE</b><br><small>Low Freq / High Pain</small></div>", unsafe_allow_html=True)
            for x in q_supp: st.write(f"• **{x['code']['id']}** ({x['code']['name']})")
            
        with g2:
            st.markdown("<div class='quad-card' style='background: rgba(16, 185, 129, 0.2);'><b>⭐ LEAD MESSAGE</b><br><small>High Freq / High Pain / Triangulated</small></div>", unsafe_allow_html=True)
            for x in q_lead: 
                sig_badge = "🛡️ Sig" if x["significant"] else ""
                st.write(f"• **{x['code']['id']}** ({x['code']['name']}) {sig_badge}")

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("<div class='quad-card' style='background: rgba(51, 65, 85, 0.3);'><b>DROP FOR NOW</b><br><small>Low Freq / Low Pain</small></div>", unsafe_allow_html=True)
            for x in q_drop: st.write(f"• **{x['code']['id']}** ({x['code']['name']})")
            
        with g4:
            st.markdown("<div class='quad-card' style='background: rgba(99, 102, 241, 0.15);'><b>NICHE / BROAD CONCERN</b><br><small>High Freq / Low Pain</small></div>", unsafe_allow_html=True)
            for x in q_niche: st.write(f"• **{x['code']['id']}** ({x['code']['name']})")

        st.markdown("**Qualitative Mention Frequency (Interviews ≥ 25%) →**")

    st.markdown("---")
    st.subheader("Campaign Messaging Strategy Output")
    selected_matrix_code = st.selectbox("Select Code for Campaign Brief:", [x["code"]["id"] for x in matrix_data])
    target_item = next(x for x in matrix_data if x["code"]["id"] == selected_matrix_code)
    
    st.info(f"**Target Code:** {target_item['code']['id']} - {target_item['code']['name']}")
    st.write(f"**Primary Campaign Headline:** Stop Drowning in {target_item['code']['name']} — Build Your Thesis Plan Fast with Thought Sweep.")
    st.write(f"**Recommended Channel:** Barrett Honors College Weekly Digest & Academic Advisor Referrals")
