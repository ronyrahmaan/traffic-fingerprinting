"""Traffic Fingerprinting — Interactive Results Dashboard.

Displays results from Assignment 1: reproducing Kennedy et al. (CNS 2019)
on Alexa voice commands + original website traffic fingerprinting.

Author: Md A Rahman
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# --- Page config ---
st.set_page_config(
    page_title="Traffic Fingerprinting Results",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load data ---
BASE = Path(__file__).parent.parent

@st.cache_data
def load_alexa():
    with open(BASE / "results" / "alexa_dataset" / "expanded_results.json") as f:
        return json.load(f)

@st.cache_data
def load_website():
    with open(BASE / "results" / "website_eval" / "evaluation_results.json") as f:
        return json.load(f)

alexa = load_alexa()
website = load_website()

# --- Color palette ---
COLORS = {
    "LL-Jaccard": "#4C72B0",
    "LL-NB": "#DD8452",
    "VNG++": "#55A868",
    "AdaBoost": "#C44E52",
    "CUMUL+XGB": "#8172B3",
    "Timing+XGB": "#937860",
    "Combined+XGB": "#DA8BC3",
    "Combined+RF": "#8C8C8C",
}
ALEXA_COLOR = "#4C72B0"
WEBSITE_COLOR = "#DD8452"
PAPER_COLOR = "#55A868"

# --- Sidebar ---
st.sidebar.title("Traffic Fingerprinting")
st.sidebar.caption("Md A Rahman — March 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Alexa Reproduction",
        "Website Results",
        "Side-by-Side Comparison",
        "Per-Site Explorer",
        "Algorithm Deep Dive",
        "Dataset Statistics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Paper:** Kennedy et al., *I Can Hear Your Alexa*, IEEE CNS 2019"
)

# --- Helper functions ---
def metric_card(label, value, delta=None, delta_color="normal"):
    """Display a metric with optional delta."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def algorithm_bar_chart(data, title, show_paper=False, paper_data=None):
    """Create a bar chart of algorithm accuracies."""
    algos = list(data.keys())
    accs = [data[a]["accuracy"] for a in algos]

    fig = go.Figure()

    if show_paper and paper_data:
        paper_algos = [a for a in algos if a in paper_data]
        paper_accs = [paper_data[a] for a in paper_algos]
        fig.add_trace(go.Bar(
            x=paper_algos, y=paper_accs,
            name="Paper (Table I)",
            marker_color=PAPER_COLOR,
            text=[f"{v:.1f}%" for v in paper_accs],
            textposition="outside",
        ))

    fig.add_trace(go.Bar(
        x=algos, y=accs,
        name="My Results",
        marker_color=[COLORS.get(a, "#666") for a in algos],
        text=[f"{v:.1f}%" for v in accs],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Accuracy (%)",
        barmode="group",
        height=450,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =====================================================================
# PAGES
# =====================================================================

if page == "Overview":
    st.title("Traffic Fingerprinting: Reproduction and Extension")
    st.markdown(
        "I reproduced four classification algorithms from Kennedy et al. (CNS 2019) "
        "on their Alexa voice command dataset, added three improvements, then collected "
        "my own website traffic dataset and ran the same pipeline on it."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Alexa Accuracy", "39.2%", "Combined+RF")
    with col2:
        st.metric("Best Website Accuracy", "94.1%", "LL-Jaccard")
    with col3:
        st.metric("Cross-Domain Gap", "+76.5pp", "Websites >> Voice")

    st.markdown("---")

    # Quick comparison chart
    cross = website["cross_domain_comparison"]
    algos = list(cross.keys())
    web_accs = [cross[a]["website"] for a in algos]
    alexa_accs = [cross[a]["alexa"] for a in algos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=algos, y=alexa_accs, name="Alexa Voice Commands",
        marker_color=ALEXA_COLOR,
        text=[f"{v:.1f}%" for v in alexa_accs],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=algos, y=web_accs, name="Website Traffic",
        marker_color=WEBSITE_COLOR,
        text=[f"{v:.1f}%" for v in web_accs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Cross-Domain Comparison: All Algorithms",
        yaxis_title="Accuracy (%)",
        barmode="group",
        height=500,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Every algorithm does much better on website traffic than voice commands. "
        "Websites load thousands of packets with distinct page structures, "
        "giving classifiers much more signal to work with."
    )

    # Key findings
    st.markdown("### Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Alexa (reproducing the paper)**\n"
            "- Matched all 4 algorithms within 0.2–3.1pp\n"
            "- Combined+RF beats paper's best by +5.4pp\n"
            "- Combined+RF runs 160x faster than AdaBoost\n"
            "- 100 classes × 10 traces = very hard problem"
        )
    with col2:
        st.markdown(
            "**Website (my collection)**\n"
            "- 50 sites × 100 visits = 5,000 traces\n"
            "- LL-Jaccard hits 94.1% (simple set similarity)\n"
            "- E-commerce sites are easiest (99.4%)\n"
            "- Social media hardest (85.8%) — dynamic content"
        )

    st.markdown("---")

    # Methodology
    st.markdown("### Methodology")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Alexa Reproduction (Part 1a)**")
        st.markdown(
            "I used the original dataset from Kennedy et al. — 1,000 packet traces "
            "(100 voice commands × 10 captures each). Each trace is a sequence of "
            "(timestamp, packet_size, direction) tuples extracted from encrypted traffic. "
            "I implemented all four algorithms from scratch since the authors' GitHub repo "
            "was missing `tools.py` and `utils.py`. Evaluation uses 5-fold stratified "
            "cross-validation with seed=42."
        )
    with col2:
        st.markdown("**Website Collection (Part 1b)**")
        st.markdown(
            "I selected 50 websites across 5 categories (news, social media, e-commerce, "
            "tech, entertainment) and visited each 100 times using Playwright with headless "
            "Chromium. Traffic was captured with `tcpdump`, filtering by the browser's destination "
            "IPs. Each visit used a fresh browser context, flushed DNS cache, and waited 5 seconds "
            "after page load for async content. Visit order was fully randomized (seed=42)."
        )


elif page == "Alexa Reproduction":
    st.title("Part 1a: Alexa Voice Command Reproduction")
    st.markdown(
        'The paper "I Know What You Are Doing With Your Smart Speaker" by Kennedy, Thomas, and Wang '
        "(IEEE CNS 2019) tests whether encrypted traffic from an Amazon Echo can reveal which voice "
        "command a user spoke. They captured 1,000 network traces: 100 different voice commands with "
        "10 traces each. Then they ran four traffic fingerprinting algorithms and measured classification "
        "accuracy with 5-fold cross validation."
    )
    st.markdown(
        "My task was to reproduce Table I from the paper using their original dataset. "
        "Their GitHub repo was missing key files (`tools.py` and `utils.py`), so I wrote "
        "all the code from scratch based on the algorithm descriptions in the papers they cited."
    )

    # Algorithm descriptions
    with st.expander("How Each Algorithm Works", expanded=False):
        st.markdown(
            "**LL-Jaccard:** Represents each trace as a *set* of unique signed packet sizes. "
            "Classification uses Jaccard similarity — the overlap between two sets divided by their union. "
            "Simple, fast, and surprisingly effective for some domains.\n\n"
            "**LL-NB:** Represents each trace as a *histogram* of signed packet sizes, then runs "
            "Gaussian Naive Bayes. The histogram captures how often each packet size appears.\n\n"
            "**VNG++:** Extracts burst-level features: groups consecutive packets in the same direction, "
            "counts them, and builds a histogram of burst lengths. Also adds summary stats like total bytes "
            "and packet counts. Runs Gaussian NB on these features.\n\n"
            "**AdaBoost:** Uses the richest feature set: burst histograms plus marker features (unique sizes, "
            "percentiles, packet counts per direction). The original paper used SVM, but Kennedy et al. switched "
            "to AdaBoost with SAMME.R. Since scikit-learn 1.6+ removed SAMME.R, I used "
            "`HistGradientBoostingClassifier` as a drop-in replacement — this likely explains the 3.1pp gap."
        )

    # Paper vs Mine comparison
    paper_algos = ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]
    fig = algorithm_bar_chart(
        {a: alexa["algorithm_results"][a] for a in paper_algos},
        "Paper vs My Reproduction",
        show_paper=True,
        paper_data=alexa["paper_results"],
    )
    fig.update_layout(yaxis_range=[0, 42])
    st.plotly_chart(fig, use_container_width=True)

    # Difference table
    st.markdown("### Reproduction Accuracy")
    cols = st.columns(4)
    for i, algo in enumerate(paper_algos):
        mine = alexa["algorithm_results"][algo]["accuracy"]
        paper = alexa["paper_results"][algo]
        diff = mine - paper
        sign = "+" if diff >= 0 else ""
        with cols[i]:
            st.metric(algo, f"{mine:.1f}%", f"{sign}{diff:.1f}pp vs paper")

    st.markdown(
        "The first three algorithms match the paper within 1 percentage point. "
        "The AdaBoost gap is expected since I used a different boosting implementation. "
        "Overall, the reproduction confirms the paper's findings: voice command fingerprinting "
        "on Alexa traffic is hard, with the best algorithm reaching only ~34% accuracy across 100 classes."
    )

    st.markdown("---")

    # All algorithms including improvements
    st.markdown("### All Algorithms (Paper + My Improvements)")
    st.markdown(
        "After reproducing the paper's results, I wanted to see if modern feature engineering could do better. "
        "I designed three additional feature sets and tested them with XGBoost and Random Forest."
    )

    with st.expander("New Feature Sets I Added", expanded=False):
        st.markdown(
            "**CUMUL** (Panchenko et al., NDSS 2016): Computes the cumulative sum of signed packet sizes "
            "and samples it at 100 evenly-spaced points. This captures the *shape* of the traffic flow "
            "over time — histograms lose order information, but CUMUL preserves it.\n\n"
            "**Timing** (inspired by Rahman et al., PETS 2020): Inter-packet timing statistics — mean, std, "
            "min, max, and percentiles of time gaps. A fast burst of 10 packets in 0.01s looks different "
            "from 10 packets spread over 2 seconds.\n\n"
            "**Combined:** Everything together — CUMUL + timing + the statistical features from VNG++. "
            "179 features total. Throw everything in and let the classifier sort it out."
        )
    fig = algorithm_bar_chart(
        alexa["algorithm_results"],
        "All Algorithms on Alexa Dataset",
    )
    fig.update_layout(yaxis_range=[0, 48])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Combined+RF hits 39.2%, beating the paper's best result (LL-NB at 33.8%) by 5.4 percentage points. "
        "It also trains in 1.8 seconds compared to AdaBoost's 287.9 seconds — about 160x faster. "
        "Random Forest's bagging approach handles the high-dimensional, noisy feature space better than boosting here."
    )
    st.markdown(
        "Timing features alone (17.2%) barely beat random chance for 100 classes (1%), confirming that "
        "*when* packets arrive matters less than *what sizes* they are for encrypted traffic analysis. "
        "But timing helps when combined with size-based features."
    )

    st.markdown("---")

    # Per-fold variance
    st.markdown("### Per-Fold Accuracy (5-Fold CV)")
    selected_algo = st.selectbox(
        "Select algorithm:",
        list(alexa["algorithm_results"].keys()),
        key="alexa_fold_select",
    )
    folds = alexa["algorithm_results"][selected_algo]["fold_accuracies"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],
        y=folds,
        marker_color=COLORS.get(selected_algo, "#666"),
        text=[f"{v:.1f}%" for v in folds],
        textposition="outside",
    ))
    mean_acc = alexa["algorithm_results"][selected_algo]["accuracy"]
    fig.add_hline(y=mean_acc, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_acc:.1f}%")
    fig.update_layout(
        title=f"{selected_algo} — Per-Fold Accuracy",
        yaxis_title="Accuracy (%)",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bin width sweep
    if "bin_width_sweep" in alexa:
        st.markdown("### LL-NB Bin Width Sweep")
        st.markdown(
            "I tested bin widths from 10 to 150 to find the best setting. "
            "Bin width 60 gives 34.1%, matching the paper's 33.8%."
        )
        sweep = alexa["bin_width_sweep"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[s["bin_width"] for s in sweep],
            y=[s["accuracy"] for s in sweep],
            mode="lines+markers",
            marker=dict(size=8, color=ALEXA_COLOR),
            line=dict(width=2),
            text=[f"Bin={s['bin_width']}, Acc={s['accuracy']:.1f}%, Feat={s['features']}" for s in sweep],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.add_hline(y=33.8, line_dash="dash", line_color="green",
                      annotation_text="Paper: 33.8%")
        fig.update_layout(
            title="LL-NB Accuracy vs Bin Width",
            xaxis_title="Bin Width",
            yaxis_title="Accuracy (%)",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Website Results":
    st.title("Part 1b: Website Traffic Fingerprinting")
    st.markdown(
        "Since I don't have a smart speaker, I collected website traffic data instead. "
        "I selected 50 websites across 5 categories "
        "(news, social media, e-commerce, tech, entertainment) and visited each site 100 times, "
        "collecting ~5,000 traces total."
    )

    with st.expander("Data Collection Pipeline", expanded=False):
        st.markdown(
            "- **Browser:** Playwright (Chromium, headless) with a fresh browser context per visit\n"
            "- **Capture:** `tcpdump` on macOS, filtering the browser's traffic by destination IP\n"
            "- **DNS flush:** Cleared DNS cache between visits (`dscacheutil -flushcache`) so DNS traffic is captured each time\n"
            "- **Post-load wait:** 5 seconds after page load to capture async content\n"
            "- **Randomization:** Visit order fully randomized with seed=42 for reproducibility\n"
            "- **Verification:** Screenshots saved for each visit to confirm page loaded correctly\n"
            "- **Quality control:** Minimum 50 packets per valid trace (literature threshold)"
        )

    st.markdown("I ran all seven algorithms (four from the paper, three of mine) using the same 5-fold CV setup.")

    # Algorithm comparison
    fig = algorithm_bar_chart(
        website["algorithm_results"],
        "All Algorithms on Website Dataset",
    )
    fig.update_layout(yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    # Per-category accuracy
    st.markdown("### Accuracy by Category")
    cats = website["per_category_accuracy"]["categories"]
    cat_names = list(cats.keys())
    cat_accs = list(cats.values())

    fig = go.Figure()
    cat_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig.add_trace(go.Bar(
        x=[c.title() for c in cat_names],
        y=cat_accs,
        marker_color=cat_colors,
        text=[f"{v:.1f}%" for v in cat_accs],
        textposition="outside",
    ))
    fig.update_layout(
        title="LL-Jaccard Accuracy by Website Category",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 108],
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why e-commerce is easiest (99.4%)**")
        st.markdown(
            "Each e-commerce site loads a unique combination of product images, "
            "tracking scripts, and payment SDKs. Amazon looks nothing like Walmart "
            "at the packet level."
        )
    with col2:
        st.markdown("**Why social media is hardest (85.8%)**")
        st.markdown(
            "Social feeds change every visit. Reddit's front page is different "
            "every time, so the traffic pattern varies more between visits to "
            "the same site."
        )

    # Top and bottom sites
    st.markdown("### Best and Worst Sites (LL-Jaccard)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 (100% accuracy)**")
        for site, acc in website["per_site_accuracy"]["top_10"].items():
            st.markdown(f"- {site}: {acc:.0f}%")
    with col2:
        st.markdown("**Bottom 10**")
        for site, acc in website["per_site_accuracy"]["bottom_10"].items():
            st.markdown(f"- {site}: {acc:.0f}%")


elif page == "Side-by-Side Comparison":
    st.title("Side-by-Side: Alexa vs Website")
    st.markdown(
        "The gap between website and voice command fingerprinting is striking. "
        "Pick algorithms below to compare them across both datasets."
    )

    cross = website["cross_domain_comparison"]
    all_algos = list(cross.keys())

    selected = st.multiselect(
        "Select algorithms to compare:",
        all_algos,
        default=["LL-Jaccard", "LL-NB", "AdaBoost", "Combined+RF"],
    )

    if not selected:
        st.warning("Pick at least one algorithm.")
    else:
        # Grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=selected,
            y=[cross[a]["alexa"] for a in selected],
            name="Alexa (1,000 traces, 100 classes)",
            marker_color=ALEXA_COLOR,
            text=[f"{cross[a]['alexa']:.1f}%" for a in selected],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=selected,
            y=[cross[a]["website"] for a in selected],
            name="Website (5,000 traces, 50 classes)",
            marker_color=WEBSITE_COLOR,
            text=[f"{cross[a]['website']:.1f}%" for a in selected],
            textposition="outside",
        ))
        fig.update_layout(
            title="Accuracy: Alexa vs Website",
            yaxis_title="Accuracy (%)",
            barmode="group",
            height=500,
            template="plotly_white",
            font=dict(size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Difference table
        st.markdown("### Accuracy Difference (Website − Alexa)")
        cols = st.columns(min(len(selected), 4))
        for i, algo in enumerate(selected):
            with cols[i % 4]:
                diff = cross[algo]["difference"]
                st.metric(algo, f"+{diff:.1f}pp", f"Website wins")

        # Radar chart
        if len(selected) >= 3:
            st.markdown("### Radar View")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[cross[a]["alexa"] for a in selected] + [cross[selected[0]]["alexa"]],
                theta=selected + [selected[0]],
                fill="toself",
                name="Alexa",
                fillcolor="rgba(76, 114, 176, 0.2)",
                line_color=ALEXA_COLOR,
            ))
            fig.add_trace(go.Scatterpolar(
                r=[cross[a]["website"] for a in selected] + [cross[selected[0]]["website"]],
                theta=selected + [selected[0]],
                fill="toself",
                name="Website",
                fillcolor="rgba(221, 132, 82, 0.2)",
                line_color=WEBSITE_COLOR,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=500,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Explanation section
        st.markdown("---")
        st.markdown("### Why Websites Are So Much Easier to Fingerprint")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**More data per trace.** Website traces average 8,522 packets per visit. "
                "Voice command traces average about 619 packets. More packets means more information "
                "for the classifier to work with. A website loads dozens of resources (HTML, CSS, JS, "
                "images, fonts, API calls) while a voice command triggers a short encrypted exchange."
            )
            st.markdown(
                "**Distinctive content structure.** Each website has a unique set of resources. "
                "CNN loads different images, scripts, and ads than Amazon. These resources have "
                "different sizes, creating a distinctive fingerprint of packet sizes per site. Voice "
                "commands all go through the same Alexa API and return similar-sized encrypted responses."
            )
        with col2:
            st.markdown(
                "**More consistent across visits.** Websites are relatively static — the same page "
                "loads the same resources on each visit (modulo ads and dynamic content). Voice commands "
                "can have variable responses depending on context, time of day, or what Alexa decides to say."
            )
            st.markdown(
                "**Why LL-Jaccard wins on websites.** LL-Jaccard uses the *set* of unique packet sizes. "
                "On websites, this set is large and distinctive (hundreds of unique sizes from all the "
                "different resources). On voice commands, the set is small and overlapping. The Jaccard "
                "coefficient rewards distinctiveness, so it excels on websites."
            )


elif page == "Per-Site Explorer":
    st.title("Per-Site Accuracy Explorer")
    st.markdown("Drill into individual website accuracy with LL-Jaccard (best algorithm).")

    # All sites from top + bottom
    all_sites = {}
    all_sites.update(website["per_site_accuracy"]["top_10"])
    all_sites.update(website["per_site_accuracy"]["bottom_10"])

    # Sort by accuracy
    sorted_sites = dict(sorted(all_sites.items(), key=lambda x: x[1], reverse=True))

    # Horizontal bar chart of all available sites
    sites = list(sorted_sites.keys())
    accs = list(sorted_sites.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sites,
        x=accs,
        orientation="h",
        marker_color=[
            "#27ae60" if a >= 95 else "#f39c12" if a >= 80 else "#e74c3c"
            for a in accs
        ],
        text=[f"{v:.0f}%" for v in accs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Per-Site Accuracy (LL-Jaccard) — Top and Bottom Sites",
        xaxis_title="Accuracy (%)",
        xaxis_range=[0, 110],
        height=max(400, len(sites) * 28),
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Green** = 95%+, **Orange** = 80-95%, **Red** = below 80%. "
        "Reddit is the hardest site (52%) because its front page content changes "
        "constantly, making each visit's traffic pattern look different."
    )

    # Category breakdown
    st.markdown("### Category Breakdown")
    cats = website["per_category_accuracy"]["categories"]
    fig = go.Figure(go.Pie(
        labels=[c.title() for c in cats.keys()],
        values=list(cats.values()),
        hole=0.4,
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.1f}%",
        marker_colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"],
    ))
    fig.update_layout(
        title="Accuracy by Category (LL-Jaccard)",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Algorithm Deep Dive":
    st.title("Algorithm Deep Dive")

    dataset = st.radio(
        "Dataset:", ["Alexa", "Website"], horizontal=True, key="deep_dive_ds"
    )
    data = alexa if dataset == "Alexa" else website
    results = data["algorithm_results"]

    algo = st.selectbox(
        "Select algorithm:",
        list(results.keys()),
        key="deep_dive_algo",
    )

    r = results[algo]

    # Algorithm description
    algo_descriptions = {
        "LL-Jaccard": "Represents each trace as a **set** of unique signed packet sizes. "
            "Classification uses Jaccard similarity — the overlap between two sets divided by "
            "their union. No machine learning model needed, just set comparison.",
        "LL-NB": "Builds a **histogram** of signed packet sizes (binned at 60-byte intervals), "
            "then runs Gaussian Naive Bayes. The histogram captures how often each packet size appears. "
            "50 features.",
        "VNG++": "Groups consecutive same-direction packets into **bursts**, builds a histogram "
            "of burst sizes, then adds traffic stats (total bytes, duration). Runs Gaussian NB. "
            "403 features from the burst histogram + 3 stats.",
        "AdaBoost": "Uses the richest feature set from the paper: burst histograms plus marker "
            "features (unique sizes, percentiles, packet counts). I used `HistGradientBoostingClassifier` "
            "since scikit-learn 1.6+ removed the original SAMME.R. 205 features.",
        "CUMUL+XGB": "Cumulative sum of signed packet sizes, sampled at 100 evenly-spaced points "
            "(Panchenko et al., NDSS 2016). Captures the **shape** of the traffic flow over time. "
            "104 features. Paired with XGBoost.",
        "Timing+XGB": "Inter-packet timing statistics: mean, std, percentiles of time gaps, "
            "plus burst-level timing and a log-scaled timing histogram. Captures the **rhythm** "
            "of the traffic. 70 features. Paired with XGBoost.",
        "Combined+XGB": "Everything combined: CUMUL + timing + burst stats from VNG++. "
            "179 features total. The kitchen-sink approach, paired with XGBoost.",
        "Combined+RF": "Same 179 combined features, but paired with Random Forest instead of "
            "XGBoost. RF's bagging approach handles noisy features better than boosting for this task.",
    }
    if algo in algo_descriptions:
        st.info(algo_descriptions[algo])

    # Metrics row
    cols = st.columns(4)
    with cols[0]:
        st.metric("Accuracy", f"{r['accuracy']:.1f}%")
    with cols[1]:
        folds = r["fold_accuracies"]
        std = (sum((f - r["accuracy"])**2 for f in folds) / len(folds)) ** 0.5
        st.metric("Std Dev", f"{std:.1f}%")
    with cols[2]:
        st.metric("Features", r.get("features", "N/A (set-based)"))
    with cols[3]:
        st.metric("Runtime", f"{r['time_sec']:.1f}s")

    # Fold chart
    st.markdown("### Per-Fold Accuracy")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],
        y=folds,
        marker_color=COLORS.get(algo, "#666"),
        text=[f"{v:.1f}%" for v in folds],
        textposition="outside",
    ))
    fig.add_hline(y=r["accuracy"], line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {r['accuracy']:.1f}%")
    fig.update_layout(
        yaxis_title="Accuracy (%)",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-class accuracy (Alexa only)
    if dataset == "Alexa" and algo in ("LL-NB", "Combined+RF"):
        key = "per_class_accuracy_llnb" if algo == "LL-NB" else "per_class_accuracy_best"
        if key in alexa:
            st.markdown(f"### Per-Command Accuracy ({algo})")
            per_class = alexa[key]["all"]
            sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)

            names = [c[0].replace("_", " ") for c in sorted_classes]
            vals = [c[1] for c in sorted_classes]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=names[:20],
                x=vals[:20],
                orientation="h",
                marker_color="#27ae60",
                text=[f"{v:.0f}%" for v in vals[:20]],
                textposition="outside",
                name="Top 20",
            ))
            fig.update_layout(
                title=f"Top 20 Voice Commands ({algo})",
                xaxis_title="Accuracy (%)",
                height=600,
                template="plotly_white",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bottom 20
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=names[-20:],
                x=vals[-20:],
                orientation="h",
                marker_color="#e74c3c",
                text=[f"{v:.0f}%" for v in vals[-20:]],
                textposition="outside",
                name="Bottom 20",
            ))
            fig2.update_layout(
                title=f"Bottom 20 Voice Commands ({algo})",
                xaxis_title="Accuracy (%)",
                height=600,
                template="plotly_white",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Confusion pairs (Alexa LL-NB only)
    if dataset == "Alexa" and algo == "LL-NB" and "confusion_top_pairs_llnb" in alexa:
        st.markdown("### Most Confused Command Pairs")
        pairs = alexa["confusion_top_pairs_llnb"][:10]
        for p in pairs:
            true_cmd = p["true"].replace("_", " ")
            pred_cmd = p["predicted"].replace("_", " ")
            st.markdown(f"- **{true_cmd}** → misclassified as **{pred_cmd}** ({p['count']}x)")


elif page == "Dataset Statistics":
    st.title("Dataset Statistics")

    dataset = st.radio(
        "Dataset:", ["Alexa", "Website", "Compare"], horizontal=True, key="stats_ds"
    )

    if dataset == "Compare":
        a_stats = alexa["dataset_stats"]
        w_stats = website["dataset_stats"]

        st.markdown("### Side-by-Side Dataset Comparison")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Alexa Voice Commands", "Website Traffic"),
            specs=[[{"type": "domain"}, {"type": "domain"}]],
        )

        # Summary table
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Alexa Voice Commands**")
            st.markdown(f"- Traces: {a_stats['num_traces']:,}")
            st.markdown(f"- Classes: {a_stats['num_classes']}")
            st.markdown(f"- Avg packets/trace: {a_stats['packets']['mean']:.0f}")
            st.markdown(f"- Avg duration: {a_stats['duration']['mean']:.1f}s")
            st.markdown(f"- Avg bytes: {a_stats['total_bytes']['mean']:,.0f}")
            st.markdown(f"- Avg bursts: {a_stats['bursts']['mean']:.0f}")
        with col2:
            st.markdown("**Website Traffic**")
            st.markdown(f"- Traces: {w_stats['num_traces']:,}")
            st.markdown(f"- Classes: {w_stats['num_classes']}")
            st.markdown(f"- Avg packets/trace: {w_stats['packets']['mean']:,.0f}")
            st.markdown(f"- Avg duration: {w_stats['duration']['mean']:.1f}s")
            st.markdown(f"- Avg bytes: {w_stats['total_bytes']['mean']:,.0f}")
            st.markdown(f"- Avg bursts: {w_stats['bursts']['mean']:,.0f}")

        # Comparison bar chart
        metrics = ["Avg Packets", "Avg Bursts", "Avg Duration (s)"]
        alexa_vals = [a_stats["packets"]["mean"], a_stats["bursts"]["mean"], a_stats["duration"]["mean"]]
        web_vals = [w_stats["packets"]["mean"], w_stats["bursts"]["mean"], w_stats["duration"]["mean"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics, y=alexa_vals, name="Alexa", marker_color=ALEXA_COLOR))
        fig.add_trace(go.Bar(x=metrics, y=web_vals, name="Website", marker_color=WEBSITE_COLOR))
        fig.update_layout(
            title="Dataset Scale Comparison",
            barmode="group",
            height=400,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "Website traces are ~14x larger in packet count and ~5x longer in duration. "
            "This extra data gives classifiers much more signal, which explains why "
            "website fingerprinting is so much easier than voice command fingerprinting."
        )

    else:
        data = alexa if dataset == "Alexa" else website
        stats = data["dataset_stats"]

        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Traces", f"{stats['num_traces']:,}")
        with cols[1]:
            st.metric("Classes", stats["num_classes"])
        with cols[2]:
            st.metric("Avg Packets", f"{stats['packets']['mean']:,.0f}")
        with cols[3]:
            st.metric("Avg Duration", f"{stats['duration']['mean']:.1f}s")

        st.markdown("### Packet Count Distribution")
        st.markdown(
            f"- **Mean:** {stats['packets']['mean']:,.0f}\n"
            f"- **Median:** {stats['packets']['median']:,}\n"
            f"- **Min:** {stats['packets']['min']:,}\n"
            f"- **Max:** {stats['packets']['max']:,}\n"
            f"- **Std Dev:** {stats['packets']['std']:,.1f}"
        )

        # Trace example (Alexa only)
        if dataset == "Alexa" and "trace_example" in alexa:
            st.markdown("### Example Trace")
            ex = alexa["trace_example"]
            st.markdown(
                f"**Command:** `{ex['label']}` | "
                f"**Packets:** {ex['num_packets']} | "
                f"**Duration:** {ex['duration']:.3f}s"
            )

            pkts = ex["packets"]
            times = [p["time"] for p in pkts]
            sizes = [p["size"] * p["direction"] for p in pkts]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=sizes,
                mode="markers",
                marker=dict(
                    size=4,
                    color=["#4C72B0" if s > 0 else "#e74c3c" for s in sizes],
                ),
                hovertemplate="Time: %{x:.4f}s<br>Size: %{y} bytes<extra></extra>",
            ))
            fig.update_layout(
                title=f'Packet Timeline: "{ex["label"]}"',
                xaxis_title="Time (seconds)",
                yaxis_title="Signed Packet Size (bytes)",
                height=400,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "Blue dots = outgoing (device → cloud), "
                "Red dots = incoming (cloud → device)."
            )


# --- Footer ---
st.markdown("---")
st.caption(
    "Md A Rahman · RA Skills Assessment · DSMI Lab, OU Polytechnic Institute · March 2026"
)
