"""Local Streamlit UI. Run: python -m streamlit run app.py"""
import hashlib
from pathlib import Path
import tempfile
import streamlit as st

from config import ROOT, METHODS, DetectionSettings
from detector import detect
from detector.preprocessing import load_rgb
from detector.output import annotate, png_bytes, results_json, results_csv, flat_rows
from icon_manager import ProfileManager
from icon_manager.icon_registry import IconRegistry
from icon_manager.variant_manager import remove_variant

st.set_page_config(page_title="Universal Classical Icon Finder", page_icon="⌕", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:1440px}
h1{letter-spacing:-1.4px;font-size:2.5rem!important;font-weight:650!important}
h2{letter-spacing:-.5px} .eyebrow{font-size:11px;letter-spacing:2px;color:#37756b;font-weight:700}
.intro{color:#647084;font-size:16px;max-width:780px;margin-bottom:24px}
[data-testid="stMetric"]{background:#fff;border:1px solid #e5e9ef;border-radius:12px;padding:16px 20px}
[data-testid="stSidebar"]{border-right:1px solid #e5e9ef}
.stButton button,.stDownloadButton button{border-radius:8px}
</style>""", unsafe_allow_html=True)
manager = ProfileManager()
registry = IconRegistry(manager)
METHOD_LABELS = {"template_matching": "Template", "edge_matching": "Edges",
                 "shape_matching": "Contours", "feature_matching": "Features"}

with st.sidebar:
    st.markdown("### ◈ Icon Finder")
    st.caption("CLASSICAL VISION · LOCAL PROCESSING")
    profiles = manager.list_profiles()
    profile_name = st.selectbox("Platform profile", profiles, index=profiles.index("generic") if "generic" in profiles else 0,
                                format_func=lambda name: "Unknown / Generic Platform" if name == "generic" else name.replace("_", " ").title())
    try:
        profile = manager.load(profile_name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(profile.description)
    st.divider()
    st.markdown("#### Detection settings")
    methods = st.multiselect("Methods", METHODS, default=["template_matching", "edge_matching"],
                              format_func=lambda m: METHOD_LABELS[m])
    threshold = st.slider("Minimum match score", 0.50, 1.0, 0.86, 0.01)
    st.caption("Similarity score; it is not a calibrated probability.")
    theme = st.selectbox("Screenshot theme", ["auto", "light", "dark"], format_func=str.title)
    scale = st.slider("Scale range", 0.25, 3.0, (0.75, 1.50), 0.125)
    with st.expander("Advanced settings"):
        step = st.number_input("Scale step", min_value=0.025, max_value=1.0, value=0.125, step=0.025)
        iou = st.slider("Overlap / IoU threshold", 0.05, 0.95, 0.35, 0.05)
        match_method = st.selectbox("Normalized template method", ["CCOEFF_NORMED", "CCORR_NORMED", "SQDIFF_NORMED"])
        feature = st.selectbox("Feature detector", ["SIFT", "ORB", "AKAZE"])
        priority = st.checkbox("Prioritize a region")
        priority_region = None
        if priority:
            st.caption("Searches the entire image; results in this region appear first.")
            p1, p2 = st.columns(2)
            x = p1.number_input("Region X", 0, value=0)
            y = p2.number_input("Region Y", 0, value=0)
            w = p1.number_input("Region width", 1, value=300)
            h = p2.number_input("Region height", 1, value=200)
            priority_region = (x, y, w, h)
    st.divider()
    st.caption("No models, AI APIs, or fixed screen coordinates. After installation, detection works offline.")

st.markdown('<div class="eyebrow">UNIVERSAL CLASSICAL ICON FINDER</div>', unsafe_allow_html=True)
st.title("Find familiar shapes. Anywhere.")
st.markdown('<div class="intro">Search a screenshot for your reference icons. Compare shapes across scales, inspect every match, and export exact pixel coordinates.</div>', unsafe_allow_html=True)
find_tab, library_tab, profiles_tab = st.tabs(["Find icons", "Reference library", "Manage profiles"])

with find_tab:
    left, right = st.columns([1.0, 1.8], gap="large")
    with left:
        st.subheader("01  Choose a screenshot")
        upload = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg", "webp", "bmp"], label_visibility="collapsed")
        demo = st.selectbox("Or try an example", ["None", "Desktop · light", "Mobile · dark"])
        screenshot = upload.getvalue() if upload else None
        if screenshot is None and demo != "None":
            screenshot = (ROOT / "sample_data" / ("desktop_light.png" if demo.startswith("Desktop") else "mobile_dark.png")).read_bytes()
        st.subheader("02  Select reference icons")
        categories = sorted({i.get("category", "custom") for i in profile.icons})
        category = st.selectbox("Category", ["All categories"] + categories)
        available = [i["name"] for i in profile.icons if category == "All categories" or i.get("category", "custom") == category]
        defaults = [name for name in ("play", "pause", "volume", "volume_muted", "favorite", "fullscreen_corners", "play_bar_02", "play_bar_03", "play_bar_04") if name in available]
        selected = st.multiselect("Icons to search", available, default=defaults, key=f"selected_{profile_name}_{category}")
        if not available:
            st.info("This profile is empty. Add an icon in Reference library.")
        with st.expander("Add references for this search"):
            custom = st.file_uploader("Custom icon images", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True,
                                      help="One icon per tight crop. Uploaded names become icon identities for this search only.")
        run = st.button("Find icons →", type="primary", use_container_width=True,
                        disabled=screenshot is None or (not selected and not custom))
        st.caption("The complete screenshot is searched. Coordinates always refer to the original image.")
    with right:
        st.subheader("03  Inspect matches")
        if screenshot is None:
            st.info("Upload a screenshot or choose an example to begin.")
            st.image(str(ROOT / "sample_data" / "reference_library.png"), caption="Your starter set · Frames 1–5 and all four play bars", use_container_width=True)
        else:
            try:
                rgb = load_rgb(screenshot)
            except ValueError as exc:
                st.error(str(exc))
                rgb = None
            if rgb is not None:
                fingerprint = hashlib.sha256(screenshot).hexdigest()
                signature = (fingerprint, profile_name, tuple(selected), tuple(methods), threshold, theme,
                             scale, step, iou, match_method, feature, priority_region,
                             tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in custom),
                             hashlib.sha256(str(profile.metadata).encode()).hexdigest())
                if run:
                    try:
                        settings = DetectionSettings(tuple(methods), threshold, scale[0], scale[1], step,
                                                     iou, theme, match_method, feature, priority_region=priority_region)
                        with st.spinner("Searching scales and combining matches…"):
                            report = detect(rgb, profile, selected, settings) if selected else None
                            if custom:
                                with tempfile.TemporaryDirectory(prefix="icon_refs_") as temp_dir:
                                    temporary_manager = ProfileManager(Path(temp_dir))
                                    temporary_manager.create("uploaded_references")
                                    temporary_registry = IconRegistry(temporary_manager)
                                    for item in custom:
                                        temporary_registry.add("uploaded_references", Path(item.name).stem, item.getvalue())
                                    extra = detect(rgb, temporary_manager.load("uploaded_references"), settings=settings)
                                if report is None:
                                    report = extra
                                else:
                                    report.results.extend(extra.results)
                                    report.warnings.extend(extra.warnings)
                                    report.evaluated_variants += extra.evaluated_variants
                                    for key, value in extra.timing_ms.items():
                                        report.timing_ms[key] = round(report.timing_ms.get(key, 0) + value, 3)
                            result_image = png_bytes(annotate(rgb, report.results))
                            st.session_state["detection"] = (signature, report, result_image)
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                previous = st.session_state.get("detection")
                if previous and previous[0] == signature:
                    _, report, result_image = previous
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Matches", len(report.results))
                    c2.metric("Search time", f"{report.timing_ms['total']/1000:.2f} s")
                    c3.metric("Original size", f"{rgb.shape[1]} × {rgb.shape[0]}")
                    st.image(result_image, use_container_width=True)
                    if not report.results:
                        st.info("No matches above the current score. Try a tighter reference crop, a wider scale range, or a lower threshold.")
                    st.dataframe(flat_rows(report.results), use_container_width=True, hide_index=True)
                    d1, d2, d3 = st.columns(3)
                    d1.download_button("↓ JSON", results_json(report.results), "icon_results.json", "application/json", use_container_width=True)
                    d2.download_button("↓ CSV", results_csv(report.results), "icon_results.csv", "text/csv", use_container_width=True)
                    d3.download_button("↓ Image", result_image, "annotated.png", "image/png", use_container_width=True)
                    with st.expander("Method timings and notes"):
                        st.dataframe([{"stage": k, "milliseconds": v} for k, v in report.timing_ms.items()], hide_index=True)
                        st.caption(f"{report.evaluated_variants} reference variants evaluated. Candidate time is the elapsed method call for that variant, shared by its candidates.")
                        for warning in dict.fromkeys(report.warnings):
                            st.caption(warning)
                else:
                    st.image(rgb, caption=f"Original screenshot · {rgb.shape[1]} × {rgb.shape[0]} pixels", use_container_width=True)
                    if previous:
                        st.caption("Inputs changed. Run Find icons again to update results.")

with library_tab:
    st.subheader("Your references, organized")
    st.caption("The starter shapes are manually recreated from the chat preview. Replace them with exact icon crops for best matching. YouTube and Netflix profiles contain example artwork.")
    with st.expander("Add an icon or visual variant", expanded=not bool(profile.icons)):
        with st.form("add_icon"):
            a, b, c = st.columns(3)
            icon_name = a.text_input("Icon identity", placeholder="play")
            icon_category = b.text_input("Category", value="custom")
            aliases = c.text_input("Aliases (comma separated)")
            theme_variant = a.selectbox("Variant theme", ["all", "light", "dark"])
            state_variant = b.selectbox("Variant state", ["default", "hover", "active", "disabled"])
            preserve_color = c.checkbox("Preserve reference color", help="Useful for distinguishing colored play bars. Leave off for icons that change color between themes.")
            files = st.file_uploader("Reference files", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True)
            submit_icon = st.form_submit_button("Save to this profile")
            if submit_icon:
                try:
                    if not files or not icon_name.strip():
                        raise ValueError("Enter an icon identity and choose at least one image.")
                    for file in files:
                        registry.add(profile_name, icon_name, file.getvalue(), icon_category,
                                     [a.strip() for a in aliases.split(",") if a.strip()], theme_variant, state_variant, preserve_color)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
    for category_name in dict.fromkeys(i.get("category", "custom") for i in profile.icons):
        st.markdown(f"#### {category_name}")
        entries = [i for i in profile.icons if i.get("category", "custom") == category_name]
        for start in range(0, len(entries), 4):
            for column, icon in zip(st.columns(4), entries[start:start+4]):
                with column:
                    with st.container(border=True):
                        st.image(str(profile.directory / icon["templates"][0]), width=min(168, 72 if not icon["name"].startswith("play_bar") else 168))
                        st.markdown(f"**{icon['name']}**")
                        st.caption(f"{len(icon['templates'])} variants · {', '.join(icon.get('aliases', [])) or 'No aliases'}")
    if profile.icons:
        st.divider()
        st.subheader("Edit an identity")
        edit_name = st.selectbox("Icon", [i["name"] for i in profile.icons])
        edit_icon = next(i for i in profile.icons if i["name"] == edit_name)
        with st.form(f"edit_{profile_name}_{edit_name}"):
            e1, e2, e3 = st.columns(3)
            new_name = e1.text_input("Name", value=edit_name)
            new_aliases = e2.text_input("Aliases", value=", ".join(edit_icon.get("aliases", [])))
            new_category = e3.text_input("Category", value=edit_icon.get("category", "custom"))
            edit_color = st.checkbox("Preserve reference color", value=edit_icon.get("color_sensitive", False))
            if st.form_submit_button("Save identity"):
                try:
                    registry.update(profile_name, edit_name, new_name, [x.strip() for x in new_aliases.split(",") if x.strip()], new_category, edit_color)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        v1, v2 = st.columns([2, 1])
        with v1:
            variant = st.selectbox("Reference variant", edit_icon["templates"])
            st.image(str(profile.directory / variant), width=min(240, 96 if "play_bar" not in edit_name else 240))
            st.caption(str(edit_icon.get("variant_metadata", {}).get(variant, {})))
            if st.button("Remove selected variant", disabled=len(edit_icon["templates"]) <= 1):
                remove_variant(manager, profile_name, edit_name, variant)
                st.rerun()
        with v2:
            st.caption("Remove this identity from the current profile. Other profiles keep their copies.")
            if st.button("Remove icon"):
                registry.remove(profile_name, edit_name)
                st.rerun()

with profiles_tab:
    st.subheader("A profile for every platform")
    p1, p2 = st.columns(2, gap="large")
    with p1:
        with st.form("create_profile"):
            st.markdown("#### Create profile")
            new_profile = st.text_input("Profile name", placeholder="my_media_app")
            description = st.text_input("Description")
            clone = st.selectbox("Start from", ["Empty profile"] + profiles)
            if st.form_submit_button("Create profile"):
                try:
                    manager.create(new_profile, description, None if clone == "Empty profile" else clone)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        st.markdown("#### Export current profile")
        st.download_button("↓ Download profile ZIP", manager.export_zip(profile_name), f"{profile_name}.zip", "application/zip")
    with p2:
        with st.form("import_profile"):
            st.markdown("#### Import profile")
            archive = st.file_uploader("Profile ZIP", type=["zip"])
            import_name = st.text_input("Import as (optional)")
            if st.form_submit_button("Import profile"):
                try:
                    if archive is None:
                        raise ValueError("Choose a profile ZIP first.")
                    manager.import_zip(archive.getvalue(), import_name or None)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        st.markdown("#### Reuse an icon")
        source_name = st.selectbox("Source profile", profiles)
        source_profile = manager.load(source_name)
        with st.form("copy_icon"):
            source_icon = st.selectbox("Source icon", [i["name"] for i in source_profile.icons])
            st.caption(f"Copy into {profile_name}, including its variants and aliases.")
            if st.form_submit_button("Copy icon", disabled=not bool(source_profile.icons) or source_name == profile_name):
                try:
                    registry.copy(source_name, profile_name, source_icon)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
