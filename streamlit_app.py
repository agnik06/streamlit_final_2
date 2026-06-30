import streamlit as st
import os
import base64
import tempfile
import cv2
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Import the EV-2 generation functions from the EXISTING (untouched) script.
# The filename has a hyphen, so we load it via importlib.
# Path is resolved relative to THIS file (so the two files just need to live in
# the same folder), with the original absolute path kept as a fallback.
# -----------------------------------------------------------------------------
import importlib.util

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_EV2_CANDIDATES = [
    os.path.join(APP_DIR, "ev-2_images_generation.py")
                  ]

EV2_SCRIPT = next((p for p in _EV2_CANDIDATES if os.path.exists(p)), _EV2_CANDIDATES[0])

spec = importlib.util.spec_from_file_location("ev2_generator", EV2_SCRIPT)
ev2_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev2_module)
convert_single_image_to_ev2 = ev2_module.convert_single_image_to_ev2
synth_ev_minus2 = ev2_module.synth_ev_minus2

# OpenAI SDK is imported lazily inside the generation function so the rest of the
# app still loads even if the package isn't installed yet.

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV-2 Image Generator",
    page_icon="📸",
    layout="wide"
)

# Session state defaults
st.session_state.setdefault("page", "home")
st.session_state.setdefault("selected_category", None)
st.session_state.setdefault("ev0_bytes", None)        # generated/uploaded EV0 (raw image bytes)
st.session_state.setdefault("ev0_source", None)       # "openai" or "upload"

# -----------------------------------------------------------------------------
# EV0 generation prompts — one entry per category.
# Each prompt asks for a realistic, correctly-exposed (EV0) reference photo and
# encodes the inclusion / exclusion rules you described.
#
# NOTE: 7 categories. The last one is a COMPOSITE (white neon text + colored
# non-text light present together). Add or edit categories by editing this dict.
# -----------------------------------------------------------------------------
BASE_PROMPT = (
    "A realistic, correctly exposed reference photograph (EV0 exposure, balanced "
    "midtones, no globally crushed shadows, sharp focus, natural color, "
    "photographic detail, no watermark, no caption text overlay). "
)

CATEGORY_PROMPTS = {
    "Neon Light – Colored": (
        BASE_PROMPT
        + "A nighttime urban scene whose dominant light source is light-emitting "
        "COLORED neon — glowing colored neon text and/or a colored neon logo "
        "(magenta, cyan, electric blue, orange, etc.) on a storefront or wall. "
        "Vivid colored glow and reflections on nearby surfaces."
    ),
    "Neon Light – White": (
        BASE_PROMPT
        + "A nighttime scene whose light source is light-emitting WHITE neon text "
        "or signage (clean cool-white glowing neon letters). All light is white. "
        "Do NOT include any colored neon, colored bulbs, or any colored light source."
    ),
    "Non Neon Light – Colored": (
        BASE_PROMPT
        + "A nighttime scene illuminated by COLORFUL non-text light sources — "
        "colored building facade lights, colored bulbs, or colored spotlights. "
        "Do NOT include any text, letters, words, or signage of any kind. "
        "Colorful glowing light sources only, no neon text."
    ),
    "Non Neon Light – White": (
        BASE_PROMPT
        + "A nighttime scene illuminated by WHITE non-text light sources — white "
        "bulbs, white street lamps, white ceiling lights, or white floodlights. "
        "Do NOT include neon text or any colored light; all sources are white and "
        "non-text."
    ),
    "Sun / Sunrise / Sunset": (
        BASE_PROMPT
        + "An outdoor landscape with the SUN clearly visible in the sky — a sunrise "
        "or sunset (or bright daytime sun). The sun appears as a bright clipped orb "
        "with a natural sky gradient and warm directional light. High dynamic range."
    ),
    "No Lights": (
        BASE_PROMPT
        + "A scene with ordinary NON-light-emitting objects and backgrounds only "
        "(furniture, walls, landscape, everyday objects) under flat ambient light. "
        "There is NO visible light source of any kind — no lamps, bulbs, neon, sun, "
        "or glowing elements."
    ),
    # Composite category: BOTH white neon text AND colored non-text light present.
    "Neon White + Non-Neon Colored": (
        BASE_PROMPT
        + "A nighttime urban scene that clearly contains BOTH of the following "
        "together: (1) light-emitting WHITE neon TEXT/signage (clean cool-white "
        "glowing neon letters), AND (2) COLORFUL non-text light sources elsewhere in "
        "the scene — colored building facade lights, colored bulbs, or colored "
        "spotlights. The ONLY text/lettering in the image is the WHITE neon; the "
        "colored light sources must be non-text (no colored letters, words, or "
        "signage). Both elements are clearly visible in the same frame."
    ),
}

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def generate_ev0_with_openai(api_key, model, prompt, size, quality):
    """Call OpenAI's image API and return the generated EV0 image as PNG bytes."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)
    kwargs = dict(model=model, prompt=prompt, size=size, n=1)
    # gpt-image models accept quality low/medium/high; skip it for "auto".
    if quality and quality != "auto":
        kwargs["quality"] = quality

    resp = client.images.generate(**kwargs)
    data = resp.data[0]
    # gpt-image models return base64 by default.
    if getattr(data, "b64_json", None):
        return base64.b64decode(data.b64_json)
    # Fallback for models that return a URL (e.g. dall-e-3).
    if getattr(data, "url", None):
        import urllib.request
        with urllib.request.urlopen(data.url) as r:
            return r.read()
    raise RuntimeError("API returned no image data (no b64_json or url).")


def synthesize_ev2(ev0_image_bytes, add_noise=True, seed=0, base_name="image"):
    """
    Run the EXISTING EV-2 script on an EV0 image given as raw bytes.

    Writes the EV0 bytes to a temp file (because convert_single_image_to_ev2
    expects a path), then returns:
        (ev0_rgb, ev2_rgb, ev2_png_bytes)  or  None on failure.
    
    Output file is saved as: /home/test/agnik/Streamlit_Final_Demo/{base_name}_ev-2.png
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(ev0_image_bytes)
        in_path = tmp_in.name
    
    # Construct output path with proper filename format
    output_dir = "/home/test/agnik/Streamlit_Final_Demo"
    out_path = os.path.join(output_dir, f"{base_name}_ev-2.png")

    try:
        ev2_bgr = convert_single_image_to_ev2(
            in_path, output_path=out_path, add_noise=add_noise, seed=seed
        )
        if ev2_bgr is None:
            return None

        ev0_bgr = cv2.imread(in_path)            # decode whatever was written (jpg/png/bmp)
        if ev0_bgr is None:
            return None

        ev0_rgb = cv2.cvtColor(ev0_bgr, cv2.COLOR_BGR2RGB)
        ev2_rgb = cv2.cvtColor(ev2_bgr, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", ev2_bgr)
        ev2_png = buf.tobytes() if ok else None
        return ev0_rgb, ev2_rgb, ev2_png
    finally:
        # Only clean up the temp input file, not the output file (user may want to keep it)
        if os.path.exists(in_path):
            os.unlink(in_path)


def show_result(ev0_rgb, ev2_rgb, ev2_png, base_name="image"):
    """Render the EV0 / EV-2 comparison plus download buttons."""
    st.markdown("### 📊 Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ev0_rgb, caption="📷 EV0 Image", use_container_width=True)
    with col2:
        st.image(ev2_rgb, caption="🌙 Generated EV-2 Image", use_container_width=True)

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        ok, ev0_buf = cv2.imencode(".png", cv2.cvtColor(ev0_rgb, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button(
                "📥 Download EV0 Image",
                data=ev0_buf.tobytes(),
                file_name=f"{base_name}_ev0.png",
                mime="image/png",
                use_container_width=True,
            )
    with dcol2:
        if ev2_png is not None:
            st.download_button(
                "📥 Download EV-2 Image",
                data=ev2_png,
                file_name=f"{base_name}_ev-2.png",
                mime="image/png",
                use_container_width=True,
            )
    st.success("✅ EV-2 image generated successfully!")


# -----------------------------------------------------------------------------
# Page: Home
# -----------------------------------------------------------------------------
def show_home_page():
    st.title("📸 EV-2 Image Generator")
    st.markdown("---")
    st.markdown(
        """
        ## Welcome!

        Produce **EV0 → EV-2** image pairs two ways:

        - 🤖 **Generate an EV0 with ChatGPT** — pick one of the lighting categories and
          the OpenAI image API creates a base EV0 for you.
        - 📤 **Upload your own EV0** — bring an existing EV0 image.

        Either way, your EV-2 is then produced by the **same mathematical script**
        (linear-light 2-stop reduction, highlight reconstruction, bloom, realistic noise).
        """
    )
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Open the Generator", use_container_width=True, type="primary"):
            st.session_state.page = "studio"
            st.rerun()


# -----------------------------------------------------------------------------
# Page: Studio (Generate with ChatGPT  OR  Upload)
# -----------------------------------------------------------------------------
def sidebar_controls():
    """Shared sidebar: OpenAI settings + EV-2 options. Returns a settings dict."""
    st.sidebar.header("🔑 ChatGPT (OpenAI) settings")
    api_key = st.sidebar.text_input(
        "OpenAI API key", type="password",
        help="Your key is used only for the API call and is not stored.",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )
    model = st.sidebar.selectbox(
        "Image model", ["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2"],
        index=0,
        help="GPT image models may require Organization Verification in your OpenAI console.",
    )
    quality = st.sidebar.selectbox("Image quality", ["low", "medium", "high"], index=1)
    size = st.sidebar.selectbox(
        "Image size", ["1024x1024", "1536x1024", "1024x1536", "auto"], index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ EV-2 options")
    add_noise = st.sidebar.checkbox("Add realistic noise", value=True)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10000, value=0)

    return dict(api_key=api_key, model=model, quality=quality, size=size,
                add_noise=add_noise, seed=seed)


def generate_tab(cfg):
    st.subheader("🤖 Generate EV0 with ChatGPT")
    st.caption("1) Pick a category  →  2) (optional) tweak the prompt  →  3) Generate")

    # --- Category buttons (your requested button-click selection) ---
    st.markdown("**Choose a category:**")
    categories = list(CATEGORY_PROMPTS.keys())
    cols = st.columns(3)
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            is_sel = st.session_state.selected_category == cat
            if st.button(
                ("✅ " if is_sel else "") + cat,
                key=f"cat_{i}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state.selected_category = cat
                st.rerun()

    cat = st.session_state.selected_category
    if not cat:
        st.info("👆 Select a category to continue.")
        return

    st.markdown(f"**Selected:** `{cat}`")

    # Optional extra scene detail + editable prompt preview
    scene_hint = st.text_input(
        "Optional scene detail (added to the prompt)",
        placeholder="e.g. a narrow alley with a noodle shop",
    )
    default_prompt = CATEGORY_PROMPTS[cat]
    if scene_hint.strip():
        default_prompt = default_prompt + f" Scene: {scene_hint.strip()}."
    prompt = st.text_area("Prompt sent to the API (editable)", value=default_prompt, height=160)

    if st.button("🎨 Generate EV0  →  EV-2", type="primary", use_container_width=True):
        if not cfg["api_key"]:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
        try:
            with st.spinner("Generating EV0 image with ChatGPT..."):
                ev0_bytes = generate_ev0_with_openai(
                    cfg["api_key"], cfg["model"], prompt, cfg["size"], cfg["quality"]
                )
            st.session_state.ev0_bytes = ev0_bytes
            st.session_state.ev0_source = "openai"
        except Exception as e:
            st.error(f"❌ Image generation failed: {e}")
            return

        with st.spinner("Running your EV-2 script..."):
            base = cat.lower().replace(" ", "_").replace("/", "_").replace("–", "-").replace("+", "plus")
            result = synthesize_ev2(ev0_bytes, add_noise=cfg["add_noise"], seed=cfg["seed"], base_name=base)
        if result is None:
            st.error("❌ EV-2 synthesis failed on the generated image.")
            return
        ev0_rgb, ev2_rgb, ev2_png = result
        show_result(ev0_rgb, ev2_rgb, ev2_png, base_name=base)


def upload_tab(cfg):
    st.subheader("📤 Upload your own EV0")
    uploaded_file = st.file_uploader(
        "Choose an EV0 image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an EV0 exposure image to generate its EV-2 counterpart",
    )
    if uploaded_file is None:
        st.info("👆 Please upload an EV0 image to begin.")
        return

    st.success(f"✅ Uploaded: {uploaded_file.name}")
    base_name = Path(uploaded_file.name).stem
    if st.button("🔄 Generate EV-2 Image", type="primary", use_container_width=True):
        with st.spinner("Running your EV-2 script..."):
            result = synthesize_ev2(
                uploaded_file.getvalue(), add_noise=cfg["add_noise"], seed=cfg["seed"], base_name=base_name
            )
        if result is None:
            st.error("❌ Failed to process the image. Please try a different image.")
            return
        ev0_rgb, ev2_rgb, ev2_png = result
        show_result(ev0_rgb, ev2_rgb, ev2_png, base_name=base_name)


def show_studio_page():
    st.title("🎛️ Generator")
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("---")

    cfg = sidebar_controls()

    tab_gen, tab_upload = st.tabs(["🤖 Generate with ChatGPT", "📤 Upload EV0"])
    with tab_gen:
        generate_tab(cfg)
    with tab_upload:
        upload_tab(cfg)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if st.session_state.page == "studio":
        show_studio_page()
    else:
        show_home_page()


if __name__ == "__main__":
    main()
