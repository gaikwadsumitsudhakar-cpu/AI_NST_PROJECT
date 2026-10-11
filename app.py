
import io
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StyleForge AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# PREMIUM CLASSIC THEME
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #17243A;
    --teal: #397C78;
    --teal-dark: #28625F;
    --ivory: #F5F3EE;
    --muted: #667085;
    --border: #E3E5E8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F5F3EE;
    color: #17243A;
}

header[data-testid="stHeader"] {
    background: rgba(245,243,238,0.9);
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hide default menu and footer */
#MainMenu, footer {
    visibility: hidden;
}

/* Navbar / Brand */
.brand {
    font-size: 1.4rem;
    font-weight: 800;
    color: #17243A;
    letter-spacing: -1px;
}

.brand span {
    color: #397C78;
}

.brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: #E5F0ED;
    margin-right: 10px;
}

/* Hero */
.hero {
    padding: 48px 0 32px;
}

.eyebrow {
    color: #397C78;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    color: #17243A;
    letter-spacing: -2.5px;
    line-height: 1.13;
    margin: 15px 0;
}

.hero-title span {
    color: #397C78;
}

.hero-description {
    color: #667085;
    font-size: 1.05rem;
    max-width: 680px;
    line-height: 1.8;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E3E5E8;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 28px rgba(23,36,58,.045);
}

.section-title {
    color: #17243A;
    font-size: 1.08rem;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #667085;
    font-size: .87rem;
}

/* Uploaders */
div[data-testid="stFileUploader"] {
    background: #F8FAF9;
    border: 1px dashed #B9CCC8;
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

div[data-testid="stFileUploader"] button {
    background: #EAF3F0;
    color: #28625F;
    border: 1px solid #D2E5DF;
    border-radius: 8px;
    font-weight: 650;
}

/* Image previews */
div[data-testid="stImage"] img {
    border-radius: 12px;
}

/* Slider */
div[data-testid="stSlider"] label {
    color: #17243A;
    font-weight: 700;
}

div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #397C78;
    border-color: #397C78;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #397C78;
    color: #FFFFFF;
    border: 1px solid #397C78;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #28625F;
    color: #FFFFFF;
    border-color: #28625F;
    transform: translateY(-1px);
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Result */
.result-heading {
    color: #17243A;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.5px;
}

.result-caption {
    color: #667085;
    font-size: .9rem;
}

/* Footer */
.custom-footer {
    border-top: 1px solid #E0E3E5;
    margin-top: 60px;
    padding-top: 24px;
    color: #667085;
    font-size: .85rem;
}

.custom-footer strong {
    color: #17243A;
}

/* Mobile */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding-top: 30px;
    }

    .hero-title {
        letter-spacing: -1.4px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():

    encoder = VGGEncoder(
        "vgg_normalised.pth"
    ).to(device)

    decoder = Decoder().to(device)

    decoder.load_state_dict(
        torch.load(
            "experiment/final_exp/decoder_final.pth",
            map_location=device
        )
    )

    encoder.eval()
    decoder.eval()

    return encoder, decoder


try:
    encoder, decoder = load_models()

except Exception as e:
    st.error("Failed to load the NST model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


# --------------------------------------------------
# STYLE TRANSFER
# --------------------------------------------------

def style_transfer(content_image, style_image, alpha):

    content_tensor = transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = transform(
        style_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        stylized_image = decoder(
            stylized_features
        )

    return stylized_image


# --------------------------------------------------
# TENSOR TO IMAGE
# --------------------------------------------------

def tensor_to_image(tensor):

    image = tensor.detach().cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    return transforms.ToPILImage()(image)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="brand">
    <span class="brand-icon">🎨</span>
    StyleForge <span>AI</span>
</div>

<div class="hero">
    <div class="eyebrow">AI-Powered Creativity</div>
    <div class="hero-title">
        Transform Images.<br>
        <span>Create Art.</span>
    </div>
    <div class="hero-description">
        Turn ordinary photographs into extraordinary artwork
        using Neural Style Transfer powered by deep learning
        and Adaptive Instance Normalization.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">📷 Content Image</div>'
            '<div class="section-subtitle">Your original photograph</div>',
            unsafe_allow_html=True
        )

        content_file = st.file_uploader(
            "Choose content image",
            type=["jpg", "jpeg", "png"],
            key="content",
            help="Upload the photograph you want to transform."
        )

        content_image = None

        if content_file is not None:
            content_image = Image.open(
                content_file
            ).convert("RGB")

            st.image(
                content_image,
                caption="Content Image",
                use_container_width=True
            )
        else:
            st.markdown(
                '<div class="section-subtitle">'
                'Upload a JPG or PNG image to get started.'
                '</div>',
                unsafe_allow_html=True
            )


with col2:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">🖌️ Style Reference</div>'
            '<div class="section-subtitle">Choose your artistic inspiration</div>',
            unsafe_allow_html=True
        )

        style_file = st.file_uploader(
            "Choose style image",
            type=["jpg", "jpeg", "png"],
            key="style",
            help="Upload a painting, sketch, or other artwork."
        )

        style_image = None

        if style_file is not None:
            style_image = Image.open(
                style_file
            ).convert("RGB")

            st.image(
                style_image,
                caption="Style Reference",
                use_container_width=True
            )
        else:
            st.markdown(
                '<div class="section-subtitle">'
                'Choose an artwork to define the output style.'
                '</div>',
                unsafe_allow_html=True
            )


# --------------------------------------------------
# STYLE STRENGTH
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):

    st.markdown(
        '<div class="section-title">⚙️ Style Strength</div>'
        '<div class="section-subtitle">'
        'Control the influence of the artistic reference on your image.'
        '</div>',
        unsafe_allow_html=True
    )

    alpha = st.slider(
        "Style intensity",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        format="%.2f"
    )

    left, right = st.columns(2)

    with left:
        st.caption("0.0 — Preserve original content")

    with right:
        st.caption("1.0 — Maximum style influence")


# --------------------------------------------------
# GENERATE ARTWORK
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "✨  Generate Artwork",
    use_container_width=True
):

    if content_image is None:
        st.warning("Please upload a content image first.")

    elif style_image is None:
        st.warning("Please upload a style image first.")

    else:
        try:
            with st.spinner(
                "Creating your artwork... Please wait."
            ):

                output = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                output_image = tensor_to_image(output)

            st.success("Your artwork is ready!")

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):

                st.markdown(
                    '<div class="result-heading">'
                    'Your Generated Artwork'
                    '</div>'
                    '<div class="result-caption">'
                    'AI-powered neural style transfer result'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    output_image,
                    use_container_width=True
                )

                buffer = io.BytesIO()

                output_image.save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download Artwork",
                    data=buffer.getvalue(),
                    file_name="styleforge_artwork.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as e:
            st.error("Style transfer failed.")
            st.exception(e)


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    '<div class="eyebrow">Simple workflow</div>'
    '<h2 style="color:#17243A;font-weight:800;">'
    'How It Works</h2>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3, gap="medium")

with step1:
    with st.container(border=True):
        st.markdown("### 01 · Upload")
        st.write(
            "Select your content photograph and style reference."
        )

with step2:
    with st.container(border=True):
        st.markdown("### 02 · Customize")
        st.write(
            "Adjust style strength to control the visual influence."
        )

with step3:
    with st.container(border=True):
        st.markdown("### 03 · Create")
        st.write(
            "Generate your artwork and download the final image."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="custom-footer">
    <strong>🎨 StyleForge AI</strong>
    <br>
    Neural Style Transfer · Python · PyTorch · Streamlit
    <br><br>
    Built to explore the intersection of artificial intelligence and art.
</div>
""", unsafe_allow_html=True)
