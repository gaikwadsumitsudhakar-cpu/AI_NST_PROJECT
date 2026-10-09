import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Neural Style Transfer",
    page_icon="🎨",
    layout="wide"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load Model
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


# --------------------------------------------------
# Load Models
# --------------------------------------------------

try:
    encoder, decoder = load_models()

except Exception as e:

    st.error("Failed to load the NST model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


# --------------------------------------------------
# Style Transfer
# --------------------------------------------------

def style_transfer(
    content_image,
    style_image,
    alpha
):

    content_tensor = transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = transform(
        style_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        # Encoder
        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        # AdaIN
        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        # Alpha blending
        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        # Decoder
        stylized_image = decoder(
            stylized_features
        )

    return stylized_image


# --------------------------------------------------
# Convert Tensor -> PIL
# --------------------------------------------------

def tensor_to_image(tensor):

    image = tensor.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    return image


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎨 AI Neural Style Transfer")

st.write(
    "Transform your content image using the artistic style "
    "of another image using AdaIN."
)


st.divider()


# --------------------------------------------------
# Upload Images
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("📷 Content Image")

    content_file = st.file_uploader(
        "Upload Content Image",
        type=["jpg", "jpeg", "png"],
        key="content"
    )


with col2:

    st.subheader("🎨 Style Image")

    style_file = st.file_uploader(
        "Upload Style Image",
        type=["jpg", "jpeg", "png"],
        key="style"
    )


# --------------------------------------------------
# Display Uploaded Images
# --------------------------------------------------

content_image = None
style_image = None


if content_file:

    content_image = Image.open(
        content_file
    ).convert("RGB")

    with col1:

        st.image(
            content_image,
            caption="Content Image",
            use_container_width=True
        )


if style_file:

    style_image = Image.open(
        style_file
    ).convert("RGB")

    with col2:

        st.image(
            style_image,
            caption="Style Image",
            use_container_width=True
        )


# --------------------------------------------------
# Alpha
# --------------------------------------------------

st.divider()

st.subheader("⚙️ Style Strength")

alpha = st.slider(
    "Alpha",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.caption(
    "0 = original content image | "
    "1 = maximum style transfer"
)


# --------------------------------------------------
# Generate Button
# --------------------------------------------------

if st.button(
    "✨ Generate Stylized Image",
    use_container_width=True
):

    if content_image is None:

        st.warning(
            "Please upload a content image."
        )

    elif style_image is None:

        st.warning(
            "Please upload a style image."
        )

    else:

        with st.spinner(
            "Generating stylized image..."
        ):

            try:

                output = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                output_image = tensor_to_image(
                    output
                )

                st.success(
                    "Style transfer completed!"
                )

                st.divider()

                st.subheader(
                    "🖼️ Generated Image"
                )

                st.image(
                    output_image,
                    caption="AI Generated Stylized Image",
                    use_container_width=True
                )

                # Download button
                import io

                buffer = io.BytesIO()

                output_image.save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download Result",
                    data=buffer.getvalue(),
                    file_name="stylized_image.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "Style transfer failed."
                )

                st.exception(e)