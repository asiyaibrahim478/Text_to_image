import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🎨",
    layout="centered"
)

# Title and instructions
st.title("🎨 Minimal Text-to-Image Generator")
st.markdown("""
Enter a description below and click **Generate** to create an image using AI.

⚠️ **Warning:** This app uses an uncensored model and may generate inappropriate content. 
Generation takes 2-5 minutes on CPU. Please be patient.
""")

# Load model with caching
@st.cache_resource
def load_model():
    """Load the Stable Diffusion model once and cache it."""
    try:
        with st.spinner("Loading AI model (this may take a few minutes on first run)..."):
            # Load model for CPU with float32
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Set to CPU device
            pipe = pipe.to("cpu")
            
            # Disable progress bar for cleaner output
            pipe.set_progress_bar_config(disable=True)
            
            st.success("Model loaded successfully!")
            return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
        return None

# Load the model
pipeline = load_model()

# User input
prompt = st.text_input(
    "Enter your image description:",
    max_chars=200,
    placeholder="A red apple on a wooden table, photorealistic"
)

# Generate button
if st.button("Generate Image", type="primary"):
    if not prompt or prompt.strip() == "":
        st.warning("Please enter a description before generating.")
    else:
        try:
            with st.spinner(f"Generating image... This will take 2-5 minutes on CPU. Please wait."):
                # Generate image
                with torch.no_grad():
                    result = pipeline(
                        prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5
                    )
                
                # Get the generated image
                image = result.images[0]
                
                # Display the image
                st.success("Image generated successfully!")
                st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                
                # Optional: provide download button
                st.markdown("---")
                # Convert PIL image to bytes for download
                import io
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            st.info("Please try again with a different prompt or refresh the page.")

# Footer
st.markdown("---")
st.markdown("""
<small>Powered by Stable Diffusion v1.4 from Hugging Face. 
Running on CPU (no GPU required).</small>
""", unsafe_allow_html=True)
