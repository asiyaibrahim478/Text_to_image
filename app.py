import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import warnings
import random

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .title-container {
        text-align: center;
        padding: 40px 0;
        color: #c6ff00;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .title-main {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.5rem;
        margin-top: 10px;
    }
    .input-container {
        max-width: 900px;
        margin: 0 auto 40px auto;
    }
    .stTextInput input {
        font-size: 1.1rem;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton button {
        font-size: 1.2rem;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: bold;
        margin: 0 10px;
    }
    .gallery-section {
        background: rgba(0,0,0,0.3);
        padding: 30px;
        border-radius: 15px;
        margin-top: 40px;
    }
    .gallery-title {
        color: white;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 30px;
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sample prompts for "Surprise Me"
SAMPLE_PROMPTS = [
    "A panda bear wearing a chef hat baking a cake in a sunny kitchen, digital art",
    "Portrait of a beautiful young woman with elegant makeup and flowing hair",
    "Renaissance painting of an elephant in a tuxedo",
    "A golden retriever puppy playing in a sunny garden with flowers",
    "A steampunk robot reading a book in an ancient library",
    "A lighthouse on a rocky cliff during a beautiful sunset",
    "A cozy cottage in the snowy woods with warm lights in the windows",
    "An astronaut floating in space with Earth in the background, photorealistic",
    "A mystical dragon flying over a medieval castle at night",
    "A field of purple lavender under a blue sky with white clouds"
]

# Load model with caching
@st.cache_resource
def load_model():
    """Load the Stable Diffusion model once and cache it."""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
        return None

# Title Section
st.markdown("""
<div class="title-container">
    <h1 class="title-main">Text to Image with<br>AI Image Generator</h1>
</div>
""", unsafe_allow_html=True)

# Input Section
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    prompt = st.text_input(
        "",
        placeholder="Describe what you want to see, Confused what to type? select Surprise Me",
        max_chars=200,
        label_visibility="collapsed"
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        generate_btn = st.button("🎨 Generate", type="primary", use_container_width=True)
    with btn_col2:
        surprise_btn = st.button("✨ Surprise Me", use_container_width=True)

# Handle Surprise Me button
if surprise_btn:
    prompt = random.choice(SAMPLE_PROMPTS)
    st.rerun()

# Load model
with st.spinner("Loading AI model... Please wait..."):
    pipeline = load_model()

# Generate image
if generate_btn and prompt and prompt.strip():
    with st.spinner("🎨 Generating your image... This may take 2-5 minutes on CPU. Please be patient..."):
        try:
            with torch.no_grad():
                result = pipeline(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                )
            
            image = result.images[0]
            
            st.success("✅ Image generated successfully!")
            
            # Display generated image
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                
                # Download button
                import io
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Download Image",
                    data=byte_im,
                    file_name="ai_generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            st.info("Please try again with a different prompt.")

elif generate_btn:
    st.warning("⚠️ Please enter a description before generating.")

# Featured Gallery Section
st.markdown("""
<div class="gallery-section">
    <h2 class="gallery-title">Featured Gallery</h2>
</div>
""", unsafe_allow_html=True)

# Gallery images (placeholder)
gallery_col1, gallery_col2, gallery_col3 = st.columns(3)

gallery_items = [
    {
        "prompt": "Panda bear baking a cake in a sunny kitchen, digital art",
        "caption": "Panda bear baking a cake in a sunny kitchen, digital art"
    },
    {
        "prompt": "Portrait of a beautiful young woman of 18 age",
        "caption": "Portrait of a beautiful young woman of 18 age"
    },
    {
        "prompt": "Renaissance painting of an elephant in a tuxedo",
        "caption": "Renaissance painting of an elephant in a tuxedo"
    }
]

with gallery_col1:
    st.info("💡 Click 'Surprise Me' to generate random creative images!")
    st.markdown(f"**Example:** {gallery_items[0]['caption']}")

with gallery_col2:
    st.info("🎨 Use detailed descriptions for better results!")
    st.markdown(f"**Example:** {gallery_items[1]['caption']}")

with gallery_col3:
    st.info("⚡ Generation takes 2-5 minutes on CPU")
    st.markdown(f"**Example:** {gallery_items[2]['caption']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <small>Powered by Stable Diffusion v1.4 from Hugging Face | Running on CPU (no GPU required)</small>
</div>
""", unsafe_allow_html=True)
