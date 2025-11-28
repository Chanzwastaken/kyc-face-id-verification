"""
TrustID | AI Identity Verification
=====================================
Production-grade KYC identity verification powered by Siamese Networks.
Built with Streamlit and PyTorch for portfolio demonstration.

Author: Senior Full-Stack ML Engineer
Tech Stack: Streamlit, PyTorch, facenet-pytorch
"""

import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import io


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for professional fintech UI."""
    st.markdown("""
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            color: white !important;
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.95;
            color: white !important;
        }
        
        /* Verify button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Success box */
        .success-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        }
        
        /* Error box */
        .error-box {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
        }
        
        /* Image containers */
        .image-container {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
            background: #f9f9f9;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_models() -> Tuple[MTCNN, InceptionResnetV1]:
    """
    Load and cache MTCNN and InceptionResnetV1 models.
    
    Uses @st.cache_resource to ensure models are loaded only once
    and reused across all sessions and interactions.
    
    Returns:
        Tuple[MTCNN, InceptionResnetV1]: Face detection and recognition models
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MTCNN for face detection
        # keep_all=True allows detection of multiple faces
        # We'll select the one with highest probability
        st.info("📥 Loading MTCNN face detection model...")
        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False  # We'll handle post-processing manually
        )
        
        # InceptionResnetV1 for face recognition (Siamese Network)
        # pretrained='vggface2' loads weights trained on VGGFace2 dataset
        st.info("📥 Downloading InceptionResnetV1 pre-trained weights (first run only, ~100MB)...")
        st.info("⏳ This may take 1-2 minutes depending on your internet connection...")
        
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # .eval() is CRITICAL:
        # - Disables dropout layers (prevents random neuron dropping)
        # - Sets batch normalization to use running statistics
        # - Ensures deterministic, consistent inference results
        
        if device == 'cuda':
            resnet = resnet.cuda()
        
        st.success("✅ Models loaded successfully!")
        return mtcnn, resnet
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a network error
        if "getaddrinfo failed" in error_msg or "URLError" in error_msg or "Connection" in error_msg:
            st.error("""
            ❌ **Network Error: Unable to Download Model Weights**
            
            The application needs to download pre-trained model weights (~100MB) on first run,
            but encountered a network connectivity issue.
            """)
            
            st.warning("""
            **Troubleshooting Steps:**
            
            1. **Check Internet Connection**
               - Ensure you have an active internet connection
               - Try opening a website in your browser to verify connectivity
            
            2. **Check Firewall/Proxy Settings**
               - Your firewall or corporate proxy may be blocking the download
               - Try temporarily disabling firewall or configuring proxy settings
            
            3. **Manual Download (Advanced)**
               - Download the weights manually from:
                 https://github.com/timesler/facenet-pytorch/releases
               - Place them in: `~/.cache/torch/checkpoints/`
            
            4. **Retry**
               - Once your connection is stable, refresh the page
               - The models will be cached after successful download
            """)
            
            with st.expander("🔧 Technical Error Details"):
                st.code(error_msg)
                st.exception(e)
            
            st.stop()
        else:
            # Other errors
            st.error(f"""
            ❌ **Error Loading Models**
            
            An unexpected error occurred while loading the AI models:
            
            {error_msg}
            """)
            
            with st.expander("🔧 Full Error Details"):
                st.exception(e)
            
            st.stop()


# ============================================================================
# FACE DETECTION & EMBEDDING EXTRACTION
# ============================================================================

def detect_and_crop_face(image: Image.Image, mtcnn: MTCNN) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
    """
    Detect face in image and return both the cropped tensor and PIL image.
    
    Handles multiple faces by selecting the one with highest detection probability.
    
    Args:
        image: PIL Image object
        mtcnn: MTCNN face detection model
        
    Returns:
        Tuple of (face_tensor, cropped_face_pil) or (None, None) if no face detected
    """
    try:
        # Convert to RGB
        image_rgb = image.convert('RGB')
        
        # Detect faces and get bounding boxes with probabilities
        boxes, probs = mtcnn.detect(image_rgb)
        
        # No face detected
        if boxes is None or len(boxes) == 0:
            return None, None
        
        # Multiple faces detected - select the one with highest probability
        if len(boxes) > 1:
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
        else:
            box = boxes[0]
        
        # Crop face from original image for display
        x1, y1, x2, y2 = [int(b) for b in box]
        cropped_face_pil = image_rgb.crop((x1, y1, x2, y2))
        
        # Get aligned face tensor for embedding extraction
        # This time use keep_all=False to get single best face
        mtcnn_single = MTCNN(
            keep_all=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        face_tensor = mtcnn_single(image_rgb)
        
        return face_tensor, cropped_face_pil
        
    except Exception as e:
        st.error(f"Error during face detection: {str(e)}")
        return None, None


def get_embedding(face_tensor: torch.Tensor, resnet: InceptionResnetV1) -> torch.Tensor:
    """
    Extract 512-dimensional face embedding using InceptionResnetV1.
    
    Args:
        face_tensor: Preprocessed face tensor from MTCNN
        resnet: InceptionResnetV1 model
        
    Returns:
        512-dimensional embedding tensor
    """
    # Add batch dimension if needed
    if face_tensor.dim() == 3:
        face_tensor = face_tensor.unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        face_tensor = face_tensor.cuda()
    
    # Extract embedding (no gradient computation for inference)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    
    return embedding


def calculate_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Calculate Euclidean (L2) distance between two embeddings.
    
    This is the core of the Siamese Network verification:
    - Lower distance = faces are similar (same person)
    - Higher distance = faces are different (different people)
    
    Args:
        emb1: First face embedding
        emb2: Second face embedding
        
    Returns:
        Euclidean distance as float
    """
    distance = (emb1 - emb2).norm().item()
    return distance


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="TrustID | AI Identity Verification",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
        <div class="main-header">
            <h1>🔐 TrustID | AI Identity Verification</h1>
            <p>Bank-grade KYC powered by Siamese Networks</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - CONTROL PANEL
    # ========================================================================
    
    st.sidebar.title("⚙️ Control Panel")
    
    st.sidebar.markdown("### Verification Threshold")
    st.sidebar.markdown("""
    Adjust the similarity threshold for identity matching:
    - **Lower (0.4-0.5)**: Stricter verification
    - **Medium (0.6)**: Balanced (recommended)
    - **Higher (0.7-0.9)**: Lenient verification
    """)
    
    threshold = st.sidebar.slider(
        "Threshold Value",
        min_value=0.4,
        max_value=0.9,
        value=0.60,
        step=0.05,
        help="Distance threshold for face matching"
    )
    
    # Visual threshold indicator
    if threshold < 0.55:
        threshold_status = "🔴 Strict"
    elif threshold < 0.70:
        threshold_status = "🟡 Balanced"
    else:
        threshold_status = "🟢 Lenient"
    
    st.sidebar.info(f"**Current Setting:** {threshold_status}\n\n**Value:** {threshold:.2f}")
    
    st.sidebar.divider()
    
    # Technical Details Expander
    with st.sidebar.expander("📚 Technical Details"):
        st.markdown("""
        ### Siamese Network Architecture
        
        This system uses a **Siamese Neural Network** approach:
        
        1. **Face Detection**: MTCNN (Multi-task CNN)
           - Detects faces in images
           - Handles multiple faces automatically
           - Selects highest probability detection
        
        2. **Feature Extraction**: InceptionResnetV1
           - Pre-trained on VGGFace2 dataset
           - 3.3M images, 9,131 identities
           - Outputs 512-dimensional embeddings
        
        3. **Similarity Metric**: Euclidean Distance
           - Measures L2 distance between embeddings
           - Lower distance = more similar faces
           - Typical ranges:
             - Same person: 0.3 - 0.7
             - Different people: 0.8 - 1.5+
        
        ### Why "Siamese"?
        
        The network processes both images through the **same** 
        InceptionResnetV1 model (like identical twins), then 
        compares the resulting embeddings. This ensures 
        consistent feature extraction for fair comparison.
        """)
    
    # Model info
    with st.sidebar.expander("🤖 Model Information"):
        st.markdown("""
        **Models Used:**
        - MTCNN (Face Detection)
        - InceptionResnetV1 (Face Recognition)
        
        **Training Dataset:**
        - VGGFace2 (3.3M images)
        
        **Device:**
        """)
        device = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
        st.markdown(f"- {device}")
    
    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    
    with st.spinner("🔄 Loading AI models..."):
        mtcnn, resnet = load_models()
    
    # ========================================================================
    # MAIN AREA - IMAGE UPLOAD
    # ========================================================================
    
    st.markdown("### 📸 Upload Identity Documents")
    st.markdown("Upload both an ID document and a selfie to verify identity.")
    
    col1, col2 = st.columns(2, gap="large")
    
    # Column 1: ID Document
    with col1:
        st.markdown("#### 🆔 ID Document")
        id_file = st.file_uploader(
            "Upload ID Photo",
            type=["jpg", "jpeg", "png"],
            key="id_upload",
            help="Upload a clear photo of the ID document"
        )
        
        if id_file:
            try:
                id_image = Image.open(id_file)
                st.markdown("**Original Image:**")
                st.image(id_image, use_container_width=True)
                
                # Store in session state for verification
                st.session_state['id_image'] = id_image
                
            except Exception as e:
                st.error(f"⚠️ Error loading ID image: {str(e)}\n\nPlease upload a valid image file.")
    
    # Column 2: Selfie
    with col2:
        st.markdown("#### 🤳 Live Selfie")
        selfie_file = st.file_uploader(
            "Upload Selfie Photo",
            type=["jpg", "jpeg", "png"],
            key="selfie_upload",
            help="Upload a clear selfie photo"
        )
        
        if selfie_file:
            try:
                selfie_image = Image.open(selfie_file)
                st.markdown("**Original Image:**")
                st.image(selfie_image, use_container_width=True)
                
                # Store in session state for verification
                st.session_state['selfie_image'] = selfie_image
                
            except Exception as e:
                st.error(f"⚠️ Error loading selfie image: {str(e)}\n\nPlease upload a valid image file.")
    
    st.divider()
    
    # ========================================================================
    # VERIFICATION BUTTON
    # ========================================================================
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        verify_button = st.button(
            "🔍 Verify Identity",
            type="primary",
            use_container_width=True
        )
    
    # ========================================================================
    # VERIFICATION LOGIC
    # ========================================================================
    
    if verify_button:
        # Validate uploads
        if 'id_image' not in st.session_state or 'selfie_image' not in st.session_state:
            st.error("⚠️ Please upload both ID document and selfie images before verification!")
            st.stop()
        
        id_image = st.session_state['id_image']
        selfie_image = st.session_state['selfie_image']
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Detect face in ID
            status_text.text("🔍 Detecting face in ID document...")
            progress_bar.progress(20)
            
            id_face_tensor, id_face_cropped = detect_and_crop_face(id_image, mtcnn)
            
            if id_face_tensor is None:
                st.markdown("""
                    <div class="error-box">
                        ⚠️ No face detected in ID document!
                    </div>
                """, unsafe_allow_html=True)
                st.warning("Please upload a clearer photo with a visible face.")
                st.stop()
            
            # Step 2: Detect face in Selfie
            status_text.text("🔍 Detecting face in selfie...")
            progress_bar.progress(40)
            
            selfie_face_tensor, selfie_face_cropped = detect_and_crop_face(selfie_image, mtcnn)
            
            if selfie_face_tensor is None:
                st.markdown("""
                    <div class="error-box">
                        ⚠️ No face detected in selfie!
                    </div>
                """, unsafe_allow_html=True)
                st.warning("Please upload a clearer photo with a visible face.")
                st.stop()
            
            # Display cropped faces
            st.markdown("### 🎯 AI-Detected Faces")
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("**ID Document Face:**")
                st.image(id_face_cropped, use_container_width=True)
            
            with col2:
                st.markdown("**Selfie Face:**")
                st.image(selfie_face_cropped, use_container_width=True)
            
            st.divider()
            
            # Step 3: Extract embeddings
            status_text.text("🧠 Extracting face embeddings...")
            progress_bar.progress(60)
            
            id_embedding = get_embedding(id_face_tensor, resnet)
            selfie_embedding = get_embedding(selfie_face_tensor, resnet)
            
            # Step 4: Calculate similarity
            status_text.text("📊 Calculating similarity score...")
            progress_bar.progress(80)
            
            distance = calculate_distance(id_embedding, selfie_embedding)
            
            # Step 5: Verification decision
            status_text.text("✅ Verification complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # ================================================================
            # RESULTS DISPLAY
            # ================================================================
            
            st.markdown("### 📋 Verification Results")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Similarity Score",
                    f"{distance:.4f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Threshold",
                    f"{threshold:.2f}",
                    delta=None
                )
            
            with col3:
                match_status = "MATCH ✅" if distance < threshold else "NO MATCH ❌"
                st.metric(
                    "Result",
                    match_status,
                    delta=None
                )
            
            st.divider()
            
            # Success or Failure Display
            if distance < threshold:
                # SUCCESS STATE
                st.markdown("""
                    <div class="success-box">
                        ✅ IDENTITY CONFIRMED
                    </div>
                """, unsafe_allow_html=True)
                
                st.balloons()  # Celebration effect!
                
                st.success(f"""
                **Verification Successful!**
                
                The faces match with high confidence. The person in the selfie 
                is verified to be the same as the person in the ID document.
                
                - **Distance Score:** {distance:.4f}
                - **Threshold:** {threshold:.2f}
                - **Result:** Distance < Threshold ✓
                - **Confidence:** {((threshold - distance) / threshold * 100):.1f}%
                """)
                
            else:
                # FAILURE STATE
                st.markdown("""
                    <div class="error-box">
                        ❌ IDENTITY MISMATCH
                    </div>
                """, unsafe_allow_html=True)
                
                st.error(f"""
                **Verification Failed!**
                
                The faces do not match. The person in the selfie appears to be 
                different from the person in the ID document.
                
                - **Distance Score:** {distance:.4f}
                - **Threshold:** {threshold:.2f}
                - **Result:** Distance ≥ Threshold ✗
                
                **Recommendation:** Please retry with clearer photos or adjust 
                the threshold if you believe this is an error.
                """)
            
            # Debug information
            with st.expander("🔧 Technical Debug Information"):
                st.json({
                    "id_face_tensor_shape": str(id_face_tensor.shape),
                    "selfie_face_tensor_shape": str(selfie_face_tensor.shape),
                    "id_embedding_shape": str(id_embedding.shape),
                    "selfie_embedding_shape": str(selfie_embedding.shape),
                    "euclidean_distance": float(distance),
                    "threshold": float(threshold),
                    "verification_passed": bool(distance < threshold),
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                })
        
        except Exception as e:
            st.error(f"❌ **Error during verification:**\n\n{str(e)}")
            with st.expander("📋 Full Error Details"):
                st.exception(e)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
