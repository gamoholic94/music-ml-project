# app.py
# This script serves as the main entry point for our Streamlit web application.
# Users can upload an audio file and later receive a predicted music genre.

# Import Streamlit for building the web application UI
import streamlit as st
import streamlit.components.v1 as components

# For numerical operations
import numpy as np
import pandas as pd

# For audio processing
import librosa

# For loading scaler/model files
import joblib


SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS
N_FFT = 2048
HOP_LENGTH = 512
NUM_MFCC = 13


# Configure page settings
st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* ===== GENERAL STYLING ===== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Ctext y='24' font-size='24'%3E%F0%9F%8E%B5%3C/text%3E%3C/svg%3E") 16 16, auto !important;
    }
    
    /* Main background */
    .stApp,
    .main,
    [data-testid="stAppViewContainer"] {
        background: #f3e8ff;
    }

    .main {
        padding: 20px;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3.2em;
        margin-bottom: 0.5em;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.4);
        font-weight: 800;
        letter-spacing: 1px;
        animation: slideDown 0.6s ease-out;
    }
    
    h2 {
        color: #667eea;
        margin-top: 2em;
        margin-bottom: 1em;
        font-size: 2em;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        border-bottom: 3px solid #764ba2;
        padding-bottom: 0.5em;
        transition: all 0.3s ease;
    }
    
    h2:hover {
        color: #764ba2;
        transform: translateX(5px);
    }
    
    h3 {
        color: #333333;
        font-size: 1.5em;
        margin: 1em 0 0.5em 0;
        font-weight: 700;
    }
    
    p {
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* ===== CONTAINERS & CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.4);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid transparent;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== INPUT FIELDS ===== */
    input, textarea, select {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border: 2px solid #764ba2 !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
        outline: none !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 30px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(245, 87, 108, 0.05) 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border: 3px dashed #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== MESSAGES & ALERTS ===== */
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 5px solid #2ecc71;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left: 5px solid #c92a2a;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #3498db;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== SIDEBAR ===== */
    .stSidebar {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stSidebar h3 {
        color: #667eea;
        margin-bottom: 1em;
        border-bottom: 2px solid #764ba2;
        padding-bottom: 0.5em;
    }

    .about-section,
    .about-section h3,
    .about-section p {
        color: white;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2em 0;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ===== LISTS ===== */
    ul {
        margin-left: 1.5em;
    }
    
    li {
        margin: 0.5em 0;
        transition: all 0.2s ease;
    }
    
    li:hover {
        color: #667eea;
        transform: translateX(5px);
    }

    .how-it-works li:hover {
        color: white;
        font-size: 1.05em;
        transform: translateX(5px);
    }
    
    /* ===== TEXT STYLING ===== */
    .header-text {
        color: #ffffff;
        font-size: 1.3em;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    strong {
        color: #764ba2;
        font-weight: 700;
    }
    
    em {
        color: #667eea;
        font-style: italic;
    }
    
    /* ===== LINKS ===== */
    a {
        color: #667eea;
        text-decoration: none;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
    }
    
    a:hover {
        color: #764ba2;
        border-bottom: 2px solid #764ba2;
    }
    
    /* ===== CUSTOM CONTAINERS ===== */
    .card-container {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border: 2px solid #667eea;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.2em;
        }
        
        h2 {
            font-size: 1.5em;
        }
        
        .main {
            padding: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

components.html("""
    <script>
        const rootDocument = window.parent.document;
        let lastTrailTime = 0;

        if (!rootDocument.getElementById("music-trail-style")) {
            const style = rootDocument.createElement("style");
            style.id = "music-trail-style";
            style.textContent = `
                .music-trail-note {
                    position: fixed;
                    z-index: 999999;
                    pointer-events: none;
                    color: #8b5cf6;
                    font-size: 22px;
                    line-height: 1;
                    text-shadow: 0 2px 8px rgba(139, 92, 246, 0.25);
                    animation: musicTrailFade 0.8s ease-out forwards;
                }

                @keyframes musicTrailFade {
                    from {
                        opacity: 0.85;
                        transform: translate(0, 0) scale(1);
                    }
                    to {
                        opacity: 0;
                        transform: translate(10px, -24px) scale(0.45);
                    }
                }
            `;
            rootDocument.head.appendChild(style);
        }

        rootDocument.addEventListener("mousemove", (event) => {
            const now = Date.now();
            if (now - lastTrailTime < 45) {
                return;
            }
            lastTrailTime = now;

            const note = rootDocument.createElement("span");
            note.className = "music-trail-note";
            note.textContent = "♪";
            note.style.left = `${event.clientX + 8}px`;
            note.style.top = `${event.clientY + 8}px`;

            rootDocument.body.appendChild(note);
            window.setTimeout(() => note.remove(), 800);
        });
    </script>
""", height=1, width=1)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """
    Loads the pre-trained Keras CNN model, StandardScaler, and genre labels.
    Cached so the model and scaler are loaded only once per app session.
    """
    try:
        import tensorflow as tf

        class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
            def __init__(self, *args, **kwargs):
                kwargs.pop("renorm", None)
                kwargs.pop("renorm_clipping", None)
                kwargs.pop("renorm_momentum", None)
                super().__init__(*args, **kwargs)

        class CompatibleDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        model = tf.keras.models.load_model(
            "music_genre_cnn.h5",
            compile=False,
            custom_objects={
                "BatchNormalization": CompatibleBatchNormalization,
                "Dense": CompatibleDense,
            },
        )
        scaler = joblib.load("scaler.joblib")

        genre_mapping = {
            0: "blues",
            1: "classical",
            2: "country",
            3: "disco",
            4: "hiphop",
            5: "jazz",
            6: "metal",
            7: "pop",
            8: "reggae",
            9: "rock",
        }

        return model, scaler, genre_mapping
    except FileNotFoundError as e:
        st.error(
            f"Error loading model or scaler file: {e}. "
            "Please ensure the files are in the correct directory."
        )
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/scaler: {e}")
        st.stop()


def extract_feature_vector(segment, sr):
    """
    Extracts one 28-value feature vector from an audio segment.
    """
    mfccs = np.mean(
        librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ),
        axis=1,
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=segment,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ),
        axis=1,
    )

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=segment,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )

    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=segment,
            hop_length=HOP_LENGTH,
        )
    )

    return np.hstack((
        mfccs,
        chroma,
        spectral_centroid,
        spectral_rolloff,
        zcr,
    ))


def extract_features(audio_file):
    """
    Extracts segment-level feature vectors from an uploaded audio file.
    """
    try:
        audio_file.seek(0)
        signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=TRACK_DURATION_SECONDS)
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

        if len(signal) < samples_per_segment:
            st.error("Audio file is too short. Please upload at least 3 seconds of audio.")
            return None

        feature_vectors = []
        available_segments = min(NUM_SEGMENTS, len(signal) // samples_per_segment)

        for segment_index in range(available_segments):
            start_sample = segment_index * samples_per_segment
            end_sample = start_sample + samples_per_segment
            segment = signal[start_sample:end_sample]
            feature_vectors.append(extract_feature_vector(segment, sr))

        return np.array(feature_vectors)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None


def predict_genre(uploaded_file, model, scaler, genre_mapping):
    """
    Scales extracted features, reshapes them for the CNN, and returns predictions.
    """
    features = extract_features(uploaded_file)
    if features is None:
        return None, None

    feature_names = getattr(scaler, "feature_names_in_", [str(i) for i in range(features.shape[1])])
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)
    features_cnn = np.expand_dims(features_scaled, axis=-1)
    segment_probabilities = model.predict(features_cnn, verbose=0)
    probabilities = np.mean(segment_probabilities, axis=0)

    predicted_index = int(np.argmax(probabilities))
    predicted_genre = genre_mapping.get(predicted_index, "unknown")

    return predicted_genre, probabilities


def initialize_session_state():
    """
    Ensures Streamlit session-state keys exist before they are read.
    """
    defaults = {
        "model_loaded": False,
        "model": None,
        "scaler": None,
        "genre_mapping": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """
    The primary function that builds and runs our Streamlit application.
    """
    initialize_session_state()

    # Load model, scaler, and genre mapping once at the start using session state
    if not st.session_state.model_loaded:
        with st.spinner("🎵 Loading AI Model... (This happens only once)"):
            try:
                st.session_state.model, st.session_state.scaler, st.session_state.genre_mapping = load_model_and_scaler()
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
    
    model = st.session_state.model
    scaler = st.session_state.scaler
    genre_mapping = st.session_state.genre_mapping
    
    # ---------------------------------------------------
    # Hero Section with Gradient Background
    # ---------------------------------------------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1>🎵 Music Genre Classification</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ffffff; font-size: 1.1em;'>Powered by Advanced AI</p>", unsafe_allow_html=True)

    st.divider()

    # ---------------------------------------------------
    # Info Section with Columns
    # ---------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='how-it-works' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3>🎯 How It Works</h3>
            <ul>
                <li>Upload a .wav audio file</li>
                <li>Our CNN model analyzes the audio</li>
                <li>Get instant genre prediction</li>
                <li>View confidence scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3>📊 Supported Genres</h3>
            <p>Blues • Classical • Country • Disco • Hip-Hop</p>
            <p>Jazz • Metal • Pop • Reggae • Rock</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------------------------------------------------
    # Upload Section
    # ---------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📁 Upload Your Audio File</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Drag and drop your audio file here or click to browse",
            type=['wav','mp3'],
            label_visibility="collapsed"
        )

    # ---------------------------------------------------
    # Handle Uploaded File
    # ---------------------------------------------------
    if uploaded_file is not None:
        st.divider()
        # --- Perform Classification ---
        # This block runs only when the user has provided a file.
        # Provide immediate feedback by displaying an audio player for the uploaded file.
        # This is great for user experience, as they can confirm what they've uploaded.
        st.audio(uploaded_file, format='audio/wav')

        # Call the prediction function
        with st.spinner("🎵 Analyzing audio... This may take a moment..."):
            predicted_genre, probabilities = predict_genre(uploaded_file, model, scaler, genre_mapping)
        
        if predicted_genre is not None and probabilities is not None:
            # Display the predicted genre
            st.success(f"✨ Predicted Genre: **{predicted_genre.upper()}**")
            
            # Display confidence scores for all genres
            st.markdown("<h3>📊 Confidence Scores</h3>", unsafe_allow_html=True)
            
            # Create a dictionary of genres with their probabilities
            genre_scores = {genre: float(probabilities[idx]) for idx, genre in genre_mapping.items()}
            
            # Sort by probability in descending order
            sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Display as a bar chart
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart({genre: score for genre, score in sorted_genres})
            
            with col2:
                st.markdown("**Top Predictions:**")
                for i, (genre, score) in enumerate(sorted_genres[:3], 1):
                    st.markdown(f"{i}. **{genre}**: {score:.2%}")
        else:
            st.error("❌ Could not process the audio file. Please try a different file.")
    
    else:
        # When no file is uploaded
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); 
                    padding: 40px; border-radius: 10px; text-align: center;'>
            <h3>📢 Ready to get started?</h3>
            <p>Upload an audio file to begin classification</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # Sidebar
    # ---------------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div class='about-section' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white;'>
            <h3>ℹ️ About</h3>
            <p>This app uses a Convolutional Neural Network (CNN) trained on music features to classify genres accurately.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("<h3>📝 Tips</h3>", unsafe_allow_html=True)
        st.info("💡 Use high-quality audio files for best results")
        st.warning("⏱️ Longer files may take more time to process")
        st.success("✨ The model supports 10 music genres")


# ---------------------------------------------------
# Run the App
# ---------------------------------------------------
if __name__ == '__main__':
    main()
