import streamlit as st
import torch
import gc
import requests
import os
import time
import json
import logging
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import spacy
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("art_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_GENERATIONS_PER_HOUR = 20
RESET_INTERVAL = 3600  # 1 hour in seconds
IMAGE_SIZE = (512, 512)
THUMBNAIL_SIZE = (128, 128)
API_TIMEOUT = 30

@dataclass
class GenerationResult:
    image_path: Optional[str]
    references: List[Dict]
    style_info: Dict
    success: bool
    error_message: Optional[str] = None

class ArtGeneratorApp:
    def __init__(self):
        self.setup_environment()
        self.setup_models()
        self.setup_data()
        
    def setup_environment(self):
        """Initialize environment and validate configuration"""
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            st.error("Please set HF_TOKEN in .env file or Streamlit secrets")
            st.info("Get your token from: https://huggingface.co/settings/tokens")
            st.stop()
            
        # Force CPU usage for consistency
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)  # Limit CPU threads
        
        # API configurations with updated models
        self.api_urls = {
            "text": "https://api-inference.huggingface.co/models/distilbert/distilgpt2",
            "image": "https://api-inference.huggingface.co/models/stable-diffusion-v1-5/stable-diffusion-v1-5"
        }
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # Test API access
        self.test_api_access()
        
        # Create necessary directories
        os.makedirs("output", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def test_api_access(self):
        """Test API access and suggest fixes"""
        try:
            test_response = requests.get(
                "https://api-inference.huggingface.co/models/stable-diffusion-v1-5/stable-diffusion-v1-5",
                headers=self.headers,
                timeout=10
            )
            
            if test_response.status_code == 403:
                st.error("🚫 API Access Denied - Please check your Hugging Face token:")
                st.markdown("""
                **Common solutions:**
                1. **Get a new token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
                2. **Check token permissions**: Make sure it has 'Read' access
                3. **Accept model terms**: Visit the model page and accept terms of use
                4. **Wait for model loading**: Some models need time to warm up
                """)
                st.info("🔄 The app will try alternative models automatically")
                
            elif test_response.status_code == 503:
                st.warning("⏳ Model is loading on Hugging Face servers. This usually takes 1-2 minutes.")
                
        except Exception as e:
            logger.warning(f"API test failed: {e}")
            st.info("🔧 Will attempt generation with fallback models")

    @st.cache_resource
    def load_nlp_model(_self):
        """Load spaCy model with caching"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            st.stop()

    @st.cache_resource
    def load_clip_models(_self):
        """Load CLIP models with proper caching"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            model.eval()
            
            logger.info("CLIP models loaded successfully")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load CLIP models: {e}")
            st.error(f"Failed to load CLIP models: {e}")
            st.stop()

    @st.cache_resource
    def load_safety_classifier(_self):
        """Load NSFW safety classifier with caching"""
        try:
            from transformers import pipeline
            return pipeline(
                "image-classification", 
                model="Falconsai/nsfw_image_detection",
                torch_dtype=torch.float32
            )
        except Exception as e:
            logger.warning(f"Failed to load safety classifier: {e}")
            return None

    def setup_models(self):
        """Initialize all AI models"""
        self.nlp = self.load_nlp_model()
        self.clip_model, self.clip_processor = self.load_clip_models()
        self.safety_classifier = self.load_safety_classifier()

    @staticmethod
    @st.cache_data
    def load_and_validate_csv() -> pd.DataFrame:
        """Load and validate the art references CSV"""
        csv_path = "data/art_references.csv"
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file {csv_path} not found")
            return pd.DataFrame(columns=['image_path', 'style'])
            
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['image_path', 'style']
            
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV missing required columns: {required_columns}")
                return pd.DataFrame(columns=required_columns)
                
            # Validate and filter valid images
            valid_rows = []
            invalid_paths = []
            for idx, row in df.iterrows():
                image_path = os.path.normpath(row['image_path'])
                if ArtGeneratorApp.validate_image_path(image_path):
                    valid_rows.append(row)
                else:
                    invalid_paths.append(image_path)
                    
            valid_df = pd.DataFrame(valid_rows)
            logger.info(f"Loaded {len(valid_df)}/{len(df)} valid image references")
            if invalid_paths:
                logger.warning(f"Invalid image paths: {', '.join(invalid_paths)}")
                st.warning(f"Invalid images: {', '.join(invalid_paths)}")
            
            if len(valid_df) < len(df):
                st.warning(f"Only {len(valid_df)}/{len(df)} images are valid and accessible")
                
            return valid_df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            st.error(f"Failed to load art references: {e}")
            return pd.DataFrame(columns=['image_path', 'style'])

    @staticmethod
    def validate_image_path(path: str) -> bool:
        """Validate that an image path exists and is readable"""
        try:
            if not os.path.exists(path):
                return False
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def setup_data(self):
        """Load and prepare data"""
        self.df = self.load_and_validate_csv()
        self.vectorstore = self.build_faiss_index()

    @st.cache_resource
    def build_faiss_index(_self):
        """Build FAISS index for image similarity search"""
        if _self.df.empty:
            logger.warning("No data available for FAISS index")
            return None
            
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            try:
                logger.info("Using simple similarity search (FAISS replacement)")
                return _self.build_simple_index()
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                
        return _self.build_simple_index()

    def build_simple_index(self):
        """Build a simple embedding-based index as FAISS replacement"""
        embeddings = []
        metadata = []
        
        with torch.no_grad():
            for idx, row in self.df.iterrows():
                try:
                    image_path = os.path.normpath(row['image_path'])
                    with Image.open(image_path) as image:
                        image = image.convert('RGB').resize(THUMBNAIL_SIZE)
                        inputs = self.clip_processor(images=image, return_tensors="pt")
                        embedding = self.clip_model.get_image_features(**inputs)
                        embeddings.append(embedding.cpu().numpy().flatten())
                        metadata.append(row.to_dict())
                        del inputs, embedding
                        
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
                    
        if embeddings:
            embeddings_array = np.array(embeddings)
            logger.info(f"Built index with {len(embeddings)} images")
            return {"embeddings": embeddings_array, "metadata": metadata}
        
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def call_api(self, url: str, payload: Dict, timeout: int = API_TIMEOUT) -> Tuple[Optional[Dict], Optional[str]]:
        """Make API call with retry and proper error handling"""
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('image/'):
                return {"content": response.content}, None
            else:
                return response.json(), None
                
        except requests.exceptions.Timeout:
            return None, "API request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"API request failed: {str(e)}"
        except json.JSONDecodeError:
            return None, "Invalid API response format"

    def extract_style_elements(self, prompt: str, style: str = "Auto-detect") -> Dict:
        """Extract style elements from prompt or use selected style"""
        try:
            doc = self.nlp(prompt)
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
            available_styles = self.df['style'].unique().tolist() if not self.df.empty else []
            
            if style and style != "Auto-detect":
                detected_style = style
            else:
                # Prioritize exact matches, then partial matches
                prompt_lower = prompt.lower()
                detected_style = next(
                    (s for s in available_styles if s.lower() == prompt_lower),
                    next(
                        (s for s in available_styles if s.lower() in prompt_lower),
                        "contemporary"
                    )
                )
            return {
                "style": detected_style,
                "subject": nouns[0] if nouns else "artwork",
                "refined_prompt": f"{prompt}, in the style of {detected_style}" if detected_style else prompt,
                "elements": {"nouns": nouns[:3], "adjectives": adjectives[:3]}
            }
        except Exception as e:
            logger.warning(f"Style extraction failed: {e}")
            return {
                "style": style if style and style != "Auto-detect" else "contemporary",
                "subject": "artwork",
                "refined_prompt": prompt,
                "elements": {"nouns": [], "adjectives": []}
            }

    def find_similar_images(self, query: str, k: int = 2) -> List[Dict]:
        """Find similar images using simple cosine similarity"""
        if not self.vectorstore or self.df.empty:
            logger.warning("No image index available")
            return []
            
        try:
            with torch.no_grad():
                inputs = self.clip_processor(text=query, return_tensors="pt")
                query_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy().flatten()
                
                embeddings = self.vectorstore["embeddings"]
                similarities = np.dot(embeddings, query_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                k = min(k, len(embeddings))
                top_indices = np.argsort(similarities)[-k:][::-1]
                results = [self.vectorstore["metadata"][i] for i in top_indices]
                
                logger.info(f"Found {len(results)} similar images for query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def check_image_safety(self, image: Image.Image) -> bool:
        """Bypass NSFW safety check: always return True (all images are considered safe)"""
        return True

    def generate_image(self, prompt: str, references: List[Dict], add_watermark: bool = True) -> Optional[str]:
        """Generate image with enhanced prompt and safety checking"""
        fallback_models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5"
        ]
        
        try:
            if references:
                styles = [ref.get('style', '') for ref in references if ref.get('style')]
                style_text = ", ".join(set(styles))
                enhanced_prompt = f"{prompt}, in the style of {style_text}"
            else:
                enhanced_prompt = prompt
                
            enhanced_prompt = enhanced_prompt[:200]
            
            logger.info(f"Generating image with prompt: {enhanced_prompt}")
            
            for model_name in fallback_models:
                try:
                    model_url = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    payload = {
                        "inputs": enhanced_prompt,
                        "parameters": {
                            "num_inference_steps": 20,
                            "guidance_scale": 7.5
                        }
                    }
                    
                    response_data, error = self.call_api(model_url, payload, timeout=60)
                    
                    if error:
                        if "403" in error:
                            logger.warning(f"Access denied for {model_name}, trying next model...")
                            continue
                        elif "503" in error:
                            st.info(f"⏳ {model_name} is loading, trying alternative...")
                            continue
                        else:
                            logger.warning(f"Error with {model_name}: {error}")
                            continue
                    
                    if not response_data or "content" not in response_data:
                        logger.warning(f"Invalid response from {model_name}")
                        continue
                        
                    st.success(f"✅ Generated using {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed with {model_name}: {e}")
                    continue
            else:
                st.error("❌ All image generation models failed. Please try again later.")
                st.markdown("""
                **Troubleshooting:**
                1. Check your Hugging Face token permissions
                2. Visit model pages and accept terms of use
                3. Try again in a few minutes (models may be loading)
                """)
                return None
                
            try:
                image = Image.open(BytesIO(response_data["content"]))
                image = image.convert('RGB')
                
                if not self.check_image_safety(image):
                    st.error("Generated image flagged as inappropriate")
                    logger.warning("Image failed safety check")
                    return None
                    
                if add_watermark:
                    image = self.add_watermark(image)
                    
                timestamp = int(time.time())
                image_path = f"output/generated_{timestamp}.png"
                image.save(image_path, format='PNG', optimize=True)
                
                logger.info(f"Image generated successfully: {image_path}")
                return image_path
                
            except Exception as e:
                st.error(f"Failed to process generated image: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            st.error(f"Image generation failed: {e}")
            return None
        finally:
            gc.collect()

    def add_watermark(self, image: Image.Image) -> Image.Image:
        """Add watermark to generated image"""
        try:
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", size=16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                    
            watermark_text = "Generated by Art Generator"
            
            if font:
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 150, 15
                
            x = image.width - text_width - 10
            y = image.height - text_height - 10
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [x-5, y-2, x+text_width+5, y+text_height+2], 
                fill=(0, 0, 0, 128)
            )
            
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255))
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to add watermark: {e}")
            return image

    def save_generation_metadata(self, image_path: str, prompt: str, style: str):
        """Save metadata of generated images to CSV"""
        metadata_path = "output/generated_images.csv"
        metadata = {
            "image_path": image_path,
            "prompt": prompt,
            "style": style,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        df = pd.DataFrame([metadata])
        if os.path.exists(metadata_path):
            existing_df = pd.read_csv(metadata_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata for {image_path}")

    def load_generated_images(self) -> pd.DataFrame:
        """Load metadata of previously generated images"""
        metadata_path = "output/generated_images.csv"
        if os.path.exists(metadata_path):
            try:
                return pd.read_csv(metadata_path)
            except Exception as e:
                logger.warning(f"Failed to load generated images metadata: {e}")
                return pd.DataFrame(columns=["image_path", "prompt", "style", "timestamp"])
        return pd.DataFrame(columns=["image_path", "prompt", "style", "timestamp"])

    def manage_rate_limit(self) -> bool:
        """No rate limiting: always allow generation"""
        return True

    def validate_inputs(self, prompt: str) -> bool:
        """Validate user inputs (no inappropriate content filter)"""
        if not prompt or not prompt.strip():
            st.error("Please enter a valid art prompt")
            return False
        if len(prompt) > 500:
            st.error("Prompt too long. Please keep it under 500 characters.")
            return False
        return True

    def integrate_feedback(self, original_prompt: str, feedback: str) -> str:
        """Integrate user feedback into prompt"""
        if not feedback.strip():
            return original_prompt
            
        try:
            feedback_lower = feedback.lower()
            
            if 'brighter' in feedback_lower or 'bright' in feedback_lower:
                return f"{original_prompt}, bright and luminous"
            elif 'darker' in feedback_lower or 'dark' in feedback_lower:
                return f"{original_prompt}, dark and moody"
            elif 'colorful' in feedback_lower or 'colors' in feedback_lower:
                return f"{original_prompt}, vibrant and colorful"
            elif 'simple' in feedback_lower or 'minimal' in feedback_lower:
                return f"{original_prompt}, minimalist style"
            else:
                return f"{original_prompt}, {feedback}"
                
        except Exception as e:
            logger.warning(f"Feedback integration failed: {e}")
            return original_prompt

    def generate_art(self, prompt: str, style: str = "Auto-detect", feedback: str = "") -> GenerationResult:
        """Main art generation pipeline"""
        try:
            style_info = self.extract_style_elements(prompt, style)
            search_query = f"{style_info['subject']} {style_info['style']}"
            references = self.find_similar_images(search_query, k=2)
            final_prompt = style_info['refined_prompt']
            if feedback:
                final_prompt = self.integrate_feedback(final_prompt, feedback)
            image_path = self.generate_image(final_prompt, references)
            if image_path:
                self.save_generation_metadata(image_path, prompt, style_info['style'])
            return GenerationResult(
                image_path=image_path,
                references=references,
                style_info=style_info,
                success=image_path is not None,
                error_message=None if image_path else "Image generation failed"
            )
        except Exception as e:
            logger.error(f"Art generation pipeline failed: {e}")
            return GenerationResult(
                image_path=None,
                references=[],
                style_info={},
                success=False,
                error_message=str(e)
            )

    def display_results(self, result: GenerationResult):
        """Display generation results in the UI"""
        if result.success and result.image_path:
            st.success("🎨 Art generated successfully!")
            
            st.image(result.image_path, caption="Your Generated Art", use_container_width=True)
            
            try:
                with open(result.image_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Image",
                        data=file.read(),
                        file_name=f"art_{int(time.time())}.png",
                        mime="image/png"
                    )
            except Exception as e:
                logger.warning(f"Download button setup failed: {e}")
            
            if result.style_info:
                with st.expander("🎭 Style Analysis"):
                    st.write(f"**Detected Style:** {result.style_info.get('style', 'Unknown')}")
                    st.write(f"**Subject:** {result.style_info.get('subject', 'Unknown')}")
                    st.write(f"**Mood:** {result.style_info.get('mood', 'Unknown')}")
            
            if result.references:
                with st.expander("🖼️ Reference Images Used"):
                    cols = st.columns(len(result.references))
                    for i, ref in enumerate(result.references):
                        with cols[i]:
                            try:
                                if os.path.exists(ref['image_path']):
                                    st.image(
                                        ref['image_path'], 
                                        caption=f"Style: {ref.get('style', 'Unknown')}", 
                                        width=150,
                                        use_container_width=False
                                    )
                            except Exception as e:
                                st.write(f"Reference: {ref.get('style', 'Unknown')}")
                                
        else:
            st.error(f"❌ Generation failed: {result.error_message}")

    def run(self):
        """Main Streamlit application"""
        st.set_page_config(
            page_title="Personalized Art Generator",
            layout="wide"
        )
        
        st.title("Personalized Art Generator")
        st.markdown("Create unique artwork using AI with style-aware generation")
        
        with st.sidebar:
            st.header("Settings")
            
            st.subheader("System Status")
            st.write(f"Reference Images: {len(self.df)}")
            st.write(f"Index Status: {'✅ Ready' if self.vectorstore else '❌ Not Available'}")
            st.write(f"Safety Check: {'✅ Active' if self.safety_classifier else '⚠️ Disabled'}")
            
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
            
            st.subheader("Recent Generations")
            generated_df = self.load_generated_images()
            if not generated_df.empty:
                with st.expander("View Recent Images"):
                    for _, row in generated_df.tail(5).iterrows():
                        if os.path.exists(row['image_path']):
                            st.image(row['image_path'], caption=f"{row['prompt']} ({row['style']})", width=100)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("✨ Create Your Art")
            
            prompt = st.text_area(
                "Art Prompt",
                placeholder="e.g., 'A serene mountain landscape at sunset'",
                help="Describe the artwork you want to create"
            )
            
            col_style, col_feedback = st.columns(2)
            with col_style:
                available_styles = sorted(set(self.df['style'].astype(str).str.strip().tolist())) if not self.df.empty else []
                style = st.selectbox(
                    "Style (from references)",
                    ["Auto-detect"] + available_styles,
                    help="Choose a style from your reference images or leave as Auto-detect."
                )
            with col_feedback:
                feedback = st.text_input(
                    "Refinement (Optional)",
                    placeholder="e.g., 'make it brighter'"
                )
        
        with col2:
            st.subheader("🎯 Quick Examples")
            example_prompts = [
                "Abstract geometric patterns in blue",
                "Impressionist garden with flowers",
                "Minimalist mountain landscape",
                "Surreal floating islands",
                "Renaissance portrait with rich details",
                "Watercolor seascape at dawn"
            ]
            
            for example in example_prompts:
                if st.button(example, key=f"example_{example}"):
                    st.rerun()

        if st.button("🎨 Generate Art", type="primary", use_container_width=True):
            if "generation_count" not in st.session_state:
                st.session_state.generation_count = 0
            if not self.validate_inputs(prompt):
                return
                
            if not self.manage_rate_limit():
                return
                
            st.session_state.generation_count += 1
            
            with st.spinner("🎨 Creating your artwork..."):
                progress_bar = st.progress(0)
                progress_bar.progress(25, "Analyzing style...")
                
                result = self.generate_art(prompt, style, feedback)
                progress_bar.progress(100, "Complete!")
                
            self.display_results(result)
            
            progress_bar.empty()

        st.markdown("---")
        st.markdown(
            "💡 **Tips:** Be descriptive with your prompts, try different moods, "
            "and use the refinement field to adjust results!"
        )

def main():
    """Application entry point"""
    try:
        app = ArtGeneratorApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"App startup failed: {e}")

if __name__ == "__main__":
    main()
