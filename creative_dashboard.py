import streamlit as st
import json
import os
from typing import Optional, Dict, Any, List
import requests
from datetime import datetime
import base64
import urllib.parse
from io import BytesIO
import time
import hashlib
import re
import streamlit as st


# Import AI API clients
try:
    import groq
    from groq import Groq
except ImportError:
    groq = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None

try:
    import cohere
    from cohere import Client as CohereClient
except ImportError:
    cohere = None

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

# Import PDF processing libraries
try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfFileReader
    except ImportError:
        PdfReader = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Import web scraping and image analysis libraries
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from PIL import Image, ImageDraw
    import numpy as np
    from sklearn.cluster import KMeans
except ImportError:
    Image = None
    np = None
    KMeans = None

# Set page configuration
st.set_page_config(
    page_title="🌀 Vision Expansion Creative Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create columns
col1, col2 = st.columns([1, 5])

# Put the logo in the first column (left side)
with col1:
    st.image("logo.png", width=100)  # adjust width

with col2:
    st.title("Moshi Moshi")
    
# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_URLS = 5
REQUEST_TIMEOUT = 15
CACHE_TTL = 3600  # 1 hour

# Initialize session state variables
session_vars = [
    "brief_content", "ia_generated", "design_strategy", "futuristic_ia",
    "designer_references", "user_creative_input", "file_uploads", 
    "reference_analysis", "creative_prompt_variants", "company_name",
    "api_provider", "selected_model", "last_analysis_hash", "analysis_cache"
]

for var in session_vars:
    if var not in st.session_state:
        if var in ["creative_prompt_variants", "file_uploads"]:
            st.session_state[var] = []
        elif var in ["analysis_cache"]:
            st.session_state[var] = {}
        else:
            st.session_state[var] = ""

class VisionExpansionDashboard:
    def __init__(self):
        self.api_providers = {
            "Groq": {
                "models": ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
                "client_class": Groq if groq else None,
                "available": groq is not None
            },
            "Anthropic Claude": {
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                "client_class": Anthropic if anthropic else None,
                "available": anthropic is not None
            },
            "Cohere": {
                "models": ["command-r-plus", "command-r", "command"],
                "client_class": CohereClient if cohere else None,
                "available": cohere is not None
            },
            "OpenAI": {
                "models": ["gpt-4o", "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
                "client_class": OpenAI if openai else None,
                "available": openai is not None
            }
        }

    def validate_input(self, text: str, max_length: int = 10000) -> str:
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            return ""
        
        # Remove potentially harmful content
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()

    def validate_url(self, url: str) -> bool:
        """Validate URL format and safety."""
        if not url or not isinstance(url, str):
            return False
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))

    @st.cache_data(ttl=CACHE_TTL)
    def extract_text_from_pdf(_self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF file content with caching."""
        try:
            if PdfReader:
                reader = PdfReader(BytesIO(file_content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            elif fitz:
                doc = fitz.open(stream=file_content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text
            else:
                st.error("PDF processing libraries not available.")
                return ""
        except Exception as e:
            st.error(f"Error extracting text from PDF {filename}: {str(e)}")
            return ""

    def call_ai_api(self, provider: str, api_key: str, model: str, messages: list, system_prompt: str = "") -> str:
        """Call the selected AI API provider with enhanced error handling."""
        if not api_key or not api_key.strip():
            return "Error: API key is required."
        
        try:
            if provider == "Groq" and self.api_providers["Groq"]["available"]:
                client = Groq(api_key=api_key)
                groq_messages = []
                if system_prompt:
                    groq_messages.append({"role": "system", "content": system_prompt})
                groq_messages.extend(messages)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=groq_messages,
                    max_tokens=4000,
                    temperature=0.8,
                    timeout=30
                )
                return response.choices[0].message.content

            elif provider == "Anthropic Claude" and self.api_providers["Anthropic Claude"]["available"]:
                client = Anthropic(api_key=api_key)
                user_message = messages[-1]["content"] if messages else ""
                if system_prompt:
                    user_message = f"{system_prompt}\n\n{user_message}"
                
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": user_message}]
                )
                return response.content[0].text

            elif provider == "Cohere" and self.api_providers["Cohere"]["available"]:
                client = CohereClient(api_key=api_key)
                user_message = messages[-1]["content"] if messages else ""
                if system_prompt:
                    user_message = f"{system_prompt}\n\n{user_message}"
                
                response = client.generate(
                    model=model,
                    prompt=user_message,
                    max_tokens=4000,
                    temperature=0.8
                )
                return response.generations[0].text

            elif provider == "OpenAI" and self.api_providers["OpenAI"]["available"]:
                client = OpenAI(api_key=api_key)
                openai_messages = []
                if system_prompt:
                    openai_messages.append({"role": "system", "content": system_prompt})
                openai_messages.extend(messages)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_tokens=4000,
                    temperature=0.8,
                    timeout=30
                )
                return response.choices[0].message.content
            else:
                return f"Provider {provider} is not available or configured."
                
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return "Error: API rate limit exceeded. Please wait and try again."
            elif "quota" in error_msg.lower():
                return "Error: API quota exceeded. Please check your billing."
            elif "authentication" in error_msg.lower():
                return "Error: Invalid API key. Please check your credentials."
            else:
                return f"Error calling {provider} API: {error_msg}"

    @st.cache_data(ttl=CACHE_TTL)
    def scrape_reference_url(_self, url: str) -> Dict[str, Any]:
        """Advanced web scraping for design references with caching."""
        if not _self.validate_url(url):
            return {"error": "Invalid URL format"}
        
        try:
            if not BeautifulSoup:
                return {"error": "BeautifulSoup not available."}
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            
            # Check content size
            if len(response.content) > 5 * 1024 * 1024:  # 5MB limit
                return {"error": "Content too large"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract comprehensive design information
            title = soup.find('title')
            title = title.get_text().strip() if title else "No title found"
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract design-specific elements
            design_elements = {
                "navigation_patterns": [],
                "layout_structure": "",
                "color_schemes": [],
                "typography_styles": [],
                "interaction_elements": []
            }
            
            # Analyze navigation patterns with better selectors
            nav_selectors = ['nav', 'header', '[class*="nav"]', '[class*="menu"]', '[id*="nav"]', '[id*="menu"]']
            for selector in nav_selectors:
                nav_elements = soup.select(selector)
                for nav in nav_elements[:3]:  # Limit to prevent overflow
                    nav_text = nav.get_text(strip=True)[:100]
                    if nav_text and nav_text not in design_elements["navigation_patterns"]:
                        design_elements["navigation_patterns"].append(nav_text)
            
            # Extract CSS colors with improved regex
            css_colors = []
            style_tags = soup.find_all(['style'])
            for style in style_tags:
                css_content = style.get_text()
                # Match hex colors, rgb, hsl
                color_patterns = [
                    r'#[0-9a-fA-F]{3,6}',
                    r'rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)',
                    r'hsl\(\s*\d+\s*,\s*\d+%\s*,\s*\d+%\s*\)'
                ]
                for pattern in color_patterns:
                    matches = re.findall(pattern, css_content)
                    css_colors.extend(matches)
            
            # Extract layout structure with better analysis
            main_content = soup.find('main') or soup.find('body')
            if main_content:
                layout_classes = main_content.get('class', [])
                design_elements["layout_structure"] = ' '.join(layout_classes) if layout_classes else "Standard layout"
            
            # Extract meta information
            meta_tags = soup.find_all('meta')
            meta_info = {}
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_info[name] = content
            
            return {
                "url": url,
                "title": title[:200],  # Limit title length
                "description": description[:500],  # Limit description length
                "design_elements": design_elements,
                "css_colors": list(set(css_colors))[:15],  # Increase color limit but deduplicate
                "meta_info": meta_info,
                "scraped_at": datetime.now().isoformat(),
                "content_size": len(response.content)
            }
            
        except requests.exceptions.Timeout:
            return {"error": f"Request timeout for {url}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request error for {url}: {str(e)}"}
        except Exception as e:
            return {"error": f"Error scraping {url}: {str(e)}"}

    def analyze_uploaded_file(self, file) -> Dict[str, Any]:
        """Advanced file analysis for creative assets with enhanced features."""
        try:
            # Check file size
            if file.size > MAX_FILE_SIZE:
                return {"error": f"File {file.name} is too large (max {MAX_FILE_SIZE/1024/1024:.1f}MB)"}
            
            file_info = {
                "filename": file.name,
                "file_type": file.type,
                "size": file.size,
                "size_mb": round(file.size / 1024 / 1024, 2),
                "analysis": {}
            }
            
            if file.type.startswith('image/'):
                if Image and np and KMeans:
                    try:
                        img = Image.open(file)
                        original_size = img.size
                        
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Create thumbnail for processing
                        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                        
                        # Validate image array
                        if img_array.size == 0:
                            return {"error": f"Invalid image data in {file.name}"}
                        
                        pixels = img_array.reshape(-1, 3)
                        
                        # Extract dominant colors with error handling
                        n_colors = min(8, len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1]))))))
                        if n_colors > 1:
                            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                            kmeans.fit(pixels)
                            
                            colors = kmeans.cluster_centers_.astype(int)
                            color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
                            
                            dominant_colors = []
                            for i, color in enumerate(colors):
                                hex_color = "#{:02x}{:02x}{:02x}".format(
                                    max(0, min(255, color[0])),
                                    max(0, min(255, color[1])),
                                    max(0, min(255, color[2]))
                                )
                                dominant_colors.append({
                                    "hex": hex_color,
                                    "rgb": [int(c) for c in color],
                                    "percentage": float(color_percentages[i])
                                })
                            
                            # Sort by percentage
                            dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
                        else:
                            dominant_colors = []
                        
                        # Calculate additional image metrics
                        brightness = np.mean(img_array)
                        contrast = np.std(img_array)
                        
                        file_info["analysis"] = {
                            "original_dimensions": original_size,
                            "processed_dimensions": img.size,
                            "dominant_colors": dominant_colors,
                            "aspect_ratio": round(original_size[0] / original_size[1], 3),
                            "brightness": round(brightness, 2),
                            "contrast": round(contrast, 2),
                            "color_complexity": len(dominant_colors)
                        }
                        
                    except Exception as e:
                        file_info["analysis"] = {"error": f"Image processing error: {str(e)}"}
                else:
                    file_info["analysis"] = {"error": "Image analysis libraries not available"}
            
            elif file.type == 'application/pdf':
                # Read file content for caching
                file_content = file.read()
                file.seek(0)  # Reset file pointer
                
                text_content = self.extract_text_from_pdf(file_content, file.name)
                
                # Basic text analysis
                word_count = len(text_content.split()) if text_content else 0
                char_count = len(text_content) if text_content else 0
                
                file_info["analysis"] = {
                    "text_content": text_content[:800] + "..." if len(text_content) > 800 else text_content,
                    "content_length": char_count,
                    "word_count": word_count,
                    "estimated_pages": max(1, word_count // 250) if word_count > 0 else 0
                }
            
            return file_info
            
        except Exception as e:
            return {"error": f"Error analyzing file {file.name}: {str(e)}"}

    def extract_company_name(self, brief_content: str, provider: str, api_key: str, model: str) -> str:
        """Extract company name from client brief using AI with validation."""
        if not brief_content or not brief_content.strip():
            return "Company not specified"
        
        # Validate input
        brief_content = self.validate_input(brief_content, 2000)
        
        system_prompt = """Extract the primary company or organization name from the provided text. Focus on the main business entity requesting the website design. 

Rules:
- Return only the company name (no additional text)
- If multiple companies mentioned, return the client/main company
- Include business entity types (Inc, LLC, Corp, Ltd, etc.) if present
- Return 'Company not specified' if unclear or no company mentioned
- Maximum 100 characters"""

        messages = [{"role": "user", "content": f"Extract company name from: {brief_content}"}]
        result = self.call_ai_api(provider, api_key, model, messages, system_prompt)
        
        # Validate and clean result
        result = self.validate_input(result, 100)
        return result if result else "Company not specified"

    def generate_futuristic_ia(self, brief_content: str, designer_references: str, user_creative_input: str, file_analysis: str, provider: str, api_key: str, model: str) -> str:
        """Generate boundary-pushing, futuristic Information Architecture with validation."""
        
        # Validate inputs
        brief_content = self.validate_input(brief_content, 5000)
        designer_references = self.validate_input(designer_references, 2000)
        user_creative_input = self.validate_input(user_creative_input, 2000)
        file_analysis = self.validate_input(file_analysis, 3000)
        
        system_prompt = """You are a visionary Information Architect specializing in FUTURISTIC, BOUNDARY-PUSHING website designs. Your mission is to create revolutionary IA that breaks ALL conventional rules and pushes the boundaries of imagination.

THINK BEYOND TRADITIONAL LAYOUTS. Design Information Architecture that includes:

🌀 **EXPERIMENTAL NAVIGATION PATTERNS**
- Floating navigation spheres that respond to user emotions
- Time-based menu systems that change throughout the day
- Gesture-controlled 3D navigation spaces
- AI-powered predictive navigation that anticipates user needs
- Gamified quest-based site exploration

🚀 **INNOVATIVE INTERACTION FLOWS**
- Multi-dimensional scrolling experiences (X, Y, Z axes)
- Parallax storytelling with interactive narrative branches
- Voice-activated section transitions
- Biometric-responsive content adaptation
- Collaborative real-time user experiences

🔮 **UNEXPECTED SECTIONS & FEATURES**
- Living data visualizations that evolve in real-time
- Augmented reality overlay integration points
- AI companion chat zones embedded in designs
- Micro-interaction playgrounds
- Contextual ambient soundscapes
- Dynamic mood-responsive color systems

💫 **FUTURISTIC STRUCTURAL CONCEPTS**
- Fractal information hierarchies
- Quantum navigation states (exist in multiple places simultaneously)
- Temporal content layers (past, present, future views)
- Emotional journey mapping with sentiment-triggered transitions
- Collective intelligence integration (crowdsourced content flows)

BREAK THE RULES. CREATE THE IMPOSSIBLE. Design an IA that makes users say "I've never seen anything like this before."

Integrate all provided references and creative inputs to create a cohesive, revolutionary vision.

IMPORTANT: Provide specific, actionable recommendations with clear structure and organized sections."""

        context_content = f"""
**Project Brief:**
{brief_content}

**Designer References & Inspiration:**
{designer_references}

**Creative Input & Vision:**
{user_creative_input}

**File Analysis & Assets:**
{file_analysis}
"""

        messages = [{"role": "user", "content": f"Create a futuristic, boundary-pushing Information Architecture based on:\n\n{context_content}"}]
        return self.call_ai_api(provider, api_key, model, messages, system_prompt)

    def generate_creative_prompt_variants(self, futuristic_ia: str, all_context: str, provider: str, api_key: str, model: str) -> List[str]:
        """Generate four distinct creative prompt variants for image generation with enhanced processing."""
        
        # Validate inputs
        futuristic_ia = self.validate_input(futuristic_ia, 8000)
        all_context = self.validate_input(all_context, 5000)
        
        system_prompt = """You are a master prompt engineer specializing in creating revolutionary AI image generation prompts. Based on the futuristic Information Architecture and project context, create 4 DISTINCT prompt variants for AI image generators (DALL-E, Midjourney, Ideogram, etc.).

Each variant should be 75-150 words and follow these creative approaches:

**VARIANT 1: MINIMALIST FUTURISM**
- Clean, space-focused design with subtle futuristic elements
- Emphasis on white space, geometric precision, and refined simplicity
- Gentle technological integration, sophisticated color palettes
- Professional, elegant aesthetic

**VARIANT 2: BOLD EXPERIMENTAL** 
- Dramatic, boundary-pushing visuals with maximum creative impact
- Vibrant colors, unexpected compositions, artistic risk-taking
- Shocking visual elements that challenge conventional design
- High contrast, dynamic energy

**VARIANT 3: INTERACTIVE NARRATIVE**
- Story-driven layouts with gamified, cinematic experiences
- Journey-based design with clear narrative progression
- Adventure-like exploration elements and character-driven interfaces
- Immersive, engaging visual storytelling

**VARIANT 4: CONTEXTUAL INTELLIGENCE**
- Adaptive, smart designs that respond to user behavior
- AI-powered personalization and environmental awareness
- Context-aware interfaces with predictive elements
- Intelligent, responsive design systems

REQUIREMENTS:
- Each prompt must be immediately usable in AI image generators
- Include specific visual details, technical parameters, and style directions
- Use precise descriptive language for optimal AI interpretation
- Include rendering quality specifications (4K, photorealistic, etc.)
- Add composition details (wide shot, close-up, isometric view, etc.)

Return ONLY the 4 prompts separated by "=== VARIANT X ===" markers."""

        messages = [{"role": "user", "content": f"Generate 4 creative prompt variants based on:\n\nFUTURISTIC IA:\n{futuristic_ia}\n\nFULL CONTEXT:\n{all_context}"}]
        
        response = self.call_ai_api(provider, api_key, model, messages, system_prompt)
        
        # Enhanced variant processing
        variants = response.split("=== VARIANT")
        processed_variants = []
        
        for i, variant in enumerate(variants):
            if variant.strip():
                clean_variant = variant.strip()
                
                # Remove variant number prefix and clean formatting
                lines = clean_variant.split('\n')
                if len(lines) > 1:
                    # Remove first line if it contains variant number or markers
                    first_line = lines[0].strip()
                    if (any(char in first_line for char in ['1', '2', '3', '4', '===']) and 
                        len(first_line) < 20):
                        clean_variant = '\n'.join(lines[1:]).strip()
                
                # Additional cleaning
                clean_variant = re.sub(r'^[^\w]*', '', clean_variant)  # Remove leading non-word chars
                clean_variant = re.sub(r'\s+', ' ', clean_variant)  # Normalize whitespace
                
                # Validate length and content
                if clean_variant and len(clean_variant) > 30 and len(clean_variant) < 1000:
                    processed_variants.append(clean_variant)
        
        # Ensure we have exactly 4 variants with fallbacks
        variant_templates = [
            "Minimalist futuristic website interface with clean geometric layouts, subtle technological elements, white space emphasis, and sophisticated color palette, photorealistic rendering, 4K quality",
            "Bold experimental website design with vibrant colors, dramatic compositions, boundary-pushing visual elements, high contrast dynamic energy, artistic risk-taking, professional photography style",
            "Interactive narrative website interface with story-driven layouts, gamified elements, cinematic experiences, adventure-like exploration features, immersive visual storytelling, detailed rendering",
            "Contextual intelligent website design with adaptive interfaces, AI-powered personalization, responsive design systems, environmental awareness, smart user behavior adaptation, modern aesthetic"
        ]
        
        while len(processed_variants) < 4:
            template_index = len(processed_variants)
            processed_variants.append(variant_templates[template_index])
        
        return processed_variants[:4]

    def create_platform_urls(self, prompt: str) -> Dict[str, str]:
        """Generate pre-filled URLs for different AI platforms with enhanced encoding."""
        # Clean and encode the prompt
        clean_prompt = self.validate_input(prompt, 2000)
        encoded_prompt = urllib.parse.quote_plus(clean_prompt)
        
        # Truncate if too long for URL
        if len(encoded_prompt) > 2000:
            # Truncate and re-encode
            truncated = clean_prompt[:500] + "..."
            encoded_prompt = urllib.parse.quote_plus(truncated)
        
        return {
            "chatgpt": f"https://chat.openai.com/?q={encoded_prompt}",
            "claude": f"https://claude.ai/chat?q={encoded_prompt}",
            "lmarena": f"https://lmarena.ai/?prompt={encoded_prompt}",
            "ideogram": f"https://ideogram.ai/?prompt={encoded_prompt}",
            "midjourney": f"https://www.midjourney.com/prompt?text={encoded_prompt}"
        }

def display_color_palette(colors: List[Dict], title: str = "Color Palette"):
    """Display color palette with enhanced visualization."""
    if not colors:
        return
    
    st.subheader(title)
    
    # Create color swatches
    cols = st.columns(min(len(colors), 5))
    for i, color in enumerate(colors[:5]):
        with cols[i]:
            # Color swatch using HTML/CSS
            color_html = f"""
            <div style="
                width: 60px; 
                height: 60px; 
                background-color: {color['hex']}; 
                border-radius: 8px; 
                border: 2px solid #ddd;
                margin: 5px auto;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            "></div>
            <div style="text-align: center; font-size: 12px; margin-top: 5px;">
                <strong>{color['hex']}</strong><br>
                {color.get('percentage', 0)*100:.1f}%
            </div>
            """
            st.markdown(color_html, unsafe_allow_html=True)

def create_progress_bar(step: int, total_steps: int, description: str):
    """Create an enhanced progress indicator."""
    progress = step / total_steps
    st.progress(progress)
    st.caption(f"Step {step}/{total_steps}: {description}")

def main():
    dashboard = VisionExpansionDashboard()

    # Enhanced CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .creative-fuel-box {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        color: white;
    }
    .variant-box {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .redirect-button {
        background: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        margin: 8px 5px;
        text-decoration: none;
        display: inline-block;
        transition: transform 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .redirect-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-success {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .status-error {
        background: linear-gradient(45deg, #ff6b6b 0%, #ffa8a8 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🌀 Vision Expansion Creative Dashboard</h1>
        <h3>AI-Augmented IA Generator for Boundary-Pushing Website Layouts</h3>
        <p style="opacity: 0.9; margin: 10px 0 0 0;">Transform ideas into revolutionary design concepts</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        available_providers = [provider for provider, config in dashboard.api_providers.items() if config["available"]]

        if not available_providers:
            st.error("❌ No AI API libraries are installed.")
            st.info("Install required libraries: `pip install groq anthropic cohere openai`")
            st.stop()

        # Provider selection with status indicators
        selected_provider = st.selectbox(
            "Select AI Provider",
            available_providers,
            help="Choose your preferred AI service provider"
        )
        
        available_models = dashboard.api_providers[selected_provider]["models"]
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help=f"Choose the {selected_provider} model to use"
        )

        # Enhanced API key input
        api_key = st.text_input(
            f"{selected_provider} API Key",
            type="password",
            help=f"Enter your {selected_provider} API key securely"
        )

        if api_key:
            st.success("✅ API key provided")
        else:
            st.warning(f"⚠️ Please enter your {selected_provider} API key.")

        st.markdown("---")
        
        # Enhanced settings section
        st.subheader("🛠️ Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            max_files = st.number_input("Max files", 1, 10, 5)
        with col2:
            max_urls = st.number_input("Max URLs", 1, 10, 3)
        
        # Clear data with confirmation
        if st.button("🗑️ Clear All Data", help="Clear all session data"):
            for key in session_vars:
                if key in ["creative_prompt_variants", "file_uploads"]:
                    st.session_state[key] = []
                elif key in ["analysis_cache"]:
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = ""
            st.success("All data cleared!")
            time.sleep(1)
            st.rerun()

    # Main content area with enhanced layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Project Brief & Context")

        # Enhanced input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "PDF Upload"],
            horizontal=True,
            help="Select how you want to provide your project brief"
        )

        brief_content = ""
        if input_method == "Text Input":
            brief_content = st.text_area(
                "Enter your project brief:",
                value=st.session_state.brief_content,
                height=200,
                placeholder="Describe your project vision, goals, target audience, and any specific requirements...",
                help="Provide detailed information about your project for better results"
            )
            
            # Character count
            if brief_content:
                st.caption(f"Characters: {len(brief_content)}/10,000")
                
        elif input_method == "PDF Upload":
            uploaded_file = st.file_uploader(
                "Upload project brief PDF",
                type=["pdf"],
                help=f"Upload a PDF file (max {MAX_FILE_SIZE/1024/1024:.0f}MB)"
            )
            
            if uploaded_file is not None:
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"File too large! Maximum size: {MAX_FILE_SIZE/1024/1024:.0f}MB")
                else:
                    with st.spinner("🔄 Extracting text from PDF..."):
                        file_content = uploaded_file.read()
                        uploaded_file.seek(0)
                        brief_content = dashboard.extract_text_from_pdf(file_content, uploaded_file.name)
                    
                    if brief_content:
                        st.success("✅ PDF processed successfully!")
                        brief_content = st.text_area(
                            "Extracted text (editable):",
                            value=brief_content,
                            height=200,
                            key="extracted_text",
                            help="Edit the extracted text if needed"
                        )
                    else:
                        st.error("❌ Could not extract text from PDF")

        # Update session state with validation
        if brief_content != st.session_state.brief_content:
            st.session_state.brief_content = dashboard.validate_input(brief_content)
            
            # Auto-extract company name with enhanced feedback
            if api_key and st.session_state.brief_content.strip():
                with st.spinner("🔍 Extracting company name..."):
                    company_result = dashboard.extract_company_name(
                        st.session_state.brief_content,
                        selected_provider,
                        api_key,
                        selected_model
                    )
                    st.session_state.company_name = company_result

        # Enhanced company name section
        if st.session_state.company_name:
            st.subheader("🏢 Detected Company")
            if st.session_state.company_name != "Company not specified":
                st.markdown(f'<div class="status-success"><strong>{st.session_state.company_name}</strong></div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ Company name not clearly specified in the brief")

    with col2:
        st.header("🎨 Creative Fuel Sources")

        # Enhanced Creative Fuel Source 1: Designer References
        st.markdown('<div class="creative-fuel-box">', unsafe_allow_html=True)
        st.subheader("🔗 1. Designer References")
        designer_references = st.text_area(
            "Handpicked Design Inspirations (URLs):",
            value=st.session_state.designer_references,
            height=100,
            placeholder="https://example1.com\nhttps://example2.com\n...",
            help=f"Add URLs to design inspirations (max {max_urls} URLs)"
        )
        st.session_state.designer_references = dashboard.validate_input(designer_references)
        
        # URL validation feedback
        if designer_references:
            urls = [url.strip() for url in designer_references.split('\n') if url.strip()]
            valid_urls = [url for url in urls if dashboard.validate_url(url)]
            if len(urls) > len(valid_urls):
                st.warning(f"⚠️ {len(urls) - len(valid_urls)} invalid URL(s) detected")
            st.caption(f"Valid URLs: {len(valid_urls)}/{min(len(urls), max_urls)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced Creative Fuel Source 2: User Inputs
        st.markdown('<div class="creative-fuel-box">', unsafe_allow_html=True)
        st.subheader("💭 2. User Creative Input")
        user_creative_input = st.text_area(
            "Your Creative Vision & Comments:",
            value=st.session_state.user_creative_input,
            height=100,
            placeholder="Share your creative vision, experimental ideas, boundary-pushing concepts...",
            help="Describe your creative vision and any specific ideas you want to explore"
        )
        st.session_state.user_creative_input = dashboard.validate_input(user_creative_input)
        st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced Creative Fuel Source 3: File Uploads
        st.markdown('<div class="creative-fuel-box">', unsafe_allow_html=True)
        st.subheader("📁 3. Upload Creative Assets")
        uploaded_files = st.file_uploader(
            "Moodboards, PDFs, Images, Sketches:",
            type=["jpg", "jpeg", "png", "gif", "webp", "pdf"],
            accept_multiple_files=True,
            help=f"Upload up to {max_files} files (max {MAX_FILE_SIZE/1024/1024:.0f}MB each)"
        )
        
        if uploaded_files:
            # Limit files
            if len(uploaded_files) > max_files:
                st.warning(f"⚠️ Only first {max_files} files will be processed")
                uploaded_files = uploaded_files[:max_files]
            
            st.session_state.file_uploads = uploaded_files
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
            
            # Enhanced file display with details
            for i, file in enumerate(uploaded_files[:4]):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    if file.type.startswith('image/'):
                        st.image(file, caption=file.name, width=100)
                with col_b:
                    st.caption(f"**{file.name}**")
                    st.caption(f"Type: {file.type}")
                    st.caption(f"Size: {file.size/1024:.1f} KB")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Generate Futuristic IA Section
    if st.session_state.brief_content.strip():
        st.markdown("---")
        st.header("🌀 AI-Augmented Information Architecture")
        
        # Enhanced Analyze References Button
        analyze_disabled = not api_key or (not st.session_state.designer_references.strip() and not st.session_state.file_uploads)
        
        if st.button(
            "🔍 Analyze Creative Fuel Sources",
            disabled=analyze_disabled,
            help="Analyze reference URLs and uploaded files for design insights"
        ):
            total_steps = 0
            current_step = 0
            
            # Count total steps
            if st.session_state.designer_references.strip():
                urls = [url.strip() for url in st.session_state.designer_references.split('\n') if url.strip()]
                valid_urls = [url for url in urls[:max_urls] if dashboard.validate_url(url)]
                total_steps += len(valid_urls)
            if st.session_state.file_uploads:
                total_steps += len(st.session_state.file_uploads[:max_files])
            
            if total_steps == 0:
                st.warning("⚠️ No valid references or files to analyze")
            else:
                progress_container = st.container()
                
                with st.spinner("🔄 Analyzing all creative fuel sources..."):
                    reference_analysis = ""
                    file_analysis = ""
                    
                    # Analyze references with progress
                    if st.session_state.designer_references.strip():
                        urls = [url.strip() for url in st.session_state.designer_references.split('\n') if url.strip()]
                        valid_urls = [url for url in urls[:max_urls] if dashboard.validate_url(url)]
                        
                        for url in valid_urls:
                            current_step += 1
                            with progress_container:
                                create_progress_bar(current_step, total_steps, f"Analyzing {url[:50]}...")
                            
                            scraped_data = dashboard.scrape_reference_url(url)
                            if 'error' not in scraped_data:
                                reference_analysis += f"""
**URL:** {url}
**Title:** {scraped_data.get('title', 'N/A')}
**Design Elements:** {scraped_data.get('design_elements', {})}
**Colors Found:** {len(scraped_data.get('css_colors', []))} colors
**Content Size:** {scraped_data.get('content_size', 0)} bytes

"""
                            else:
                                reference_analysis += f"**URL:** {url} - ❌ {scraped_data['error']}\n\n"
                    
                    # Analyze uploaded files with progress
                    if st.session_state.file_uploads:
                        for file in st.session_state.file_uploads[:max_files]:
                            current_step += 1
                            with progress_container:
                                create_progress_bar(current_step, total_steps, f"Analyzing {file.name}...")
                            
                            file_info = dashboard.analyze_uploaded_file(file)
                            if 'error' not in file_info:
                                file_analysis += f"""
**File:** {file_info['filename']} ({file_info['size_mb']} MB)
**Type:** {file_info['file_type']}
**Analysis:** {file_info.get('analysis', {})}

"""
                                
                                # Display color palette if available
                                if 'dominant_colors' in file_info.get('analysis', {}):
                                    colors = file_info['analysis']['dominant_colors']
                                    if colors:
                                        display_color_palette(colors, f"Colors from {file.name}")
                            else:
                                file_analysis += f"**File:** {file.name} - ❌ {file_info['error']}\n\n"
                    
                    st.session_state.reference_analysis = f"# REFERENCE ANALYSIS\n{reference_analysis}\n\n# FILE ANALYSIS\n{file_analysis}"
                
                st.markdown('<div class="status-success">✅ Creative fuel analysis completed!</div>', unsafe_allow_html=True)
                st.rerun()

        # Enhanced Generate Futuristic IA Button
        ia_disabled = not api_key or not st.session_state.brief_content.strip()
        
        if st.button(
            "🚀 Generate Futuristic IA",
            disabled=ia_disabled,
            help="Generate revolutionary Information Architecture based on all inputs"
        ):
            with st.spinner("🌟 Generating boundary-pushing Information Architecture..."):
                futuristic_ia = dashboard.generate_futuristic_ia(
                    st.session_state.brief_content,
                    st.session_state.designer_references,
                    st.session_state.user_creative_input,
                    st.session_state.reference_analysis,
                    selected_provider,
                    api_key,
                    selected_model
                )
                st.session_state.futuristic_ia = futuristic_ia
            
            st.markdown('<div class="status-success">✅ Futuristic IA generated!</div>', unsafe_allow_html=True)
            st.rerun()

        # Enhanced Display Futuristic IA
        if st.session_state.futuristic_ia:
            st.subheader("🌟 Revolutionary Information Architecture")
            
            # Check for errors in IA generation
            if st.session_state.futuristic_ia.startswith("Error:"):
                st.markdown(f'<div class="status-error">{st.session_state.futuristic_ia}</div>', unsafe_allow_html=True)
            else:
                st.markdown(st.session_state.futuristic_ia)

                # Enhanced Generate Creative Prompt Variants
                st.markdown("---")
                st.header("⚡ Generate Creative Prompt Variants")
                
                if st.button(
                    "🎭 Generate 4 Creative Prompts",
                    disabled=not api_key,
                    help="Generate optimized prompts for AI image generation tools"
                ):
                    with st.spinner("🎨 Generating creative prompt variants..."):
                        all_context = f"""
COMPANY: {st.session_state.company_name}
BRIEF: {st.session_state.brief_content}
REFERENCES: {st.session_state.designer_references}
CREATIVE INPUT: {st.session_state.user_creative_input}
ANALYSIS: {st.session_state.reference_analysis}
"""
                        
                        variants = dashboard.generate_creative_prompt_variants(
                            st.session_state.futuristic_ia,
                            all_context,
                            selected_provider,
                            api_key,
                            selected_model
                        )
                        st.session_state.creative_prompt_variants = variants
                    
                    st.markdown('<div class="status-success">✅ Creative prompt variants generated!</div>', unsafe_allow_html=True)
                    st.rerun()

                # Enhanced Display Creative Prompt Variants with Redirect Options
                if st.session_state.creative_prompt_variants:
                    st.subheader("🎨 Four Distinct Creative Variations")
                    
                    variant_names = [
                        "🔹 Minimalist Futurism",
                        "🔸 Bold Experimental", 
                        "🔷 Interactive Narrative",
                        "🔶 Contextual Intelligence"
                    ]
                    
                    for i, (name, prompt) in enumerate(zip(variant_names, st.session_state.creative_prompt_variants)):
                        with st.expander(f"{name} - Variant {i+1}", expanded=(i == 0)):
                            st.markdown(f'<div class="variant-box">', unsafe_allow_html=True)
                            st.write(prompt)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Enhanced platform URLs with more options
                            urls = dashboard.create_platform_urls(prompt)
                            
                            st.markdown("**🚀 Redirect to AI Platforms:**")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.markdown(f'<a href="{urls["chatgpt"]}" target="_blank" class="redirect-button">ChatGPT</a>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<a href="{urls["claude"]}" target="_blank" class="redirect-button">Claude</a>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f'<a href="{urls["ideogram"]}" target="_blank" class="redirect-button">Ideogram</a>', unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f'<a href="{urls["midjourney"]}" target="_blank" class="redirect-button">Midjourney</a>', unsafe_allow_html=True)
                            
                            with col5:
                                st.markdown(f'<a href="{urls["lmarena"]}" target="_blank" class="redirect-button">LM Arena</a>', unsafe_allow_html=True)
                            
                            # Enhanced download option
                            col_download, col_copy = st.columns([1, 1])
                            with col_download:
                                st.download_button(
                                    f"📥 Download Variant {i+1}",
                                    data=prompt,
                                    file_name=f"creative_variant_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    key=f"download_variant_{i}"
                                )
                            
                            with col_copy:
                                if st.button(f"📋 Copy to Clipboard", key=f"copy_variant_{i}"):
                                    st.info("Prompt copied! (Feature requires manual copy)")

                    # Enhanced Export All Options
                    st.markdown("---")
                    st.subheader("📤 Export Complete Project")
                    
                    # Create comprehensive export data
                    export_data = {
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "provider": selected_provider,
                            "model": selected_model,
                            "version": "2.0"
                        },
                        "project_info": {
                            "company_name": st.session_state.company_name,
                            "project_brief": st.session_state.brief_content,
                            "brief_length": len(st.session_state.brief_content)
                        },
                        "creative_inputs": {
                            "designer_references": st.session_state.designer_references,
                            "user_creative_input": st.session_state.user_creative_input,
                            "reference_analysis": st.session_state.reference_analysis,
                            "files_processed": len(st.session_state.file_uploads) if st.session_state.file_uploads else 0
                        },
                        "generated_content": {
                            "futuristic_ia": st.session_state.futuristic_ia,
                            "creative_prompt_variants": st.session_state.creative_prompt_variants
                        }
                    }

                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.download_button(
                            "📄 Complete JSON Export",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"vision_expansion_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                    with col2:
                        # Enhanced markdown content
                        markdown_content = f"""# Vision Expansion Creative Project

## Project Metadata
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Company:** {st.session_state.company_name}
- **AI Provider:** {selected_provider} ({selected_model})
- **Version:** 2.0

## Project Brief
{st.session_state.brief_content}

## Creative Fuel Sources

### Designer References
{st.session_state.designer_references}

### User Creative Input
{st.session_state.user_creative_input}

### Reference Analysis
{st.session_state.reference_analysis}

## Futuristic Information Architecture
{st.session_state.futuristic_ia}

## Creative Prompt Variants

{chr(10).join([f"### {name}{chr(10)}{prompt}{chr(10)}" for name, prompt in zip(variant_names, st.session_state.creative_prompt_variants)])}

---
*Generated by Vision Expansion Creative Dashboard v2.0*
"""
                        
                        st.download_button(
                            "📝 Markdown Report",
                            data=markdown_content,
                            file_name=f"vision_expansion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )

                    with col3:
                        # Individual prompts file
                        all_prompts = "\n\n" + "="*60 + "\n\n".join([
                            f"CREATIVE VARIANT {i+1}: {name}\n\n{prompt}" 
                            for i, (name, prompt) in enumerate(zip(variant_names, st.session_state.creative_prompt_variants))
                        ])
                        
                        st.download_button(
                            "🎭 All Creative Prompts",
                            data=all_prompts,
                            file_name=f"all_creative_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    with col4:
                        # IA only export
                        ia_content = f"""# Futuristic Information Architecture
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Company: {st.session_state.company_name}

{st.session_state.futuristic_ia}
"""
                        
                        st.download_button(
                            "🏗️ IA Document",
                            data=ia_content,
                            file_name=f"information_architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🌀 Vision Expansion Creative Dashboard v2.0</strong></p>
        <p>Boundary-Pushing AI-Augmented IA Generator with Enhanced Platform Integration</p>
        <p style='font-size: 12px; opacity: 0.7;'>
            Built with Streamlit • Enhanced Performance • Advanced Security • Better UX
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
