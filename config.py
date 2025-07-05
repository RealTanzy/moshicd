# Creative Dashboard Configuration

This file contains customizable settings for the Creative Dashboard application.

## Default Settings (modify creative_dashboard.py to implement these)

### UI Configuration
PAGE_TITLE = "Creative Dashboard - Web Design Strategy Generator"
PAGE_ICON = "🎨"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

### AI Providers Configuration
# Add or remove providers as needed
DEFAULT_PROVIDERS = {
    "Groq": {
        "models": ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        "recommended": True,
        "description": "Fast inference, cost-effective"
    },
    "Anthropic Claude": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
        "recommended": True,
        "description": "High-quality analysis and reasoning"
    },
    "Cohere": {
        "models": ["command-r-plus", "command-r", "command"],
        "recommended": False,
        "description": "Enterprise-focused language models"
    },
    "OpenAI": {
        "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
        "recommended": False,
        "description": "Industry standard, higher cost"
    }
}

### Default API Settings
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 30

### File Upload Settings
MAX_FILE_SIZE_MB = 10
ALLOWED_PDF_EXTENSIONS = ["pdf"]
PDF_PROCESSING_TIMEOUT = 30

### Export Settings
DEFAULT_EXPORT_FORMATS = ["json", "markdown"]
INCLUDE_METADATA = True
INCLUDE_TIMESTAMP = True

### Session State Keys
SESSION_KEYS = {
    "brief_content": "",
    "ia_generated": "",
    "design_strategy": "",
    "api_provider": "Groq",
    "selected_model": "",
    "api_key": ""
}

### UI Text Customization
UI_TEXT = {
    "main_title": "🎨 Creative Dashboard",
    "subtitle": "Web Design Strategy Generator",
    "brief_section": "📝 Client Brief Input", 
    "ia_section": "🏗️ Information Architecture",
    "design_section": "🎨 Design Strategy & Parameters",
    "config_section": "⚙️ Configuration",
    "export_section": "📤 Export Options"
}

### Prompt Templates (Advanced Users)
# These can be modified to change AI output style and focus

IA_SYSTEM_PROMPT = """You are an expert web design strategist and information architect. Your task is to analyze client briefs and create comprehensive information architecture strategies.

Based on the provided client brief, generate a detailed information architecture that includes:

1. **Site Structure & Hierarchy**
   - Main navigation categories
   - Sub-navigation elements
   - Content organization strategy

2. **User Journey Mapping**
   - Primary user paths
   - Key conversion funnels
   - User experience flow

3. **Content Strategy**
   - Required content types
   - Content prioritization
   - SEO considerations

4. **Technical Requirements**
   - Recommended features and functionality
   - Integration needs
   - Performance considerations

5. **Business Growth Strategies**
   - Conversion optimization opportunities
   - User engagement tactics
   - Scalability recommendations

Please provide actionable, specific recommendations that align with modern web design best practices."""

DESIGN_SYSTEM_PROMPT = """You are a senior web design strategist specializing in visual design and user experience. Based on the provided information architecture, create a comprehensive design strategy.

Generate detailed design recommendations covering:

1. **Visual Design System**
   - Color palette suggestions
   - Typography hierarchy
   - Visual style direction
   - Brand integration approaches

2. **Layout & Grid Systems**
   - Page layout strategies
   - Responsive design considerations
   - Component organization

3. **User Interface Elements**
   - Navigation design patterns
   - Interactive elements
   - Form design approaches
   - Call-to-action strategies

4. **User Experience Optimization**
   - Accessibility considerations
   - Mobile-first design approach
   - Performance optimization strategies
   - Conversion rate optimization

5. **Implementation Guidelines**
   - Design system documentation
   - Developer handoff recommendations
   - Testing and iteration strategies

Provide specific, actionable design recommendations that will help the business achieve its goals while delivering excellent user experience."""

## Environment Variables (Recommended for Production)
# Set these in your environment instead of entering API keys in the UI

# GROQ_API_KEY=your_groq_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here  
# COHERE_API_KEY=your_cohere_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

## Customization Notes

### Adding New AI Providers
To add a new AI provider:
1. Install the provider's Python SDK
2. Add import statements in creative_dashboard.py
3. Add provider configuration to api_providers dictionary
4. Implement API calling logic in call_ai_api method

### Modifying Output Style
To change the AI output style:
1. Modify the IA_SYSTEM_PROMPT or DESIGN_SYSTEM_PROMPT above
2. Update the prompts in the generate_information_architecture and generate_design_strategy methods
3. Adjust the output formatting in the Streamlit interface

### UI Customization
To customize the user interface:
1. Modify the UI_TEXT dictionary above
2. Update the Streamlit components in creative_dashboard.py
3. Add custom CSS using st.markdown with unsafe_allow_html=True

### Adding New Export Formats
To add new export formats:
1. Create new export functions in creative_dashboard.py
2. Add export buttons in the export section
3. Update the DEFAULT_EXPORT_FORMATS list above
