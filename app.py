import streamlit as st
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List
import time
import markdown2
import base64
from io import BytesIO
from datetime import datetime
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Create dummy classes to prevent NameError
    class canvas:
        class Canvas:
            pass

# Configure page
st.set_page_config(
    page_title="AI/ML/DL Comprehensive Learning Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.chapter-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
}

.topic-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stSelectbox {
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Chapter data structure
CHAPTERS = {
    "1. Foundations": [
        "Introduction to AI, ML, DL",
        "Types of Learning (Supervised, Unsupervised, Semi, Self, RL)",
        "Probability Theory & Bayesian Thinking",
        "Statistics for ML",
        "Linear Algebra & Vector Spaces",
        "Calculus for Optimization",
        "Information Theory (Entropy, KL Divergence)",
        "Optimization Basics (Gradient Descent & Variants)"
    ],
    "2. Data Handling & Engineering": [
        "Data Collection & Pipelines",
        "Data Cleaning & Preprocessing",
        "Feature Engineering & Selection",
        "Feature Scaling (Standardization, Normalization, Robust Scaling)",
        "Handling Missing Data & Outliers",
        "Encoding Methods (One-Hot, Target, Embeddings)",
        "Dimensionality Reduction (PCA, LDA, ICA, t-SNE, UMAP)",
        "Handling Imbalanced Data (SMOTE, Class Weights, Oversampling, Undersampling)"
    ],
    "3. Regression Models": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression",
        "Bayesian Regression",
        "Huber Regression",
        "Theil-Sen Estimator",
        "RANSAC Regression",
        "Quantile Regression",
        "Isotonic Regression",
        "Poisson Regression",
        "Negative Binomial Regression",
        "Logistic Regression (for binary/multi-class)",
        "Probit & Tobit Regression",
        "Robust Regression Methods"
    ],
    "4. Classical ML Classification & Algorithms": [
        "K-Nearest Neighbors (KNN)",
        "Naive Bayes (Gaussian, Multinomial, Bernoulli)",
        "Decision Trees",
        "Random Forests",
        "Extra Trees",
        "Gradient Boosting Machines (GBM)",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Support Vector Machines (Linear, Polynomial, RBF, Sigmoid kernels)",
        "Perceptron Algorithm",
        "Probabilistic Graphical Models (Bayesian Networks, Markov Networks)"
    ],
    "5. Clustering & Unsupervised Learning": [
        "K-Means Clustering",
        "Hierarchical Clustering (Agglomerative, Divisive)",
        "DBSCAN & HDBSCAN",
        "Gaussian Mixture Models (GMM)",
        "Mean Shift Clustering",
        "Spectral Clustering",
        "BIRCH Clustering",
        "Affinity Propagation",
        "OPTICS Clustering",
        "Self-Organizing Maps (SOM)",
        "Association Rule Learning (Apriori, FP-Growth, Eclat)"
    ],
    "6. Ensemble Methods": [
        "Bagging (Bootstrap Aggregation)",
        "Boosting (AdaBoost, Gradient Boosting)",
        "Stacking & Blending",
        "Voting Classifiers (Hard & Soft Voting)",
        "Bayesian Model Averaging",
        "Random Subspace Method"
    ],
    "7. Deep Learning (Neural Networks)": [
        "Perceptron & MLP",
        "Activation Functions (Sigmoid, Tanh, ReLU, GELU, Swish)",
        "Backpropagation Algorithm",
        "Optimizers (SGD, Adam, RMSProp, Nadam, L-BFGS, Adagrad, AdamW)",
        "Loss Functions (Cross-Entropy, Hinge, Contrastive, Triplet, Dice, Focal)",
        "Regularization (Dropout, BatchNorm, LayerNorm, Weight Decay, Early Stopping)",
        "CNNs (LeNet, AlexNet, VGG, ResNet, DenseNet, EfficientNet)",
        "RNNs (Vanilla RNN, Bidirectional RNNs)",
        "LSTMs (Basic, Bi-LSTM, Attention LSTM)",
        "GRUs",
        "Sequence-to-Sequence Models (Seq2Seq)",
        "Attention Mechanisms",
        "Transformers (Encoder, Decoder, BERT, GPT)",
        "Vision Transformers (ViT, Swin Transformer)",
        "Convolutional Attention Hybrids",
        "Autoencoders",
        "Variational Autoencoders (VAE)",
        "Generative Adversarial Networks (GANs, DCGAN, CycleGAN, StyleGAN, BigGAN)",
        "Diffusion Models (DDPM, Stable Diffusion)",
        "Capsule Networks",
        "Neural ODEs",
        "Reservoir Computing (Echo State Networks)",
        "Graph Neural Networks (GCN, GraphSAGE, GAT, GIN)"
    ],
    "8. Reinforcement Learning (RL)": [
        "Markov Decision Processes (MDPs)",
        "Dynamic Programming (Value Iteration, Policy Iteration)",
        "Monte Carlo Methods",
        "Temporal Difference Learning",
        "Q-Learning",
        "SARSA",
        "Deep Q-Networks (DQN, Double DQN, Dueling DQN)",
        "Policy Gradient Methods (REINFORCE, Actor-Critic)",
        "Advanced Policy Gradient (PPO, TRPO, A2C, A3C, DDPG, TD3, SAC)",
        "Multi-Agent Reinforcement Learning",
        "Hierarchical RL",
        "Inverse Reinforcement Learning",
        "Model-Based RL"
    ],
    "9. Advanced ML/AI Topics": [
        "Time Series Analysis (ARIMA, SARIMA, Prophet, LSTMs for forecasting)",
        "Bayesian Inference & Probabilistic Programming (PyMC, Stan, Edward)",
        "Gaussian Processes & Kernel Methods",
        "Hidden Markov Models (HMMs)",
        "Causal Inference & Do-Calculus",
        "Meta-Learning (MAML, Reptile)",
        "Few-Shot & Zero-Shot Learning",
        "Self-Supervised Learning (SimCLR, BYOL, MoCo, CLIP)",
        "Contrastive Learning",
        "Multi-Task Learning",
        "Federated Learning & Privacy Preserving ML",
        "Continual Learning (Lifelong ML)",
        "Explainable AI (LIME, SHAP, Grad-CAM, Integrated Gradients)",
        "Uncertainty Estimation in ML",
        "Out-of-Distribution (OOD) Detection",
        "Adversarial Attacks & Defenses",
        "AI Safety & Robustness"
    ],
    "10. Data Engineering for AI": [
        "Data Warehousing (ETL, ELT)",
        "Big Data Systems (Hadoop, Spark MLlib)",
        "Distributed Training (Horovod, DDP, Parameter Servers)",
        "Model Deployment (Flask, FastAPI, Streamlit, Gradio, TF Serving, TorchServe)",
        "Model Optimization & Compression (Quantization, Pruning, Distillation, ONNX, TensorRT)",
        "MLOps (CI/CD for ML, Monitoring, Drift Detection, MLflow, Kubeflow)"
    ],
    "11. AI Ethics & Applications": [
        "Fairness & Bias in AI",
        "Ethical AI & Governance",
        "AI in Healthcare, Finance, Robotics, and Edge AI",
        "Social Implications of AI (AGI, Future AI Risks)"
    ]
}

def create_beautiful_pdf_styles():
    """Create beautiful styles for ReportLab PDF."""
    styles = getSampleStyleSheet()
    
    # Custom styles with colors
    custom_styles = {
        'CustomTitle': ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.Color(0.4, 0.31, 0.64),  # Purple color
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'CustomHeading1': ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.Color(0.18, 0.22, 0.28),  # Dark gray
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.Color(0.4, 0.31, 0.64),
            borderPadding=8,
            backColor=colors.Color(0.96, 0.97, 0.99)
        ),
        'CustomHeading2': ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.Color(0.29, 0.34, 0.41),
            spaceBefore=15,
            spaceAfter=8,
            fontName='Helvetica-Bold',
            leftIndent=10,
            borderWidth=1,
            borderColor=colors.Color(0.31, 0.68, 0.87),
            borderPadding=5,
            backColor=colors.Color(0.94, 0.97, 1.0)
        ),
        'CustomBody': ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.Color(0.18, 0.22, 0.28),
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ),
        'CustomCode': ParagraphStyle(
            'CustomCode',
            parent=styles['Code'],
            fontSize=9,
            textColor=colors.Color(0.89, 0.91, 0.94),
            backColor=colors.Color(0.18, 0.22, 0.28),
            fontName='Courier',
            leftIndent=15,
            rightIndent=15,
            spaceBefore=5,
            spaceAfter=5,
            borderWidth=1,
            borderColor=colors.Color(0.4, 0.31, 0.64),
            borderPadding=8
        ),
        'CustomBullet': ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.Color(0.18, 0.22, 0.28),
            leftIndent=20,
            spaceAfter=5,
            bulletIndent=10
        ),
        'HeaderStyle': ParagraphStyle(
            'HeaderStyle',
            fontSize=10,
            textColor=colors.Color(0.4, 0.31, 0.64),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'FooterStyle': ParagraphStyle(
            'FooterStyle',
            fontSize=8,
            textColor=colors.Color(0.45, 0.55, 0.64),
            alignment=TA_CENTER,
            fontName='Helvetica'
        )
    }
    
    return custom_styles

def create_colored_table_style():
    """Create a beautiful table style with colors."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.4, 0.31, 0.64)),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.98, 0.98, 1.0)),  # Alternating rows
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.Color(0.98, 0.98, 1.0), colors.white]),
        ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.85, 0.85, 0.9))
    ])

def parse_markdown_to_reportlab(content: str, styles: dict) -> list:
    """Convert markdown content to ReportLab elements with beautiful formatting."""
    elements = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            elements.append(Spacer(1, 6))
            i += 1
            continue
            
        # Headers
        if line.startswith('## '):
            header_text = line[3:].strip()
            # Remove emoji and format
            clean_header = ''.join(c for c in header_text if not c.encode('utf-8').isalpha() or ord(c) < 127)
            elements.append(Paragraph(f"<b>{clean_header}</b>", styles['CustomHeading1']))
            elements.append(Spacer(1, 12))
            
        elif line.startswith('### '):
            header_text = line[4:].strip()
            clean_header = ''.join(c for c in header_text if not c.encode('utf-8').isalpha() or ord(c) < 127)
            elements.append(Paragraph(f"<b>{clean_header}</b>", styles['CustomHeading2']))
            elements.append(Spacer(1, 8))
            
        # Code blocks
        elif line.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            
            if code_lines:
                code_text = '\n'.join(code_lines)
                # Split long code lines
                formatted_code = []
                for code_line in code_text.split('\n'):
                    if len(code_line) > 80:
                        # Break long lines
                        while len(code_line) > 80:
                            formatted_code.append(code_line[:80])
                            code_line = '  ' + code_line[80:]
                        if code_line.strip():
                            formatted_code.append(code_line)
                    else:
                        formatted_code.append(code_line)
                
                elements.append(Paragraph('<font color="#e2e8f0"><pre>' + '\n'.join(formatted_code) + '</pre></font>', styles['CustomCode']))
                elements.append(Spacer(1, 10))
        
        # Bullet points
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            elements.append(Paragraph(f"• {bullet_text}", styles['CustomBullet']))
            
        # Numbered lists
        elif line and line[0].isdigit() and '. ' in line:
            list_text = line.split('. ', 1)[1]
            number = line.split('. ')[0]
            elements.append(Paragraph(f"{number}. {list_text}", styles['CustomBullet']))
            
        # Regular paragraphs
        elif line:
            # Handle bold text
            formatted_line = line.replace('**', '<b>').replace('**', '</b>')
            # Handle italic text  
            formatted_line = formatted_line.replace('*', '<i>').replace('*', '</i>')
            # Handle inline code
            formatted_line = formatted_line.replace('`', '<font color="#4a5568" fontName="Courier">')
            formatted_line = formatted_line.replace('`', '</font>')
            
            elements.append(Paragraph(formatted_line, styles['CustomBody']))
            
        i += 1
    
    return elements

class NumberedCanvas(canvas.Canvas):
    """Custom canvas for adding headers and footers with colors."""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page_num, page in enumerate(self.pages, 1):
            self.__dict__.update(page)
            self.draw_page_elements(page_num, page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_elements(self, page_num, page_count):
        # Header with gradient-like effect
        self.setFillColor(colors.Color(0.4, 0.31, 0.64))
        self.setFont("Helvetica-Bold", 12)
        self.drawString(50, letter[1] - 30, "🤖 AI/ML/DL Comprehensive Guide")
        
        # Footer
        self.setFillColor(colors.Color(0.45, 0.55, 0.64))
        self.setFont("Helvetica", 9)
        footer_text = f"Generated: {datetime.now().strftime('%B %d, %Y')} | Page {page_num} of {page_count}"
        text_width = self.stringWidth(footer_text)
        self.drawString((letter[0] - text_width) / 2, 30, footer_text)

def generate_beautiful_pdf_reportlab(content: str, topic: str, chapter: str) -> bytes:
    """Generate a beautiful PDF using ReportLab with colors and styling."""
    try:
        buffer = BytesIO()
        
        # Create document with custom canvas
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=60,
            bottomMargin=60,
            canvasmaker=NumberedCanvas
        )
        
        # Get custom styles
        styles = create_beautiful_pdf_styles()
        elements = []
        
        # Title page elements
        elements.append(Spacer(1, 50))
        
        # Main title with color
        title_text = f"🤖 AI/ML/DL Comprehensive Guide"
        elements.append(Paragraph(title_text, styles['CustomTitle']))
        elements.append(Spacer(1, 30))
        
        # Chapter and topic info with colored boxes
        chapter_info = f"""
        <para alignment="center">
            <b>📚 Chapter:</b> {chapter}<br/>
            <b>🎯 Topic:</b> {topic}<br/>
            <b>🗓️ Generated:</b> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        </para>
        """
        elements.append(Paragraph(chapter_info, styles['CustomBody']))
        elements.append(Spacer(1, 40))
        
        # Add a colored separator
        separator_table = Table([['']], colWidths=[500])
        separator_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.4, 0.31, 0.64)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(separator_table)
        elements.append(Spacer(1, 30))
        
        # Content description box
        desc_text = f"""
        <b>📋 About This Guide:</b><br/>
        This comprehensive guide covers all aspects of <b>{topic}</b> including theoretical foundations, 
        mathematical concepts, practical implementations, and real-world applications. Generated using 
        advanced AI to provide maximum educational value with beautiful formatting and colors.
        """
        elements.append(Paragraph(desc_text, styles['CustomBody']))
        elements.append(PageBreak())
        
        # Parse and add main content
        content_elements = parse_markdown_to_reportlab(content, styles)
        elements.extend(content_elements)
        
        # Add footer section
        elements.append(Spacer(1, 30))
        footer_table = Table([['']], colWidths=[500])
        footer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.4, 0.31, 0.64)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(footer_table)
        elements.append(Spacer(1, 20))
        
        footer_text = """
        <para alignment="center">
            <b>🤖 AI/ML/DL Comprehensive Learning Hub</b><br/>
            Powered by Google Gemini<br/>
            📚 Complete coverage of 120+ AI/ML/DL topics across 11 comprehensive chapters
        </para>
        """
        elements.append(Paragraph(footer_text, styles['FooterStyle']))
        
        # Build PDF
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating PDF with ReportLab: {str(e)}")
        return None

def generate_simple_html_pdf(content: str, topic: str, chapter: str) -> str:
    """Generate HTML content that can be printed to PDF by browser."""
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    html_content = markdown2.markdown(content, extras=[
        'fenced-code-blocks', 
        'tables', 
        'header-ids'
    ])
    
    html_pdf = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{topic} - AI/ML/DL Guide</title>
        <style>
            @media print {{
                @page {{ margin: 1in; }}
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .content {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}
            .title {{
                font-size: 2.5em;
                color: #667eea;
                margin-bottom: 10px;
                font-weight: bold;
            }}
            .subtitle {{
                color: #666;
                font-size: 1.1em;
            }}
            .badges {{
                margin: 15px 0;
            }}
            .badge {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                margin: 5px;
                font-weight: bold;
            }}
            h1, h2 {{
                color: #667eea;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #764ba2;
            }}
            pre {{
                background: #2d3748;
                color: #e2e8f0;
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                font-size: 14px;
            }}
            code {{
                background: #f1f3f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }}
            blockquote {{
                border-left: 4px solid #667eea;
                background: #f8f9fa;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #eee;
            }}
            tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            <div class="header">
                <h1 class="title">🤖 AI/ML/DL Guide</h1>
                <p class="subtitle">Generated on {current_time}</p>
                <div class="badges">
                    <span class="badge">📚 {chapter}</span>
                    <span class="badge">🎯 {topic}</span>
                </div>
            </div>
            
            {html_content}
            
            <div class="footer">
                <p><strong>AI/ML/DL Comprehensive Learning Hub</strong></p>
                <p>Powered by Google Gemini | {current_time}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_pdf

def create_pdf_options(content: str, topic: str, chapter: str):
    """Create multiple PDF export options."""
    st.markdown("### 📄 PDF Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if REPORTLAB_AVAILABLE:
            if st.button("🎨 Beautiful PDF (ReportLab)", type="primary", use_container_width=True):
                with st.spinner("🎨 Creating beautiful PDF..."):
                    pdf_bytes = generate_beautiful_pdf_reportlab(content, topic, chapter)
                    
                    if pdf_bytes:
                        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        filename = f"AI_ML_Guide_{safe_topic.replace(' ', '_')}.pdf"
                        
                        b64_pdf = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">'
                        
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin: 20px 0;">
                                {href}
                                    <button style="
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        color: white; border: none; padding: 12px 24px;
                                        border-radius: 20px; font-weight: bold; cursor: pointer;
                                    ">📥 Download Beautiful PDF</button>
                                </a>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        
                        st.success("✅ Beautiful PDF with colors and styling generated!")
        else:
            st.info("💡 Install ReportLab for beautiful PDFs: `pip install reportlab`")
    
    with col2:
        if st.button("🌐 Printable HTML", type="secondary", use_container_width=True):
            html_content = generate_simple_html_pdf(content, topic, chapter)
            
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"AI_ML_Guide_{safe_topic.replace(' ', '_')}.html"
            
            b64_html = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64_html}" download="{filename}">'
            
            st.markdown(
                f"""
                <div style="text-align: center; margin: 20px 0;">
                    {href}
                        <button style="
                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white; border: none; padding: 12px 24px;
                            border-radius: 20px; font-weight: bold; cursor: pointer;
                        ">📥 Download HTML</button>
                    </a>
                </div>
                """, unsafe_allow_html=True
            )
            
            st.info("💡 Download HTML and use browser's 'Print to PDF' for colored PDF export!")

def create_pdf_styles() -> str:
    """Create beautiful CSS styles for PDF export."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #2d3748;
            margin: 0;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .content-container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            margin: 20px auto;
            max-width: 1200px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }
        
        .main-title {
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            color: #4a5568;
            font-weight: 500;
        }
        
        .chapter-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 10px 5px;
        }
        
        .topic-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 6px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
            margin: 5px;
        }
        
        h1 {
            color: #2d3748;
            font-size: 2.2em;
            font-weight: 700;
            margin: 30px 0 20px 0;
            padding: 15px 0;
            border-bottom: 2px solid #e2e8f0;
        }
        
        h2 {
            color: #4a5568;
            font-size: 1.6em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            padding: 12px 20px;
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            border-radius: 8px;
        }
        
        h3 {
            color: #2d3748;
            font-size: 1.3em;
            font-weight: 600;
            margin: 20px 0 10px 0;
            padding: 8px 15px;
            background: #f7fafc;
            border-left: 3px solid #4facfe;
            border-radius: 5px;
        }
        
        p {
            margin: 15px 0;
            text-align: justify;
            font-size: 1em;
            line-height: 1.7;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border: 1px solid #feb2b2;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #f56565;
        }
        
        .info-box {
            background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
            border: 1px solid #90cdf4;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #4299e1;
        }
        
        .success-box {
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
            border: 1px solid #9ae6b4;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #48bb78;
        }
        
        code {
            background: #2d3748;
            color: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }
        
        pre {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: #e2e8f0;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.3);
        }
        
        pre code {
            background: transparent;
            padding: 0;
            color: inherit;
        }
        
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 8px 0;
            line-height: 1.6;
        }
        
        blockquote {
            border-left: 4px solid #667eea;
            background: #f7fafc;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: #4a5568;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tr:hover {
            background: #e2e8f0;
        }
        
        .emoji {
            font-size: 1.2em;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            font-size: 0.9em;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        @media print {
            .content-container {
                box-shadow: none;
                margin: 0;
                padding: 20px;
            }
        }
    </style>
    """

def create_pdf_content(topic: str, chapter: str, content: str) -> str:
    """Create beautifully formatted PDF content."""
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(content, extras=[
        'fenced-code-blocks', 
        'tables', 
        'header-ids',
        'footnotes',
        'task_list',
        'strike',
        'cuddled-lists'
    ])
    
    pdf_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{topic} - AI/ML/DL Guide</title>
        {create_pdf_styles()}
    </head>
    <body>
        <div class="content-container">
            <div class="header">
                <h1 class="main-title">🤖 AI/ML/DL Comprehensive Guide</h1>
                <p class="subtitle">Generated on {current_time}</p>
                <div>
                    <span class="chapter-badge">📚 {chapter}</span>
                    <span class="topic-badge">🎯 {topic}</span>
                </div>
            </div>
            
            <div class="highlight-box">
                <strong>📋 About This Guide:</strong><br>
                This comprehensive guide covers all aspects of <strong>{topic}</strong> including theoretical foundations, 
                mathematical concepts, practical implementations, and real-world applications. Generated using advanced AI 
                to provide maximum educational value.
            </div>
            
            {html_content}
            
            <div class="footer">
                <p><strong>🤖 AI/ML/DL Comprehensive Learning Hub</strong></p>
                <p>Powered by Google Gemini | Generated: {current_time}</p>
                <p>📚 Complete coverage of 120+ AI/ML/DL topics across 11 comprehensive chapters</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return pdf_html

def generate_pdf(content: str, topic: str, chapter: str) -> bytes:
    """Generate a beautiful PDF from content."""
    try:
        pdf_html = create_pdf_content(topic, chapter, content)
        pdf_bytes = weasyprint.HTML(string=pdf_html).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def create_download_button(pdf_bytes: bytes, filename: str):
    """Create a download button for the PDF."""
    if pdf_bytes:
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" target="_blank">'
        
        st.markdown(
            f"""
            <div style="text-align: center; margin: 30px 0;">
                {href}
                    <button style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        border-radius: 25px;
                        font-size: 16px;
                        font-weight: 600;
                        cursor: pointer;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                        transition: all 0.3s ease;
                    ">
                        📄 Download Beautiful PDF Guide
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Additional info
        st.info(f"""
        **📋 PDF Features:**
        - 🎨 Beautiful gradient styling and colors
        - 📚 Professional layout with proper typography  
        - 🖼️ Formatted code blocks and tables
        - 📄 Complete content with headers and navigation
        - 🗓️ Generated on {datetime.now().strftime("%B %d, %Y")}
        """)

def create_comprehensive_prompt(topic: str) -> str:
    """Create a comprehensive prompt for the Gemini model."""
    return f"""
    You are an expert AI/ML/DL educator. Provide a COMPREHENSIVE and DETAILED explanation for the topic: "{topic}"

    Please structure your response with the following sections:

    ## 1. 🎯 Concept Overview
    - Clear definition and explanation
    - Key principles and theory
    - Historical context and evolution
    - Why this topic is important in AI/ML

    ## 2. 📊 Mathematical Foundation
    - Core mathematical concepts
    - Key equations and formulas
    - Step-by-step mathematical derivations
    - Probability/statistical foundations if applicable

    ## 3. 🔧 Algorithm Details
    - Detailed algorithm steps
    - Pseudocode implementation
    - Complexity analysis (time/space)
    - Variations and improvements

    ## 4. 💻 Complete Code Implementation
    - Full Python implementation from scratch
    - Using popular libraries (scikit-learn, TensorFlow, PyTorch)
    - Multiple examples with different datasets
    - Code comments explaining each step

    ## 5. 📈 Visual Examples & Demonstrations
    - Describe what visualizations would be helpful
    - Data flow diagrams
    - Algorithm workflow charts
    - Performance comparison charts

    ## 6. 🎯 Real-World Applications
    - Industry use cases
    - Success stories
    - Current research applications
    - Future prospects

    ## 7. ⚖️ Advantages & Limitations
    - When to use this method
    - When NOT to use it
    - Comparison with alternatives
    - Best practices and pitfalls

    ## 8. 🔬 Advanced Topics & Research
    - Recent developments
    - State-of-the-art variations
    - Open research problems
    - Future directions

    ## 9. 📚 Hands-on Exercises
    - Practice problems
    - Project ideas
    - Dataset recommendations
    - Step-by-step tutorials

    ## 10. 🔗 Related Topics & Extensions
    - Connected concepts
    - Prerequisite knowledge
    - Next steps for learning
    - Advanced variations

    Make your explanation EXTREMELY detailed, educational, and practical. Include as much technical depth as possible while keeping it accessible. Provide complete, runnable code examples and explain every concept thoroughly.
    """

def initialize_gemini(api_key: str):
    """Initialize Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return None

def generate_content(model, prompt: str) -> str:
    """Generate content using Gemini model."""
    try:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            top_k=40
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

def create_sample_visualization(topic: str):
    """Create sample visualizations based on the topic."""
    if "regression" in topic.lower():
        # Sample regression visualization
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 1, 100)
        
        fig = px.scatter(x=x, y=y, title=f"{topic} - Sample Data Visualization")
        fig.add_scatter(x=x, y=2*x+1, mode='lines', name='True Regression Line')
        return fig
        
    elif "cluster" in topic.lower():
        # Sample clustering visualization
        np.random.seed(42)
        data1 = np.random.normal([2, 2], 0.5, (50, 2))
        data2 = np.random.normal([6, 6], 0.5, (50, 2))
        data3 = np.random.normal([2, 6], 0.5, (50, 2))
        
        data = np.vstack([data1, data2, data3])
        labels = ['Cluster 1'] * 50 + ['Cluster 2'] * 50 + ['Cluster 3'] * 50
        
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels, 
                        title=f"{topic} - Sample Clustering Visualization")
        return fig
        
    elif "neural" in topic.lower() or "deep" in topic.lower():
        # Sample neural network architecture visualization
        layers = ['Input (784)', 'Hidden 1 (128)', 'Hidden 2 (64)', 'Output (10)']
        x_pos = [0, 1, 2, 3]
        y_pos = [0, 0, 0, 0]
        
        fig = go.Figure()
        for i, (layer, x) in enumerate(zip(layers, x_pos)):
            fig.add_trace(go.Scatter(
                x=[x], y=[0], mode='markers+text',
                marker=dict(size=60, color=px.colors.qualitative.Set1[i]),
                text=[layer], textposition="middle center",
                name=layer, showlegend=False
            ))
        
        # Add connections
        for i in range(len(x_pos)-1):
            fig.add_shape(
                type="line", x0=x_pos[i], y0=0, x1=x_pos[i+1], y1=0,
                line=dict(color="gray", width=2)
            )
        
        fig.update_layout(
            title=f"{topic} - Neural Network Architecture",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=300
        )
        return fig
    
    else:
        # Generic visualization
        x = np.arange(10)
        y = np.random.rand(10)
        fig = px.bar(x=x, y=y, title=f"{topic} - Conceptual Visualization")
        return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI/ML/DL Comprehensive Learning Hub</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for API key and navigation
    st.sidebar.markdown("## 🔧 Configuration")
    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key:",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    if not api_key:
        st.sidebar.warning("Please enter your Gemini API key to proceed.")
        st.info("""
        ## Welcome to the AI/ML/DL Comprehensive Learning Hub! 🚀
        
        This application provides detailed explanations for over 120 AI/ML/DL topics using Google's Gemini AI.
        
        ### Features:
        - 📚 **Comprehensive Coverage**: 11 major chapters covering all aspects of AI/ML/DL
        - 🔍 **Detailed Explanations**: Mathematical foundations, algorithms, code implementations
        - 📊 **Visual Learning**: Interactive charts and diagrams
        - 💻 **Practical Code**: Complete implementations and examples
        - 🎯 **Real-world Applications**: Industry use cases and best practices
        - 📄 **Beautiful PDFs**: Export guides as styled PDFs with colors and formatting
        
        ### To Get Started:
        1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Select a chapter and topic to explore
        4. Export as beautiful PDF (requires `pip install reportlab markdown2`)
        
        **PDF Options Available:**
        - 🎨 **Beautiful PDF**: Full-color styled PDF with ReportLab
        - 🌐 **Printable HTML**: Browser-printable HTML with styling
        
        **Note**: The app uses up to 8192 tokens for maximum comprehensive content generation.
        """)
        return
    
    # Initialize Gemini
    model = initialize_gemini(api_key)
    if not model:
        return
    
    st.sidebar.success("✅ Gemini API initialized successfully!")
    
    # Chapter selection
    st.sidebar.markdown("## 📖 Select Learning Content")
    selected_chapter = st.sidebar.selectbox(
        "Choose a Chapter:",
        list(CHAPTERS.keys()),
        help="Select the chapter you want to explore"
    )
    
    # Topic selection
    if selected_chapter:
        selected_topic = st.sidebar.selectbox(
            "Choose a Topic:",
            CHAPTERS[selected_chapter],
            help="Select the specific topic to learn about"
        )
        
        # Learning preferences
        st.sidebar.markdown("## ⚙️ Learning Preferences")
        include_advanced = st.sidebar.checkbox(
            "Include Advanced Topics", 
            value=True,
            help="Include cutting-edge research and advanced concepts"
        )
        
        include_code = st.sidebar.checkbox(
            "Include Code Examples", 
            value=True,
            help="Generate complete code implementations"
        )
        
        include_math = st.sidebar.checkbox(
            "Include Mathematical Details", 
            value=True,
            help="Include mathematical derivations and formulas"
        )
        
        # Generate content button
        if st.sidebar.button("🚀 Generate Comprehensive Guide", type="primary"):
            # Display selected topic
            st.markdown(f"""
            <div class="chapter-container">
                <h2>📚 {selected_chapter}</h2>
                <h3>🎯 Topic: {selected_topic}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Generate and display content
                with st.spinner("🤖 Generating comprehensive content... This may take a moment."):
                    prompt = create_comprehensive_prompt(selected_topic)
                    content = generate_content(model, prompt)
                    
                st.markdown("## 📖 Comprehensive Guide")
                st.markdown(content)
                
                # PDF Export Section
                st.markdown("---")
                create_pdf_options(content, selected_topic, selected_chapter)
            
            with col2:
                # Sample visualization
                st.markdown("## 📊 Sample Visualization")
                try:
                    fig = create_sample_visualization(selected_topic)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Visualization will be available based on topic content")
                
                # Topic statistics
                st.markdown("## 📈 Topic Statistics")
                stats_data = {
                    'Metric': ['Content Tokens', 'Sections', 'Complexity Level', 'Estimated Reading Time'],
                    'Value': ['8192 max', '10+', 'Advanced', '15-20 min']
                }
                st.table(pd.DataFrame(stats_data))
                
                # Related topics
                st.markdown("## 🔗 Related Topics")
                current_topics = CHAPTERS[selected_chapter]
                current_index = current_topics.index(selected_topic)
                
                if current_index > 0:
                    st.info(f"**Previous**: {current_topics[current_index-1]}")
                if current_index < len(current_topics) - 1:
                    st.info(f"**Next**: {current_topics[current_index+1]}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 AI/ML/DL Comprehensive Learning Hub | Powered by Google Gemini</p>
        <p>📚 Covering 120+ topics across 11 comprehensive chapters</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
