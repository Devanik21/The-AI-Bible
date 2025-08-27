import streamlit as st
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List
import time

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
        
        ### To Get Started:
        1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Select a chapter and topic to explore
        
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
