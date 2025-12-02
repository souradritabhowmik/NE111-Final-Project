import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Graph Fitting",
    page_icon=".",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Distribution Fitting Tool</div>', unsafe_allow_html=True)
st.write("Upload your data or enter it manually to fit various statistical distributions and visualize the results.")

DISTRIBUTIONS = {
    "Normal": {"func": stats.norm, "params": ["μ (mean)", "σ (std)"]},
    "Gamma": {"func": stats.gamma, "params": ["α (shape)", "β (scale)", "loc"]},
    "Weibull": {"func": stats.weibull_min, "params": ["λ (scale)", "k (shape)", "loc"]},
    "Exponential": {"func": stats.expon, "params": ["λ (scale)", "loc"]},
    "Log-Normal": {"func": stats.lognorm, "params": ["σ", "μ", "scale"]},
    "Beta": {"func": stats.beta, "params": ["α", "β", "loc", "scale"]},
    "Chi-squared": {"func": stats.chi2, "params": ["k (df)"]},
    "Student's t": {"func": stats.t, "params": ["ν (df)", "loc", "scale"]},
    "Uniform": {"func": stats.uniform, "params": ["loc", "scale"]},
    "Poisson": {"func": stats.poisson, "params": ["μ (lambda)"]},
    "Rayleigh": {"func": stats.rayleigh, "params": ["σ (scale)", "loc"]},
    "Cauchy": {"func": stats.cauchy, "params": ["loc", "scale"]}
}

def fit_distribution(data, dist_name):
    """Fit a distribution to data and return parameters and fit metrics"""
    try:
        dist_info = DISTRIBUTIONS[dist_name]
        dist_func = dist_info["func"]
        
        params = dist_func.fit(data)
        
        fitted_dist = dist_func(*params)
        
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf_values = fitted_dist.pdf(bin_centers)
        
        mask = (pdf_values > 0) & (hist > 0)
        if np.sum(mask) > 0:
            kl_divergence = stats.entropy(hist[mask], pdf_values[mask])
        else:
            kl_divergence = float('inf')
            
        mse = np.mean((hist - pdf_values) ** 2)
        
        max_error = np.max(np.abs(hist - pdf_values))
        
        return {
            "params": params,
            "fitted_dist": fitted_dist,
            "mse": mse,
            "max_error": max_error,
            "kl_divergence": kl_divergence
        }
    except Exception as e:
        st.error(f"Error fitting {dist_name} distribution: {str(e)}")
        return None

def manual_fit_distribution(data, dist_name, manual_params):
    """Create distribution with manually specified parameters"""
    try:
        dist_info = DISTRIBUTIONS[dist_name]
        dist_func = dist_info["func"]
        fitted_dist = dist_func(*manual_params)
        
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf_values = fitted_dist.pdf(bin_centers)
        
        mask = (pdf_values > 0) & (hist > 0)
        if np.sum(mask) > 0:
            kl_divergence = stats.entropy(hist[mask], pdf_values[mask])
        else:
            kl_divergence = float('inf')
            
        mse = np.mean((hist - pdf_values) ** 2)
        max_error = np.max(np.abs(hist - pdf_values))
        
        return {
            "params": manual_params,
            "fitted_dist": fitted_dist,
            "mse": mse,
            "max_error": max_error,
            "kl_divergence": kl_divergence
        }
    except Exception as e:
        st.error(f"Error with manual fit: {str(e)}")
        return None

def create_streamlit_plot(data, results, title):
    """Create plot using Streamlit's native charting"""
    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    chart_data = pd.DataFrame({
        'value': bin_centers,
        'data_histogram': hist
    })
    
    x = np.linspace(np.min(data), np.max(data), 1000)
    for dist_name, result in results.items():
        pdf_values = result['fitted_dist'].pdf(x)
        # Interpolate to match histogram bins
        chart_data[f'{dist_name}_fit'] = np.interp(bin_centers, x, pdf_values)
    
    st.line_chart(
        chart_data.set_index('value'),
        use_container_width=True
    )

def create_manual_fit_plot(data, result, dist_name):
    """Create manual fit plot using Streamlit native charts"""
    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x = np.linspace(np.min(data), np.max(data), 1000)
    pdf_values = result['fitted_dist'].pdf(x)
    
    chart_data = pd.DataFrame({
        'value': bin_centers,
        'data_histogram': hist,
        f'{dist_name}_fit': np.interp(bin_centers, x, pdf_values)
    })
    
    st.line_chart(
        chart_data.set_index('value'),
        use_container_width=True
    )

tab1, tab2, tab3 = st.tabs(["Input", "Auto Fitting", "Manual Fitting"])

with tab1:
    st.markdown('<div class="section-header">Data Input Methods</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Manual Entry")
        manual_data = st.text_area(
            "Enter data (comma or space separated):",
            value="2.3, 3.1, 4.5, 2.8, 3.9, 4.2, 3.7, 2.9, 3.4, 4.1, 3.2, 4.7",
            height=100
        )
        
        if manual_data:
            try:
                data_str = manual_data.replace(',', ' ').split()
                data = np.array([float(x) for x in data_str])
                st.success(f"✅ Successfully loaded {len(data)} data points")
                st.session_state.data = data
            except ValueError:
                st.error("❌ Please enter valid numerical data")
    
    with col2:
        st.subheader("CSV File Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if len(df.columns) > 0:
                    selected_column = st.selectbox("Select column for analysis:", df.columns)
                    data = df[selected_column].dropna().values
                    st.session_state.data = data
                    st.success(f"✅ Successfully loaded {len(data)} data points from column '{selected_column}'")
                else:
                    st.error("❌ No valid columns found in the CSV file")
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")

if 'data' not in st.session_state:
    st.info("Please enter or upload data in the 'Data Input' tab to get started")
    st.stop()

data = st.session_state.data

with st.expander("📋 Data Summary"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        st.metric("Mean", f"{np.mean(data):.4f}")
    with col3:
        st.metric("Standard Deviation", f"{np.std(data):.4f}")
    with col4:
        st.metric("Range", f"{np.min(data):.4f} - {np.max(data):.4f}")

with tab2:
    st.markdown('<div class="section-header">Automatic Distribution Fitting</div>', unsafe_allow_html=True)
    
    selected_dists = st.multiselect(
        "Select distributions to fit:",
        list(DISTRIBUTIONS.keys()),
        default=["Normal", "Gamma", "Weibull"]
    )
    
    if st.button(" Fit Selected Distributions") and selected_dists:
        results = {}
        
        progress_bar = st.progress(0)
        for i, dist_name in enumerate(selected_dists):
            progress_bar.progress((i + 1) / len(selected_dists), text=f"Fitting {dist_name}...")
            result = fit_distribution(data, dist_name)
            if result:
                results[dist_name] = result
        
        st.session_state.fitting_results = results
        
        if results:
            st.markdown('<div class="success-box">✅ All distributions fitted successfully!</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Visualization")
                create_streamlit_plot(data, results, 'Distribution Fitting Results')
                st.caption("Note: The histogram shows your data, and the lines show fitted distributions")
            
            with col2:
                st.subheader("📊 Fit Quality Metrics")
                
                metrics_data = []
                for dist_name, result in results.items():
                    metrics_data.append({
                        'Distribution': dist_name,
                        'MSE': f"{result['mse']:.6f}",
                        'Max Error': f"{result['max_error']:.4f}",
                        'KL Divergence': f"{result['kl_divergence']:.4f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                if metrics_data:
                    best_fit = min(metrics_data, key=lambda x: float(x['MSE']))
                    st.info(f" **Best Fit**: {best_fit['Distribution']} (MSE: {best_fit['MSE']})")
                
                st.subheader("🔢 Fitted Parameters")
                for dist_name, result in results.items():
                    with st.expander(f"{dist_name} Parameters"):
                        param_names = DISTRIBUTIONS[dist_name]["params"]
                        params = result['params']
                        
                        for i, (name, value) in enumerate(zip(param_names, params)):
                            st.write(f"{name}: {value:.4f}")
                        
                        if len(params) > len(param_names):
                            for i in range(len(param_names), len(params)):
                                st.write(f"Parameter {i+1}: {params[i]:.4f}")

with tab3:
    st.markdown('<div class="section-header">Manual Distribution Fitting</div>', unsafe_allow_html=True)
    
    manual_dist = st.selectbox("Select distribution for manual fitting:", list(DISTRIBUTIONS.keys()))
    
    if manual_dist:
        dist_info = DISTRIBUTIONS[manual_dist]
        param_names = dist_info["params"]
        
        st.write("Adjust the parameters using the sliders below:")
        
        manual_params = []
        cols = st.columns(len(param_names))
        
        for i, param_name in enumerate(param_names):
            with cols[i]:
                data_mean = np.mean(data)
                data_std = np.std(data)
                
                if 'scale' in param_name.lower() or 'std' in param_name.lower():
                    value = st.slider(param_name, 0.1, 2.0 * data_std, float(data_std), 0.1)
                elif 'mean' in param_name.lower() or 'loc' in param_name.lower():
                    value = st.slider(param_name, float(np.min(data)), float(np.max(data)), float(data_mean), 0.1)
                elif 'shape' in param_name.lower() or 'α' in param_name or 'k' in param_name:
                    value = st.slider(param_name, 0.1, 10.0, 2.0, 0.1)
                else:
                    value = st.slider(param_name, -10.0, 10.0, 0.0, 0.1)
                
                manual_params.append(value)
        
        if manual_dist == "Gamma" and len(manual_params) < 3:
            manual_params.extend([0.0])  # Add loc parameter
        elif manual_dist == "Weibull" and len(manual_params) < 3:
            manual_params.extend([0.0])  # Add loc parameter
        
        if st.button("Evaluate Manual Fit"):
            result = manual_fit_distribution(data, manual_dist, manual_params)
            
            if result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(" Manual Fit Visualization")
                    create_manual_fit_plot(data, result, manual_dist)
                    st.caption(f"Manual {manual_dist} distribution fit to your data")
                
                with col2:
                    st.subheader("📈 Fit Quality")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Mean Squared Error", f"{result['mse']:.6f}")
                    with metrics_col2:
                        st.metric("Maximum Error", f"{result['max_error']:.4f}")
                    with metrics_col3:
                        st.metric("KL Divergence", f"{result['kl_divergence']:.4f}")
                    
                    st.subheader(" Parameters")
                    for i, (name, value) in enumerate(zip(param_names, manual_params)):
                        st.write(f"{name}: {value:.4f}")

st.markdown("---")
st.markdown()
