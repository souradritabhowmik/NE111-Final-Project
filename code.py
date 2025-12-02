import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Set up the page with pink theme
st.set_page_config(
    page_title="Distribution Fitting Studio",
    page_icon="📊",
    layout="wide"
)

# Custom pink theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #E91E63;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(233, 30, 99, 0.2);
    }
    .section-header {
        font-size: 1.8rem;
        color: #C2185B;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #F48FB1;
        padding-left: 15px;
        font-weight: 600;
    }
    .success-box {
        background-color: #FCE4EC;
        border: 2px solid #F8BBD0;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        color: #880E4F;
    }
    .metric-card {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(233, 30, 99, 0.1);
    }
    .stButton button {
        background-color: #E91E63;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #C2185B;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(233, 30, 99, 0.3);
    }
    .stSelectbox, .stMultiselect {
        border: 2px solid #F48FB1;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FCE4EC;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E91E63 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">🌸 Distribution Fitting Studio</div>', unsafe_allow_html=True)
st.markdown("Upload your dataset or enter values manually to explore statistical distributions with interactive fitting tools.")

# Define available distributions
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

def fit_distribution_to_data(data, distribution_name):
    """Fit a statistical distribution to the provided dataset"""
    try:
        distribution_info = DISTRIBUTIONS[distribution_name]
        distribution_function = distribution_info["func"]
        
        fitted_params = distribution_function.fit(data)
        fitted_distribution = distribution_function(*fitted_params)
        
        # Create histogram for comparison
        histogram, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf_values = fitted_distribution.pdf(bin_centers)
        
        # Calculate KL divergence (requires positive values)
        mask = (pdf_values > 0) & (histogram > 0)
        if np.sum(mask) > 0:
            kl_div = stats.entropy(histogram[mask], pdf_values[mask])
        else:
            kl_div = float('inf')
        
        # Additional fit metrics
        mean_squared_error = np.mean((histogram - pdf_values) ** 2)
        maximum_error = np.max(np.abs(histogram - pdf_values))
        
        return {
            "parameters": fitted_params,
            "distribution": fitted_distribution,
            "mse": mean_squared_error,
            "max_error": maximum_error,
            "kl_divergence": kl_div
        }
    except Exception as error:
        st.error(f"Unable to fit {distribution_name} distribution: {str(error)}")
        return None

def create_manual_fit(data, distribution_name, manual_parameters):
    """Generate distribution with user-specified parameters"""
    try:
        distribution_info = DISTRIBUTIONS[distribution_name]
        distribution_function = distribution_info["func"]
        fitted_distribution = distribution_function(*manual_parameters)
        
        histogram, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf_values = fitted_distribution.pdf(bin_centers)
        
        mask = (pdf_values > 0) & (histogram > 0)
        if np.sum(mask) > 0:
            kl_div = stats.entropy(histogram[mask], pdf_values[mask])
        else:
            kl_div = float('inf')
        
        mean_squared_error = np.mean((histogram - pdf_values) ** 2)
        maximum_error = np.max(np.abs(histogram - pdf_values))
        
        return {
            "parameters": manual_parameters,
            "distribution": fitted_distribution,
            "mse": mean_squared_error,
            "max_error": maximum_error,
            "kl_divergence": kl_div
        }
    except Exception as error:
        st.error(f"Manual fit error: {str(error)}")
        return None

def create_comparison_plot(data, fitting_results, plot_title):
    """Visualize data histogram with fitted distributions"""
    histogram, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    chart_data = pd.DataFrame({
        'Data Value': bin_centers,
        'Data Histogram': histogram
    })
    
    x_range = np.linspace(np.min(data), np.max(data), 500)
    for dist_name, result in fitting_results.items():
        pdf_curve = result['distribution'].pdf(x_range)
        chart_data[f'{dist_name} Fit'] = np.interp(bin_centers, x_range, pdf_curve)
    
    return chart_data

def create_single_fit_plot(data, result, distribution_name):
    """Visualize single distribution fit"""
    histogram, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x_range = np.linspace(np.min(data), np.max(data), 500)
    pdf_curve = result['distribution'].pdf(x_range)
    
    chart_data = pd.DataFrame({
        'Data Value': bin_centers,
        'Data Histogram': histogram,
        f'{distribution_name} Fit': np.interp(bin_centers, x_range, pdf_curve)
    })
    
    return chart_data

# Main app tabs
tab_input, tab_auto_fit, tab_manual_fit = st.tabs(["📥 Data Input", "🤖 Auto Fitting", "🎛️ Manual Fitting"])

with tab_input:
    st.markdown('<div class="section-header">Data Input Options</div>', unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns([1, 1])
    
    with input_col1:
        st.subheader("Manual Data Entry")
        sample_data = "3.2, 4.1, 3.8, 4.5, 3.9, 4.2, 3.7, 4.0, 3.5, 4.3, 3.6, 4.4"
        user_input = st.text_area(
            "Enter numerical values (comma or space separated):",
            value=sample_data,
            height=120,
            help="Example: 1.2, 3.4, 5.6 or 1.2 3.4 5.6"
        )
        
        if user_input:
            try:
                cleaned_input = user_input.replace(',', ' ').split()
                parsed_data = np.array([float(val) for val in cleaned_input])
                st.success(f"✅ Successfully parsed {len(parsed_data)} data points")
                st.session_state['dataset'] = parsed_data
            except ValueError:
                st.error("⚠️ Please enter valid numbers only")
    
    with input_col2:
        st.subheader("CSV File Upload")
        uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'], 
                                       help="Upload a CSV file with your data")
        
        if uploaded_csv is not None:
            try:
                data_frame = pd.read_csv(uploaded_csv)
                st.write("**Data Preview:**")
                st.dataframe(data_frame.head(), use_container_width=True)
                
                if len(data_frame.columns) > 0:
                    column_selector = st.selectbox(
                        "Select column for analysis:",
                        data_frame.columns,
                        key="col_select"
                    )
                    
                    if column_selector:
                        column_data = data_frame[column_selector].dropna().values
                        st.session_state['dataset'] = column_data
                        st.success(f"✅ Loaded {len(column_data)} values from '{column_selector}'")
                else:
                    st.warning("No columns found in the uploaded file")
                    
            except Exception as upload_error:
                st.error(f"File reading error: {str(upload_error)}")

# Check if data is available
if 'dataset' not in st.session_state:
    st.info("👈 Please enter or upload data in the 'Data Input' tab")
    st.stop()

current_data = st.session_state['dataset']

# Data summary section
with st.expander("📊 Data Summary Statistics", expanded=True):
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Points", len(current_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean", f"{np.mean(current_data):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Std Dev", f"{np.std(current_data):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Range", f"{np.min(current_data):.3f} - {np.max(current_data):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_auto_fit:
    st.markdown('<div class="section-header">Automatic Distribution Fitting</div>', unsafe_allow_html=True)
    
    available_distributions = list(DISTRIBUTIONS.keys())
    default_selections = ["Normal", "Gamma", "Weibull"]
    
    selected_distributions = st.multiselect(
        "Choose distributions to fit automatically:",
        available_distributions,
        default=default_selections,
        help="Select multiple distributions to compare their fits"
    )
    
    if st.button("🚀 Run Automatic Fitting", type="primary") and selected_distributions:
        fitting_results = {}
        
        progress_indicator = st.progress(0)
        status_text = st.empty()
        
        for idx, dist_name in enumerate(selected_distributions):
            progress = (idx + 1) / len(selected_distributions)
            progress_indicator.progress(progress)
            status_text.text(f"Fitting {dist_name}...")
            
            fit_result = fit_distribution_to_data(current_data, dist_name)
            if fit_result:
                fitting_results[dist_name] = fit_result
        
        progress_indicator.empty()
        status_text.empty()
        
        if fitting_results:
            st.session_state['auto_fit_results'] = fitting_results
            
            st.markdown('<div class="success-box">✨ All distributions fitted successfully!</div>', unsafe_allow_html=True)
            
            results_col1, results_col2 = st.columns([1, 1])
            
            with results_col1:
                st.subheader("📈 Fit Visualization")
                
                if fitting_results:
                    plot_data = create_comparison_plot(current_data, fitting_results, 'Distribution Fits')
                    st.line_chart(plot_data.set_index('Data Value'), use_container_width=True)
                    st.caption("Histogram shows your data; colored lines show fitted distributions")
            
            with results_col2:
                st.subheader("📋 Fit Quality Comparison")
                
                metrics_table = []
                for dist_name, result in fitting_results.items():
                    metrics_table.append({
                        'Distribution': dist_name,
                        'MSE': f"{result['mse']:.6f}",
                        'Max Error': f"{result['max_error']:.4f}",
                        'KL Divergence': f"{result['kl_divergence']:.4f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_table)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                if metrics_table:
                    best_fitting = min(metrics_table, key=lambda x: float(x['MSE']))
                    st.info(f"🏆 **Best Fit**: {best_fitting['Distribution']} (MSE: {best_fitting['MSE']})")
                
                st.subheader("⚙️ Fitted Parameters")
                for dist_name, result in fitting_results.items():
                    with st.expander(f"{dist_name} Parameters"):
                        param_labels = DISTRIBUTIONS[dist_name]["params"]
                        param_values = result['parameters']
                        
                        for i, (label, value) in enumerate(zip(param_labels, param_values)):
                            st.write(f"**{label}**: `{value:.6f}`")

with tab_manual_fit:
    st.markdown('<div class="section-header">Manual Parameter Exploration</div>', unsafe_allow_html=True)
    
    manual_distribution = st.selectbox(
        "Select a distribution to tune manually:",
        list(DISTRIBUTIONS.keys()),
        index=0,
        key="manual_dist_select"
    )
    
    if manual_distribution:
        distribution_details = DISTRIBUTIONS[manual_distribution]
        parameter_names = distribution_details["params"]
        
        st.write("Adjust the distribution parameters using the sliders below:")
        
        user_parameters = []
        parameter_cols = st.columns(len(parameter_names))
        
        data_mean = np.mean(current_data)
        data_std = np.std(current_data)
        data_min = np.min(current_data)
        data_max = np.max(current_data)
        
        for i, param_name in enumerate(parameter_names):
            with parameter_cols[i]:
                # Set appropriate slider ranges based on parameter type
                if 'scale' in param_name.lower() or 'std' in param_name.lower():
                    default_val = max(0.1, float(data_std))
                    param_value = st.slider(
                        param_name, 
                        0.1, 
                        3.0 * data_std, 
                        default_val, 
                        0.1,
                        help="Scale/standard deviation parameter"
                    )
                elif 'mean' in param_name.lower() or 'loc' in param_name.lower():
                    param_value = st.slider(
                        param_name,
                        float(data_min),
                        float(data_max),
                        float(data_mean),
                        0.1,
                        help="Location/mean parameter"
                    )
                elif 'shape' in param_name.lower() or 'α' in param_name or 'k' in param_name:
                    param_value = st.slider(
                        param_name,
                        0.1,
                        15.0,
                        2.0,
                        0.1,
                        help="Shape parameter"
                    )
                elif 'λ' in param_name or 'mu' in param_name.lower():
                    param_value = st.slider(
                        param_name,
                        0.1,
                        10.0,
                        1.0,
                        0.1,
                        help="Rate/mean parameter"
                    )
                else:
                    param_value = st.slider(
                        param_name,
                        -5.0,
                        5.0,
                        0.0,
                        0.1
                    )
                
                user_parameters.append(param_value)
        
        # Ensure all required parameters are present
        if manual_distribution in ["Gamma", "Weibull"] and len(user_parameters) < 3:
            user_parameters.append(0.0)
        
        if st.button("📐 Evaluate Manual Fit", type="primary"):
            manual_result = create_manual_fit(current_data, manual_distribution, user_parameters)
            
            if manual_result:
                manual_col1, manual_col2 = st.columns([1, 1])
                
                with manual_col1:
                    st.subheader("🎨 Fit Visualization")
                    
                    plot_data = create_single_fit_plot(current_data, manual_result, manual_distribution)
                    st.line_chart(plot_data.set_index('Data Value'), use_container_width=True)
                    st.caption(f"Manual {manual_distribution} distribution fit")
                
                with manual_col2:
                    st.subheader("📊 Fit Assessment")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("MSE", f"{manual_result['mse']:.6f}")
                    with metric_col2:
                        st.metric("Max Error", f"{manual_result['max_error']:.4f}")
                    with metric_col3:
                        st.metric("KL Divergence", f"{manual_result['kl_divergence']:.4f}")
                    
                    st.subheader("🔧 Current Parameters")
                    for i, (name, value) in enumerate(zip(parameter_names, user_parameters)):
                        st.code(f"{name}: {value:.6f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #E91E63; padding: 20px;'>
        <p>🌸 <strong>Distribution Fitting Studio</strong> • Built with Streamlit</p>
        <p style='font-size: 0.9rem; color: #F48FB1;'>Explore, fit, and visualize statistical distributions</p>
    </div>
    """,
    unsafe_allow_html=True
)
