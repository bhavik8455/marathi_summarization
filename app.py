import streamlit as st
from datasets import load_dataset

from src.enhanced_abstractive import generate_summary_with_options
from src.preprocess import split_sentences
from src.evaluation import overlap_precision_recall_f1

st.set_page_config(page_title="Marathi Abstractive Summarizer", page_icon="📝", layout="wide")

# Subtle custom styling for a more attractive UI
st.markdown(
    """
    <style>
      .app-title {font-size: 34px; font-weight: 800; margin-bottom: 0.25rem;}
      .app-sub {color: #6b7280; margin-bottom: 1rem;}
      .card {background: #ffffff; padding: 1rem 1.25rem; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.03);} 
      .muted {color: #6b7280;}
      .section-title {margin: 0.25rem 0 0.75rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">📝 Marathi Abstractive Summarization</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Rule-based prototype with synonym-based paraphrasing & abstractive generation</div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader("Input Text")
    user_text = st.text_area("Enter Marathi text", value="", height=260, placeholder="Paste Marathi text here…")

with right:
    st.subheader("Summary Options")
    max_sentences = st.slider(
        "Max Summary Sentences",
        min_value=1,
        max_value=6,
        value=3,
        help="Maximum number of sentences in the summary (enhanced method only)"
    )

summary = ""
if user_text and user_text.strip():
    # Always use the enhanced summarization method
    summary = generate_summary_with_options(user_text.strip(), method="enhanced", max_sentences=max_sentences)

st.subheader("Summary")
st.write(summary or "(Type text above to see the summary)")

st.divider()

# Evaluation Section
st.subheader("Model Evaluation")

if st.button("🚀 Evaluate Model", type="primary"):
    with st.spinner("Loading dataset and evaluating model..."):
        try:
            # Load the Marathi summarization dataset
            dataset = load_dataset("Existance/Marathi_summarization")
            
            # Handle different dataset structures
            if isinstance(dataset, dict):
                # If dataset is a dict, get the first available split
                available_splits = list(dataset.keys())
                if 'test' in available_splits:
                    test_data = dataset['test']
                elif 'train' in available_splits:
                    test_data = dataset['train']
                else:
                    # Get the first available split
                    first_split = available_splits[0]
                    test_data = dataset[first_split]
            else:
                # If dataset is already a Dataset object
                test_data = dataset
            
            # Convert to list if needed and take ALL available samples
            if hasattr(test_data, 'select'):
                sample_size = len(test_data)
                test_subset = test_data.select(range(sample_size))
            else:
                test_data_list = list(test_data) if not isinstance(test_data, list) else test_data
                sample_size = len(test_data_list)
                test_subset = test_data_list[:sample_size]
            
            # Initialize metrics (simple/original)
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            valid_samples = 0
            
            # Determine column names dynamically
            text_column = None
            summary_column = None
            
            if hasattr(test_data, 'column_names'):
                columns = test_data.column_names
                # Try different possible column names
                for col in columns:
                    if col.lower() in ['text', 'article', 'document', 'content', 'input']:
                        text_column = col
                    elif col.lower() in ['summary', 'target', 'reference', 'output']:
                        summary_column = col
                
                # If not found, use the first two columns
                if not text_column and len(columns) >= 2:
                    text_column = columns[0]
                    summary_column = columns[1]
                elif not text_column and len(columns) >= 1:
                    text_column = columns[0]
            
            
            # Evaluate on each sample
            progress_bar = st.progress(0)
            for i, sample in enumerate(test_subset):
                # Handle different ways to access sample data
                if hasattr(sample, 'keys'):
                    # It's a dict-like object
                    reference_summary = sample.get(summary_column, "")
                    source_text = sample.get(text_column, "")
                else:
                    # It might be a tuple or other structure
                    reference_summary = sample[1] if len(sample) > 1 else ""
                    source_text = sample[0] if len(sample) > 0 else ""
                
                # Generate summary using enhanced model and evaluate (simple)
                try:
                    predicted_summary = generate_summary_with_options(
                        source_text.strip(), 
                        method="enhanced", 
                        max_sentences=max_sentences
                    )

                    p, r, f1 = overlap_precision_recall_f1(reference_summary, predicted_summary)
                    total_precision += p
                    total_recall += r
                    total_f1 += f1
                    valid_samples += 1
                    
                except Exception as e:
                    st.warning(f"Error processing sample {i+1}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / sample_size)
            
            if valid_samples > 0:
                # Calculate average metrics
                avg_precision = (total_precision / valid_samples) * 100
                avg_recall = (total_recall / valid_samples) * 100
                avg_f1 = (total_f1 / valid_samples) * 100
                avg_accuracy = avg_f1  # F1 as accuracy proxy

                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{avg_accuracy:.1f}%", help="Overall model performance (F1)")
                with col2:
                    st.metric("📊 Precision", f"{avg_precision:.1f}%")
                with col3:
                    st.metric("🔄 Recall", f"{avg_recall:.1f}%")
                with col4:
                    st.metric("⚖️ F1-Score", f"{avg_f1:.1f}%")
                
                # Additional info
                st.info(f"📈 Evaluation completed on {valid_samples} samples from the test dataset")
                
                # Show sample evaluation
                if st.checkbox("Show sample evaluation details"):
                    sample_idx = st.slider("Select sample", 0, min(10, len(test_subset)-1), 0)
                    if sample_idx < len(test_subset):
                        sample = test_subset[sample_idx]
                        
                        # Get text and summary using dynamic column access
                        if hasattr(sample, 'keys'):
                            sample_text = sample.get(text_column, "")
                            sample_summary = sample.get(summary_column, "")
                        else:
                            sample_text = sample[0] if len(sample) > 0 else ""
                            sample_summary = sample[1] if len(sample) > 1 else ""
                        
                        st.write("**Original Text:**")
                        st.write(sample_text[:200] + "...")
                        st.write("**Reference Summary:**")
                        st.write(sample_summary)
                        
                        # Generate prediction for this sample
                        sents = split_sentences(sample_text.strip())
                        n = len(sents)
                        if n >= 8:
                            sample_max_sents = 4
                        elif n >= 5:
                            sample_max_sents = 3
                        else:
                            sample_max_sents = 2
                        predicted = generate_summary_with_options(
                            sample_text.strip(),
                            method="enhanced",
                            max_sentences=sample_max_sents
                        )
                        
                        st.write("**Model Prediction:**")
                        st.write(predicted)
                        
                        # Show individual metrics for this sample
                        p, r, f = overlap_precision_recall_f1(sample_summary, predicted)
                        st.write(f"**Sample Metrics:** Precision: {p*100:.1f}%, Recall: {r*100:.1f}%, F1: {f*100:.1f}%")
            
            else:
                st.error("No valid samples could be processed for evaluation.")
                
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            st.info("Make sure you have internet connection to load the dataset.")

st.divider()
st.caption("Rule-based prototype. Expect limited accuracy vs. neural models.")
