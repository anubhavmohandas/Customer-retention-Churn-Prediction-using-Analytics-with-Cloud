import streamlit as st
import pandas as pd

st.title("Page 1: Data Upload")

# Security configurations
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['csv']

uploaded_file = st.file_uploader("Upload CSV file", type=ALLOWED_EXTENSIONS)

if uploaded_file is not None:
    
    # ========== SECURITY CHECK 1: File Size ==========
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large! Maximum size allowed is {MAX_FILE_SIZE_MB}MB. Your file: {file_size_mb:.2f}MB")
        st.stop()
    
    # ========== SECURITY CHECK 2: File Extension ==========
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        st.error(f"❌ Invalid file type! Only {', '.join(ALLOWED_EXTENSIONS).upper()} files are allowed.")
        st.stop()
    
    # ========== SECURITY CHECK 3: CSV Parsing ==========
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty!")
        st.stop()
    except pd.errors.ParserError:
        st.error("❌ Invalid CSV format! Please check your file structure.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()
    
    # ========== SECURITY CHECK 4: Data Validation ==========
    if df.empty:
        st.error("❌ The uploaded file contains no data!")
        st.stop()
    
    if len(df.columns) < 2:
        st.error("❌ Dataset must have at least 2 columns (features + target)!")
        st.stop()
    
    # ========== SECURITY CHECK 5: Row Limit ==========
    MAX_ROWS = 100000
    if len(df) > MAX_ROWS:
        st.warning(f"⚠️ Large dataset detected ({len(df)} rows). Limiting to {MAX_ROWS} rows for performance.")
        df = df.head(MAX_ROWS)
    
    # ========== SECURITY CHECK 6: Sanitize Column Names ==========
    # Remove special characters from column names
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df.columns = df.columns.str.strip()
    
    # Store in session state
    st.session_state["data"] = df
    st.session_state["file_name"] = file_name
    
    # Success message with file info
    st.success("✅ File uploaded successfully!")
    
    # Display file info
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("File Size", f"{file_size_mb:.2f} MB")
    
    st.write("### Preview of Data")
    st.dataframe(df.head(10))
    
    # Show column info
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info)
