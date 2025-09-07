import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# Dummy data
# ========================
A = "Wardah Lightening Series"
B = "Wardah UV Shield Sunscreen Gel SPF 30 PA+++"
C = "Wardah Lightening Face Toner (125 mL)"
D = "Wardah Exclusive Matte Lip Cream"
E = "Wardah Glasting Liquid Lip"
products = {
    "Wardah Lightening Series": np.array([0.9, 0.1, 0.2]),
    "Wardah UV Shield Sunscreen Gel SPF 30 PA+++": np.array([0.8, 0.2, 0.1]),
    "Wardah Lightening Face Toner (125 mL)": np.array([0.1, 0.9, 0.3]),
    "Wardah Exclusive Matte Lip Cream": np.array([0.2, 0.1, 0.9]),
    "Wardah Glasting Liquid Lip": np.array([0.3, 0.8, 0.2]),
}

user_purchases = {
    "U1": [A, B],
    "U2": [C],
    "U3": [D, E],
    "U4": [B, C],
    "U5": [A, D],
    "U6": [A],
    "U7": [B, E],
    "U8": [C, D],
    "U9": [A, C],
    "U10": [E],
}

cf_matrix = pd.DataFrame({
    A: [0.95,0.1,0.2,0.3,0.85,0.9,0.15,0.25,0.2,0.4],  # user U1, U5, U6 tinggi
    B: [0.2,0.9,0.3,0.8,0.1,0.2,0.7,0.4,0.3,0.6],
    C: [0.1,0.8,0.2,0.7,0.15,0.2,0.25,0.9,0.85,0.3],
    D: [0.3,0.2,0.95,0.25,0.9,0.3,0.4,0.7,0.2,0.8],  # user U3, U5, U10 tinggi di D
    E: [0.4,0.3,0.85,0.4,0.25,0.2,0.9,0.6,0.2,0.95], # user U3, U7, U10 tinggi di E
}, index=["U1","U2","U3","U4","U5","U6","U7","U8","U9","U10"])

# ========================
# Streamlit UI
# ========================
st.title("E-commerce Recommendation Demo")

product_choice = st.selectbox("Choose Product:", list(products.keys()))
strategy = st.selectbox("Choose Strategy:", ["Exploitative (Content-Based)", 
                                            "Explorative (Collaborative Filtering)", 
                                            "Balanced (Hybrid)"])

threshold = st.slider("Threshold Similarity/Preference", 0.0, 1.0, 0.7)

# ========================
# Content-Based
# ========================
def get_content_based(product_choice, threshold):
    product_vec = products[product_choice]
    results = {}
    for user, items in user_purchases.items():
        user_vec = np.mean([products[i] for i in items], axis=0)
        sim = cosine_similarity([user_vec], [product_vec])[0][0]
        results[user] = sim
    df = pd.DataFrame(results.items(), columns=["User","Similarity"]).sort_values("Similarity", ascending=False)
    return df[df["Similarity"] >= threshold]

# ========================
# Collaborative Filtering
# ========================
def get_cf(product_choice, threshold):
    scores = cf_matrix[product_choice]
    df = scores.reset_index().rename(columns={"index":"User", product_choice:"Preference"})
    return df[df["Preference"] >= threshold].sort_values("Preference", ascending=False)

# ========================
# Hybrid
# ========================
def get_hybrid(product_choice, threshold, ratio_content=0.7):
    cb_df = get_content_based(product_choice, threshold)
    cf_df = get_cf(product_choice, threshold)
    print(f"cf_df: {cf_df}")
    n_cb = max(1, int(len(cb_df) * ratio_content))
    n_cf = max(1, int(len(cf_df) * (1 - ratio_content)))

    cb_top = cb_df.head(n_cb).copy()
    cb_top["Source"] = "Content-Based"

    cf_top = cf_df.head(n_cf).copy()
    print(f"cf_top: {cf_top}")
    cf_top["Source"] = "Collaborative Filtering"

    # gabung langsung, tidak dihapus overlap
    hybrid = pd.concat([cb_top[["User","Source"]], cf_top[["User","Source"]]], ignore_index=True)

    return hybrid

# ========================
# Main Logic
# ========================
if strategy == "Exploitative (Content-Based)":
    filtered = get_content_based(product_choice, threshold)
    st.subheader("Content-Based Results")
    st.dataframe(filtered)

elif strategy == "Explorative (Collaborative Filtering)":
    filtered = get_cf(product_choice, threshold)
    st.subheader("Collaborative Filtering Results")
    st.dataframe(filtered)

elif strategy == "Balanced (Hybrid)":
    filtered = get_hybrid(product_choice, threshold)
    st.subheader("Hybrid Results (70% Content-Based + 30% CF)")
    st.dataframe(filtered)

# ========================
# Strategi Promo
# ========================
if not filtered.empty:
    st.subheader("Promotion Strategy")
    n = len(filtered)
    top = int(n*0.3) if n>=3 else 1
    mid = int(n*0.6) if n>=3 else 1

    high = filtered.head(top)
    medium = filtered.iloc[top:mid]
    low = filtered.iloc[mid:]

    st.write("**High User Bucket:**")
    st.dataframe(high)
    st.write("**Medium User Bucket:**")
    st.dataframe(medium)
    st.write("**Low User Bucket:**")
    st.dataframe(low)
