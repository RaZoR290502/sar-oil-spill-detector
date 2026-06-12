import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import tempfile

st.set_page_config(page_title="SAR Oil Spill Detection", layout="wide")

st.title("SAR Oil Spill Detection App")
st.write("Upload a preprocessed Sentinel-1 SAR GeoTIFF image to detect possible oil spill areas.")

uploaded_file = st.file_uploader("Upload SAR GeoTIFF file", type=["tif", "tiff"])

st.sidebar.header("Detection Parameters")

THRESHOLD_PERCENTILE = st.sidebar.slider(
    "Threshold Percentile",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

KERNEL_SIZE = st.sidebar.selectbox(
    "Kernel Size",
    options=[3, 5, 7, 9],
    index=2
)

MIN_AREA = st.sidebar.slider(
    "Minimum Area (pixels)",
    min_value=10,
    max_value=1000,
    value=150,
    step=10
)

PIXEL_SIZE = st.sidebar.number_input(
    "Pixel Size (m)",
    min_value=1,
    value=10
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    with rasterio.open(file_path) as src:
        sar_db = src.read(1).astype(float)

    st.subheader("Input SAR Image Information")
    st.write("Minimum value:", float(np.min(sar_db)))
    st.write("Maximum value:", float(np.max(sar_db)))

    p2 = np.percentile(sar_db, 2)
    p98 = np.percentile(sar_db, 98)

    filtered = cv2.medianBlur(
        sar_db.astype(np.float32),
        3
    )

    threshold = np.percentile(
        filtered,
        THRESHOLD_PERCENTILE
    )

    oil_mask = filtered < threshold

    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    clean = cv2.morphologyEx(
        oil_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        kernel
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    final = np.zeros_like(clean)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > MIN_AREA:
            final[labels == i] = 1

    pixel_area = PIXEL_SIZE * PIXEL_SIZE
    num_pixels = np.sum(final)

    total_area_m2 = num_pixels * pixel_area
    total_area_km2 = total_area_m2 / 1e6

    if num_pixels > 0:
        spill_mean = np.mean(sar_db[final == 1])
    else:
        spill_mean = 0

    st.subheader("Detection Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Threshold Used", f"{threshold:.6f}")
    col2.metric("Detected Pixels", f"{int(num_pixels)}")
    col3.metric("Area (km²)", f"{total_area_km2:.4f}")
    col4.metric("Mean Backscatter", f"{spill_mean:.6f}")

    def show_image(image, title, cmap="gray", overlay=False):
        fig, ax = plt.subplots(figsize=(6, 5))

        if overlay:
            ax.imshow(sar_db, cmap="gray", vmin=p2, vmax=p98)
            ax.imshow(final, cmap="jet", alpha=0.5)
        else:
            ax.imshow(image, cmap=cmap, vmin=p2, vmax=p98)

        ax.set_title(title)
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Processing Outputs")

    colA, colB = st.columns(2)

    with colA:
        show_image(sar_db, "SAR Image (dB)")

    with colB:
        show_image(filtered, "Filtered Image")

    colC, colD = st.columns(2)

    with colC:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(oil_mask, cmap="gray")
        ax.set_title("Initial Detection")
        ax.axis("off")
        st.pyplot(fig)

    with colD:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(final, cmap="gray")
        ax.set_title("Final Detection")
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Oil Spill Overlay")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sar_db, cmap="gray", vmin=p2, vmax=p98)
    ax.imshow(final, cmap="jet", alpha=0.5)
    ax.set_title("Oil Spill Overlay")
    ax.axis("off")
    st.pyplot(fig)

else:
    st.info("Please upload a SAR GeoTIFF file to start detection.")