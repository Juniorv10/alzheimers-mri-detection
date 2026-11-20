import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

st.set_page_config(
    page_title="AI Alzheimer's MRI Detection & Clinical Screening",
    page_icon="🧠",
    layout="wide",
)

STAGE_DESCRIPTIONS = {
    "NonDemented": (
        "No radiographic signs of cortical atrophy or ventricular enlargement associated with Alzheimer's disease. "
        "Continue routine cognitive wellness habits, regular physical activity, and periodic health checkups."
    ),
    "VeryMildDemented": (
        "Subtle early hippocampal and medial temporal changes consistent with Very Mild Cognitive Impairment (prodromal stage). "
        "Early neurological evaluation, structured cognitive exercises, and lifestyle interventions are strongly advised."
    ),
    "MildDemented": (
        "Noticeable cortical thinning and memory-network changes consistent with Mild Alzheimer's Disease. "
        "Individuals typically experience challenges with recent recall and complex planning; timely clinical care and family support help slow progression."
    ),
    "ModerateDemented": (
        "Pronounced ventricular enlargement and diffuse cortical atrophy consistent with Moderate Alzheimer's Disease. "
        "Comprehensive neurological management, structured daily routines, and dedicated caregiver support are essential."
    ),
}

@st.cache_resource
def load_mri_model():
    ckpt = torch.load("models/alzheimers_resnet18.pth", map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    return model, ckpt

model, ckpt = load_mri_model()
CLASSES = ckpt["classes"]
IMG_SIZE = ckpt["img_size"]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_with_gradcam(pil_img):
    orig_resized = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    input_tensor = preprocess(orig_resized).unsqueeze(0)

    activations, gradients = [], []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.layer4[-1].conv2
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)[0].detach().numpy()
    pred_idx = int(np.argmax(probs))

    model.zero_grad()
    logits[0, pred_idx].backward()
    h1.remove()
    h2.remove()

    grads = gradients[0][0]
    acts = activations[0][0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for w, a in zip(weights, acts):
        cam += w * a

    cam = F.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_resized = np.array(Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    heatmap_rgb = np.uint8(plt.get_cmap("jet")(cam_resized)[:, :, :3] * 255)
    overlay = Image.fromarray(np.uint8(0.58 * np.array(orig_resized) + 0.42 * heatmap_rgb))
    return pred_idx, probs, orig_resized, overlay

st.title("🧠 AI-Enhanced Alzheimer's Disease Detection & Clinical Screening")
st.markdown(
    f"**Transfer Learning (`ResNet18`) + Grad-CAM Explainable AI + MMSE Cognitive Screening**  |  "
    f"Held-Out Test Accuracy: **{ckpt['test_accuracy']*100:.2f}%**  |  Macro ROC-AUC: **{ckpt['macro_auc']:.4f}**"
)

tab1, tab2, tab3 = st.tabs([
    "🔬 Brain MRI Detection & Grad-CAM",
    "📋 MMSE Cognitive Screening Test",
    "🏥 Care Tips & Specialist Directory",
])

# TAB 1: MRI CLASSIFICATION + GRAD-CAM + REPORT GENERATION
with tab1:
    col_in, col_out = st.columns([1, 1.35])

    with col_in:
        st.subheader("1️⃣ Patient & MRI Scan Input")
        patient_name = st.text_input("Patient Name:", value="Patient_001")
        patient_age = st.number_input("Patient Age:", min_value=18, max_value=110, value=68)

        source_mode = st.radio(
            "Select MRI Input Source:",
            ["Use Unseen Test-Set Sample MRI", "Upload Brain MRI Image"],
            horizontal=True,
        )

        selected_pil = None
        true_label_note = None

        if source_mode == "Upload Brain MRI Image":
            uploaded_mri = st.file_uploader("Upload Axial Brain MRI (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_mri is not None:
                selected_pil = Image.open(uploaded_mri)
        else:
            sample_choice = st.selectbox("Select an unseen test-set scan:", CLASSES)
            sample_path = f"sample_mris/{sample_choice}.jpg"
            if os.path.exists(sample_path):
                selected_pil = Image.open(sample_path)
                true_label_note = sample_choice

    with col_out:
        if selected_pil is not None:
            pred_idx, probs, orig_img, gradcam_img = predict_with_gradcam(selected_pil)
            pred_stage = CLASSES[pred_idx]
            pred_conf = probs[pred_idx] * 100

            st.subheader("2️⃣ Diagnostic Prediction & Explainability")
            if pred_stage == "NonDemented":
                st.success(f"**Predicted Stage: {pred_stage}** ({pred_conf:.1f}% confidence)")
            else:
                st.warning(f"**Predicted Stage: {pred_stage}** ({pred_conf:.1f}% confidence)")

            if true_label_note:
                st.caption(f"Ground-Truth Unseen Label: **{true_label_note}**")

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(orig_img, caption="Preprocessed Axial MRI (160×160)", width=260)
            with img_c2:
                st.image(gradcam_img, caption="Grad-CAM Attention Heatmap (Red = High Activation)", width=260)

            # Probability Bar Chart
            fig_prob = go.Figure(go.Bar(
                x=probs * 100,
                y=CLASSES,
                orientation="h",
                marker_color=["#2a9d8f" if i == pred_idx else "#8d99ae" for i in range(len(CLASSES))],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="auto",
            ))
            fig_prob.update_layout(
                title="Class Probability Distribution (%)",
                xaxis=dict(range=[0, 105], title="Confidence (%)"),
                height=230,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Clinical Interpretation & Downloadable Report
            st.info(f"**Clinical Interpretation:** {STAGE_DESCRIPTIONS[pred_stage]}")
            report_text = (
                f"====================================================\n"
                f"   ALZHEIMER'S MRI SCREENING & DIAGNOSTIC REPORT\n"
                f"====================================================\n"
                f"Date:            {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Patient Name:    {patient_name}\n"
                f"Patient Age:     {patient_age}\n"
                f"Predicted Stage: {pred_stage} ({pred_conf:.2f}% confidence)\n"
                f"Model Architecture: Fine-Tuned ResNet18 (Test Acc: {ckpt['test_accuracy']*100:.2f}%, Macro AUC: {ckpt['macro_auc']:.4f})\n\n"
                f"Stage Probability Breakdown:\n"
                + "\n".join(f"  - {c}: {p*100:.2f}%" for c, p in zip(CLASSES, probs))
                + f"\n\nClinical Notes:\n{STAGE_DESCRIPTIONS[pred_stage]}\n\n"
                f"Disclaimer: Decision-support research tool only; confirm findings with a qualified neurologist.\n"
            )
            st.download_button(
                "📥 Download Clinical Diagnostic Report (.txt)",
                data=report_text,
                file_name=f"{patient_name.replace(' ', '_')}_alzheimers_report.txt",
                mime="text/plain",
            )
        else:
            st.info("👈 Upload a brain MRI scan or pick an unseen sample scan on the left to run AI classification & Grad-CAM.")

# TAB 2: MMSE SCREENING QUESTIONNAIRE (FROM PROJECT ARCHITECTURE)
with tab2:
    st.subheader("📋 Mini-Mental State Examination (MMSE) Risk Screening")
    st.caption("Initial behavioral & cognitive screening prior to or alongside brain MRI imaging.")

    q1 = st.radio("1. How often do you experience memory lapses or forgetfulness?", ["Low", "Medium", "High"], horizontal=True)
    q2 = st.radio("2. Can you recall recent events or conversations easily?", ["High (Easily)", "Medium", "Low (Rarely)"], horizontal=True)
    q3 = st.radio("3. Do you often repeat questions or statements in conversations?", ["Low", "Medium", "High", "Very High"], horizontal=True)
    q4 = st.radio("4. How often do you struggle to find the right words when speaking?", ["Low", "Medium", "High"], horizontal=True)
    q5 = st.radio("5. Can you remember familiar locations and routes easily?", ["Very High (Easily)", "High", "Medium", "Low / Negative"], horizontal=True)
    q6 = st.radio("6. Are you able to manage your finances and bills independently?", ["Very High (Easily)", "High", "Medium", "Low"], horizontal=True)

    if st.button("Evaluate MMSE Screening Risk"):
        score_map = {
            "Low": 0, "High (Easily)": 0, "Very High (Easily)": 0,
            "High": 1, "Medium": 2,
            "Low (Rarely)": 3, "Low / Negative": 3, "Very High": 3,
        }
        # Adjust Q2/Q5/Q6 where 'High' means good recall
        risk_points = (
            {"Low": 0, "Medium": 2, "High": 3}[q1]
            + {"High (Easily)": 0, "Medium": 1, "Low (Rarely)": 3}[q2]
            + {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}[q3]
            + {"Low": 0, "Medium": 2, "High": 3}[q4]
            + {"Very High (Easily)": 0, "High": 1, "Medium": 2, "Low / Negative": 3}[q5]
            + {"Very High (Easily)": 0, "High": 1, "Medium": 2, "Low": 3}[q6]
        )
        max_points = 18
        risk_pct = (risk_points / max_points) * 100

        if risk_points <= 5:
            st.success(f"**Low Cognitive Risk ({risk_points}/{max_points} risk points — {risk_pct:.0f}%)**: Routine monitoring recommended.")
        elif risk_points <= 11:
            st.warning(f"**Moderate Cognitive Risk ({risk_points}/{max_points} risk points — {risk_pct:.0f}%)**: Consider a baseline Brain MRI scan & clinical consultation.")
        else:
            st.error(f"**Elevated Cognitive Risk ({risk_points}/{max_points} risk points — {risk_pct:.0f}%)**: Brain MRI screening (Tab 1) and neurology consultation strongly recommended.")

# TAB 3: PREVENTIVE TIPS & SPECIALIST DIRECTORY
with tab3:
    c_tips, c_docs = st.columns([1.2, 1])
    with c_tips:
        st.subheader("🌿 Evidence-Based Tips for Slowing Cognitive Decline")
        st.markdown("""
        1. **Stay Mentally Active:** Engage in puzzles, reading, learning new skills, or playing musical instruments to build cognitive reserve.
        2. **Stay Physically Active:** Regular aerobic exercise (walking, swimming, yoga) increases cerebral blood flow and neuroplasticity.
        3. **Follow a Brain-Healthy Diet:** Prioritize leafy greens, berries, whole grains, omega-3 fatty acids, and lean proteins (MIND / Mediterranean diet).
        4. **Prioritize Quality Sleep:** Aim for 7–9 hours of uninterrupted sleep nightly to support glymphatic clearance of beta-amyloid.
        5. **Stay Socially Engaged:** Maintain active social connections with family, friends, and community groups.
        """)
    with c_docs:
        st.subheader("🩺 Recommended Neurology Specialists")
        docs_df = pd.DataFrame([
            {"Specialist": "Dr. Ramesh CV", "Location": "Thalassery, Kannur", "Contact": "+91-8792065418"},
            {"Specialist": "Dr. KK Nirmal Raj", "Location": "Kochi, Kerala", "Contact": "+91-7942697713"},
            {"Specialist": "Dr. Sujith Ovallath", "Location": "Kannur, Kerala", "Contact": "+91-7942697943"},
        ])
        st.dataframe(docs_df, use_container_width=True, hide_index=True)
