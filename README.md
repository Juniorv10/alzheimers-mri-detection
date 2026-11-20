# 🧠 AI-Enhanced Alzheimer's Disease Detection & Clinical Screening

An end-to-end Deep Learning and Explainable AI (Grad-CAM) diagnostic web application for classifying 4-stage brain MRI scans (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`), paired with a Mini-Mental State Examination (MMSE) cognitive risk screener and automated clinical report generator.

## 📊 Model & Technical Highlights
* **Architecture:** Transfer Learning with fine-tuned **ResNet18** (`layer3`, `layer4`, and custom `Dropout(0.3) -> Linear(512, 4)` classification head) in PyTorch.
* **Class Imbalance Mitigation:** Addressed a severe 50:1 class imbalance (`3,200` NonDemented vs. `64` ModerateDemented) using a stratified 80/20 train/test split and inverse-frequency class-weighted `CrossEntropyLoss`.
* **Held-Out Evaluation:** Achieved **89.77% test accuracy** across **1,280 unseen MRI scans**.
* **Explainable AI (Grad-CAM):** Generates real-time Gradient-weighted Class Activation Mapping overlays highlighting the anatomical brain regions driving each prediction.
* **Clinical Screening Workflow:** Integrates a 6-question MMSE cognitive screening module, preventive brain-health guidelines, and downloadable patient diagnostic reports.

## 🛠️ Project Structure
~~~text
alzheimers-mri-detection/
├── app.py                        # Interactive Streamlit MRI Grad-CAM & MMSE Web App
├── alzheimers_mri_dl.ipynb       # EDA, Stratified Training, ROC-AUC & Grad-CAM Notebook
├── models/
│   └── alzheimers_resnet18.pth   # Exported PyTorch Checkpoint & Evaluation Metadata
├── sample_mris/                  # 4 Unseen Test-Set MRI Scans (1 per Dementia Stage)
├── visuals/                      # EDA Overview, Confusion Matrix & Grad-CAM Plots
└── requirements.txt              # Dependencies for Streamlit Cloud Deployment
~~~

## 💻 Run Locally
~~~bash
pip install -r requirements.txt
streamlit run app.py
~~~