# **Room Impulse Response (RIR) Reconstruction**
**LSTM and Conv. Net based EDC Prediction**
![Banner](https://img.shields.io/badge/ICASSP-2026-blue) 
![Python](https://img.shields.io/badge/Python-3.10%2B-green) 
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange) 
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 1. Introduction

This repository provides a **deep learning–based framework** for predicting **Energy Decay Curves (EDCs)** and reconstructing **Room Impulse Responses (RIRs)** directly from **room geometric and material properties**.

The method uses a **Long Short-Term Memory (LSTM)** neural network trained on simulated room acoustic datasets, enabling accurate and efficient prediction of reverberation characteristics without running computationally expensive acoustic simulations.

### **Inputs**

* 🏠 Room dimensions (Length, Width, Height)
* 📉 Material absorption coefficients (7 octave-band values)
* 🔊 Source position (x, y, z)
* 🎧 Receiver position (x, y, z)

### **Outputs**

* Predicted EDC curve
* Reconstructed RIR waveform
* Temporal and spectral plots for evaluation

This framework is applicable to:

* Real-time auralization engines
* Architectural acoustic design
* Speech enhancement and dereverberation
* Intelligent room tuning

---

## 📂 2. Repository Structure

```
.
├── dataset/
│   └── room_acoustic_largedataset/
│       ├── EDC/                       # Numpy .npy files containing reference EDCs (Download link below)
│       ├── roomFeaturesDataset.csv    # Room geometry + material features
│       └── dataSource.txt             # Dataset description
│
├── Models/
│   ├── best_model.ckpt               # Trained LSTM model checkpoint (Download link below)
│   ├── scaler_X_*.save               # Scaler for input features
│   └── scaler_edc_*.save             # Scaler for EDC predictions
│
├── inference_edcModelPytorchLighteningV3.py   # Main inference and analysis script
├── requirements.txt
└── README.md
```

---

## 📥 3. Downloading the Dataset and Model

Before running inference, **download the dataset and pretrained model** from Zenodo:

👉 [EDC Datset](https://zenodo.org/records/17210197): DOI: 10.5281/zenodo.17210196   
👉 [RIR Datset](https://zenodo.org/records/17503961): DOI: 10.5281/zenodo.17503960  
👉 [Model](https://zenodo.org/records/17215057): DOI: 10.5281/zenodo.17215057  
👉 [RIR Measurements](https://cloud.tu-ilmenau.de/s/oqN3rcg4N2ETBza)    
👉 [Open Dataset RIR Measurements](https://github.com/RoyJames/room-impulse-responses)  


After downloading:

1. Place the `EDC` folder inside `dataset/room_acoustic_largedataset/`
2. Place the model checkpoint `.ckpt` and scaler `.save` files inside `Models/`

---

## 🧪 4. Model Inference

The inference is performed using:

```
inference_edcModelPytorchLighteningV3.py
```

Two modes are supported:

### **Mode 1 – Existing Dataset**

* Randomly selects a room from the dataset
* Loads the corresponding reference EDC
* Runs model inference and compares prediction with ground truth
* Generates EDC, RIR, and FFT plots

### **Mode 2 – Custom Room Features**

* Allows the user to **manually input room geometry and materials**
* Generates EDC and RIR predictions **without reference data**
* Useful for real-world rooms or hypothetical designs

---

### ▶ Run the Inference Script

```bash
python inference_edcModelPytorchLighteningV3.py
```

You will be prompted:

```
Select data source:
1. Use existing dataset
2. Enter custom room features
```

---

## 📝 5. Custom Room Features Format

For **Mode 2**, input the features in this order:

| Parameter               | Count | Description                         |
| ----------------------- | ----- | ----------------------------------- |
| Room Dimensions         | 3     | Length, Width, Height (m)           |
| Source Position         | 3     | X, Y, Z (m)                         |
| Receiver Position       | 3     | X, Y, Z (m)                         |
| Absorption Coefficients | 7     | Octave-band absorption values (0–1) |

### **Example Input**

```
Room Dimensions: 3.0, 4.0, 3.0
Source Position: 1.4, 1.4, 1.5
Receiver Position: 1.8, 3.0, 1.5
Absorption: 0.14, 0.27, 0.36, 0.3, 0.24, 0.24, 0.03
```

---

## 📊 6. Output and Visualization

For each inference run, the following are generated:

* 📈 Predicted EDC Curve
* 🔊 Reconstructed RIR using the **Random Sign-Sticky** method
* 🌐 FFT Magnitude Spectrum (log frequency)
* 🔍 Zoomed plots for fine detail inspection

All outputs are saved in:

```
inference_results/
```

Including plots:

* `comparison_plot.png`
* `comparison_plot_Zoom.png`

---

## 🧠 7. Model Training (Overview)

The LSTM model was trained using a large simulated dataset of shoebox rooms.

### **Training Configuration**

* **Input features:** 16 (geometry, positions, 7 absorptions)
* **Target:** EDC sequence (96,000 samples → 2 s @ 48 kHz)
* **Loss function:** MSE + custom energy-decay loss
* **Optimizer:** Adam (LR = 0.001)
* **Early stopping:** Based on validation loss (10 epochs patience)
* **Framework:** PyTorch Lightning

You may adapt the training script to retrain the model with your own dataset if desired.

---

## 🧰 8. Installation & Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes **pinned versions** of all dependencies to ensure compatibility.

Example contents:

```
torch==2.3.0
pytorch-lightning==2.3.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.5.1
SoundFile==0.12.1
joblib==1.4.2
```

---

## 📝 9. Citation

If you use this repository in your research, please cite:

```
@inproceedings{ImranSchuller2025aRIR,
  title={Deep learning-based prediction of energy decay curves from room geometry and material properties},
  author={Imran MUhammad, Gerald Schuller},
  booktitle={https://arxiv.org/abs/2509.24769},
  year={2026},
  DOI= {https://doi.org/10.48550/arXiv.2509.24769}
}
```
---

```
@inproceedings{ImranSchuller2025bRIR,
  title={Room impulse response prediction with neural networks: from energy decay curves to perceptual validation},
  author={Imran MUhammad, Gerald Schuller},
  booktitle={https://arxiv.org/abs/2509.24834},
  year={2025},
  DOI= {https://doi.org/10.48550/arXiv.2509.24834}
}
```


---

## 📧 10. Contact

For inquiries, suggestions, or collaborations:

* **Contact:** Dr. Imran
* **Affiliation:** Applied Media Systems, TU Ilmenau Germany
* **Email:** [muhammad.imran@tu-ilmenau.de](mailto:muhammad.imran@tu-ilmenau.de)

---

## ✅ Summary

This repository provides a **practical, academically validated framework** for:

* Predicting EDCs using LSTM networks
* Reconstructing full-band RIRs via stochastic methods
* Supporting both dataset-based and custom room inference
* Enabling fast and accurate acoustic parameter estimation for auralization and speech applications.

