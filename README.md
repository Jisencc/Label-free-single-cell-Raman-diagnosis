# Raman Spectroscopy Patient-Level Classification Code

This repository contains the publication-ready reproduction code and locked input artifacts for patient-level Raman spectroscopy classification experiments. The code covers classical SVM/PCA Raman-only models, Raman-only MIL attention models, and multimodal clinical + Raman MIL models.

The repository is organized so that every reported CV setting has its own prepared input folder and its own runnable public script. The public scripts replay locked folds, selected repetitions, tuned parameters, saved checkpoints, and selected outputs. Training-monitoring curve/figure folders are intentionally excluded from the public release.

## Repository structure

```text
code4publication/
├── 3-cls_identification/
│   ├── publication_stage1_disease3_inputs_CV3/
│   ├── publication_stage1_disease3_inputs_CV4/
│   ├── publication_stage1_disease3_inputs_CV5/
│   ├── publication_stage1_disease3_inputs_CV3_Ablation/
│   ├── publication_stage1_disease3_inputs_CV4_Ablation/
│   ├── publication_stage1_disease3_inputs_CV5_Ablation/
│   ├── raman_stage1_disease3_publication_final_CV3.py
│   ├── raman_stage1_disease3_publication_final_CV4.py
│   ├── raman_stage1_disease3_publication_final_CV5.py
│   ├── raman_stage1_disease3_fpch_ablation_publication_final_CV3.py
│   ├── raman_stage1_disease3_fpch_ablation_publication_final_CV4.py
│   ├── raman_stage1_disease3_fpch_ablation_publication_final_CV5.py
│   ├── 3-cls_pub.sh
│   └── 3-cls_ablation_pub.sh
│
├── Binary_identification/
│   ├── publication_stage1_binary_inputs_CV3/
│   ├── publication_stage1_binary_inputs_CV4/
│   ├── publication_stage1_binary_inputs_CV5/
│   ├── publication_stage1_binary_ablation_inputs_CV3/
│   ├── publication_stage1_binary_ablation_inputs_CV4/
│   ├── publication_stage1_binary_ablation_inputs_CV5/
│   ├── raman_stage1_binary_publication_final_CV3.py
│   ├── raman_stage1_binary_publication_final_CV4.py
│   ├── raman_stage1_binary_publication_final_CV5.py
│   ├── raman_stage1_binary_fpch_ablation_publication_final_CV3.py
│   ├── raman_stage1_binary_fpch_ablation_publication_final_CV4.py
│   ├── raman_stage1_binary_fpch_ablation_publication_final_CV5.py
│   ├── binary_classic_pub.sh
│   └── binary_classic_ablation_pub.sh
│
├── MIL_attention/
│   ├── 3-CLS/
│   │   ├── publication_stage1_disease3_MIL_inputs_CV3/
│   │   ├── publication_stage1_disease3_MIL_inputs_CV4/
│   │   ├── publication_stage1_disease3_MIL_inputs_CV5/
│   │   ├── raman_stage1_MIL_disease3_publication_final_CV3.py
│   │   ├── raman_stage1_MIL_disease3_publication_final_CV4.py
│   │   ├── raman_stage1_MIL_disease3_publication_final_CV5.py
│   │   └── 3-cls_mil_pub.sh
│   │
│   ├── Binary/
│   │   ├── publication_stage1_binary_MIL_inputs_CV3/
│   │   ├── publication_stage1_binary_MIL_inputs_CV4/
│   │   ├── publication_stage1_binary_MIL_inputs_CV5/
│   │   ├── raman_stage1_MIL_binary_publication_final_CV3.py
│   │   ├── raman_stage1_MIL_binary_publication_final_CV4.py
│   │   ├── raman_stage1_MIL_binary_publication_final_CV5.py
│   │   ├── selected_binary_reps_template_MIL_binary_CV3.csv
│   │   ├── selected_binary_reps_template_MIL_binary_CV4.csv
│   │   ├── selected_binary_reps_template_MIL_binary_CV5.csv
│   │   └── binary_mil_pub.sh
│   │
│   └── Clinical/
│       ├── publication_MIL_clinical_disease3_inputs_CV3/
│       ├── publication_MIL_clinical_disease3_inputs_CV4/
│       ├── publication_MIL_clinical_disease3_inputs_CV5/
│       ├── raman_multimodal_MIL_clinical_disease3_publication_final_CV3.py
│       ├── raman_multimodal_MIL_clinical_disease3_publication_final_CV4.py
│       ├── raman_multimodal_MIL_clinical_disease3_publication_final_CV5.py
│       └── clinical_mil_pub.sh
│
└── clinical/
    ├── publication_clinical_inputs_CV3/
    ├── publication_clinical_inputs_CV4/
    ├── publication_clinical_inputs_CV5/
    ├── raman_multimodal_disease3_publication_final_CV3.py
    ├── raman_multimodal_disease3_publication_final_CV4.py
    ├── raman_multimodal_disease3_publication_final_CV5.py
    └── clinical_mil.pub.sh
```

If your local folder contains both `clinical/` and `MIL_attention/Clinical/`, keep the one that corresponds to the final code version you intend to publish. The `MIL_attention/Clinical/` folder is the recommended location for the final multimodal MIL clinical checkpoint-replay code.

## Task families

### 1. Classical Raman-only identification

This part uses patient-level locked reproduction for classical Raman SVM/PCA experiments.

- `3-cls_identification/`: disease-only 3-class tasks, Liver vs Bile vs Pancreas.
- `Binary_identification/`: Control vs Liver, Control vs Bile, and Control vs Pancreas.
- `*_fpch_ablation_*`: FP+CH ablation, where the all-wavenumber view is removed.
- `*_publication_final_CV3.py`, `*_CV4.py`, `*_CV5.py`: CV-specific public reproduction scripts.

### 2. Raman-only MIL attention

This part replays trained MIL models from selected checkpoints and regenerates selected metrics, patient predictions, and attention outputs.

- `MIL_attention/3-CLS/`: disease-only 3-class Raman-only MIL.
- `MIL_attention/Binary/`: Control-vs-disease Raman-only MIL.
- The public scripts do not retrain. They load selected checkpoint bundles and replay inference.
- Training curve and figure folders are intentionally not included.

### 3. Multimodal clinical + Raman MIL

This part replays multimodal MIL checkpoints that combine Raman cell bags with clinical tabular features.

- `MIL_attention/Clinical/`: preferred final location for multimodal MIL clinical reproduction.
- `clinical/`: optional/legacy location if used in your final folder layout.
- Clinical preprocessors and selected checkpoints are kept in the prepared input folders.
- Curve outputs are intentionally excluded from the public release.

## What is included and what is intentionally excluded

Included:

- public Python reproduction scripts
- CV-specific locked input folders
- selected fold/repetition tables
- selected tuned JSON files
- selected checkpoint bundles
- patient-level prediction tables
- misclassified-patient tables where available
- attention cell and patient-summary outputs where available
- HPC wrapper shell scripts

Excluded:

- `curves/`
- `curves_inner/`
- `curves_outer/`
- `figures/`
- any private raw data that cannot be released
- any unselected exploratory repetitions/folds

## Environment

Recommended Python environment:

```bash
conda create -n raman_pub python=3.10 -y
conda activate raman_pub
pip install numpy pandas scipy scikit-learn torch matplotlib tqdm joblib openpyxl
```

For GPU MIL replay, install the PyTorch build that matches your CUDA/HPC environment. CPU inference is supported but slower for MIL attention export.

## Running the code

Always run the script whose CV suffix matches the input folder. Do not mix `CV3` scripts with `CV4` or `CV5` input folders.

### Classical 3-class Raman-only identification

```bash
cd code4publication/3-cls_identification
python raman_stage1_disease3_publication_final_CV3.py
python raman_stage1_disease3_publication_final_CV4.py
python raman_stage1_disease3_publication_final_CV5.py
```

FP+CH ablation:

```bash
cd code4publication/3-cls_identification
python raman_stage1_disease3_fpch_ablation_publication_final_CV3.py
python raman_stage1_disease3_fpch_ablation_publication_final_CV4.py
python raman_stage1_disease3_fpch_ablation_publication_final_CV5.py
```

### Classical binary Raman-only identification

```bash
cd code4publication/Binary_identification
python raman_stage1_binary_publication_final_CV3.py
python raman_stage1_binary_publication_final_CV4.py
python raman_stage1_binary_publication_final_CV5.py
```

FP+CH ablation:

```bash
cd code4publication/Binary_identification
python raman_stage1_binary_fpch_ablation_publication_final_CV3.py
python raman_stage1_binary_fpch_ablation_publication_final_CV4.py
python raman_stage1_binary_fpch_ablation_publication_final_CV5.py
```

### Raman-only 3-class MIL attention

```bash
cd code4publication/MIL_attention/3-CLS
python raman_stage1_MIL_disease3_publication_final_CV3.py
python raman_stage1_MIL_disease3_publication_final_CV4.py
python raman_stage1_MIL_disease3_publication_final_CV5.py
```

### Raman-only binary MIL attention

```bash
cd code4publication/MIL_attention/Binary
python raman_stage1_MIL_binary_publication_final_CV3.py
python raman_stage1_MIL_binary_publication_final_CV4.py
python raman_stage1_MIL_binary_publication_final_CV5.py
```

### Multimodal clinical + Raman MIL

Preferred final location:

```bash
cd code4publication/MIL_attention/Clinical
python raman_multimodal_MIL_clinical_disease3_publication_final_CV3.py
python raman_multimodal_MIL_clinical_disease3_publication_final_CV4.py
python raman_multimodal_MIL_clinical_disease3_publication_final_CV5.py
```

If you keep the alternative `clinical/` folder:

```bash
cd code4publication/clinical
python raman_multimodal_disease3_publication_final_CV3.py
python raman_multimodal_disease3_publication_final_CV4.py
python raman_multimodal_disease3_publication_final_CV5.py
```

## Running on HPC

The `.sh` files are simple wrappers around the Python commands above. On a SLURM cluster, inspect the script first, then submit it:

```bash
cd code4publication/MIL_attention/Binary
cat binary_mil_pub.sh
sbatch binary_mil_pub.sh
```

For local Windows users, running the Python files directly is usually easier than running `.sh` files. If you want to run shell scripts on Windows, use Git Bash or WSL.

## Expected outputs

Classical SVM/PCA scripts usually produce:

```text
*_SUMMARY.csv
*_patient_preds.csv
*_splits.csv
*_roc_points.csv
```

MIL replay scripts usually produce:

```text
*_SUMMARY.csv
*_PATIENT_PREDS*.csv
*_MISCLASSIFIED*.csv
*_ATTN_CELLS*.csv
*_ATTN_PATIENT_SUMMARY*.csv
checkpoints/ or models_outer/
tuned/ or tuned_params/
```

The exact filenames vary slightly by task family, but every public script is configured to write outputs into its own reproduction output folder.

## Reproducibility notes

- All public scripts use locked selected repetitions/folds prepared from the original training outputs.
- Classical SVM/PCA scripts replay locked splits, tuned hyperparameters, fusion weights, and temperature values.
- MIL scripts replay selected saved checkpoints and selected tuned metadata.
- MIL public scripts should not generate training curves or figures.
- For binary MIL, labels are manually encoded as `0 = Control` and `1 = disease`, so the LabelEncoder ordering issue from some SVM scripts does not apply.
- For 3-class disease tasks, the disease order is handled consistently by the public scripts.

## Data and privacy

Before making the repository public, check that no private or identifiable patient information is included. If clinical covariates are required for reproduction, release only de-identified and approved tables, or provide synthetic/example inputs and instructions for requesting the original data through the appropriate governance route.

## Citation

If you use this repository, please cite the associated manuscript.

```bibtex
@article{your_paper_key,
  title   = {Patient-level Raman spectroscopy classification using classical and MIL-based models},
  author  = {Author list},
  journal = {Journal name},
  year    = {2026},
  doi     = {DOI when available}
}
```

## License

Add the final license chosen by the authors and institution. If unsure, discuss with the supervisory team before making the repository public.
