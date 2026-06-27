# Multimodal clinical + Raman MIL

This folder replays selected multimodal MIL checkpoints using Raman cell bags and clinical tabular features.

Run:

```bash
python raman_multimodal_MIL_clinical_disease3_publication_final_CV3.py
python raman_multimodal_MIL_clinical_disease3_publication_final_CV4.py
python raman_multimodal_MIL_clinical_disease3_publication_final_CV5.py
```

Matching input folders:

```text
publication_MIL_clinical_disease3_inputs_CV3/
publication_MIL_clinical_disease3_inputs_CV4/
publication_MIL_clinical_disease3_inputs_CV5/
```

Expected selected outputs include metrics, patient-level predictions, misclassified cases, attention tables, selected checkpoints, tuned JSON files, and clinical preprocessors. Training curves and figures are excluded.

Before public release, check that all clinical tables are de-identified and approved for release.
