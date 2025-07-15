# DisasterNet-LLM

A multimodal Large Language Model for disaster classification using text, images, and geospatial data.


## Environment Setup

1. Install Python 3.10 or higher.
2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Datasets Used

This project uses the following datasets. Please download each dataset from the original research paper or official source. Some datasets may require you to request permission or access from the authors or data providers:

- **Disaster Image Dataset** (Niloy et al., 2020): [Kaggle link](https://www.kaggle.com/datasets/frgfm/disaster-image-dataset)
- **MEDIC Dataset** (Mozannar et al., 2021): [GitHub link](https://github.com/husseinmozannar/multimodal-deep-learning-for-disaster-response)
- **ERA5 Meteorological Raster Dataset**: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **Environmental News Dataset**: [Kaggle link](https://www.kaggle.com/datasets/sbhatti/one-million-climate-headlines) or as referenced in the project
- **Delhi Urban Risk Dataset**: Not publicly available; request access from NIDM or the dataset authors as described in the relevant research paper

> **Note:** Always follow the terms of use and citation requirements for each dataset. Some datasets may require you to fill out a request form or obtain explicit permission from the data owners.

## Dataset-Specific Notes & Troubleshooting

### Disaster Image Dataset
- Ensure the folder structure matches the expected class/subclass layout.
- Preprocessing will log missing/corrupt images and folder issues to `Disaster Image Dataset/preprocess_error.log`.
- If you see errors about missing images or classes, check your extraction and folder names.

### MEDIC Dataset
- Requires class folders for each disaster type (earthquake, flood, fire, etc.).
- Preprocessing logs missing files, label mismatches, and class imbalance warnings to `MEDIC Dataset/preprocess_error.log`.
- Multi-label support may require custom handling; check logs for label consistency issues.

### ERA5 Meteorological Raster Dataset
- Only NetCDF (`.nc`) or GRIB files are supported.
- Preprocessing checks for required variables and logs missing/corrupt files to `ERA5 Meteorological Raster Dataset/preprocess_error.log`.
- Large files may require significant memory; errors will be logged if files cannot be loaded.

### Environmental News Dataset
- Input files must be CSV or JSON with at least `title`, `content`, `date`, and `publisher` columns.
- Preprocessing logs missing columns, encoding issues, and duplicates to `Environmental News Dataset/preprocess_error.log`.
- If you see errors about missing columns, check your data format and field names.

### Delhi Urban Risk Dataset
- Input files must be CSV or JSON with `event`, `date`, and `location` columns.
- Preprocessing logs missing/malformed entries to `Delhi Urban Risk Dataset/preprocess_error.log`.
- If you see errors about missing columns or empty files, check your data format and content.

### General Troubleshooting
- All preprocessing scripts log errors and warnings to a dataset-specific log file in the dataset folder.
- If you encounter a `FileNotFoundError` or `RuntimeError` when loading a dataset, check the corresponding log file and ensure preprocessing has completed successfully.
- For further debugging, run preprocessing scripts with a clean dataset copy and review the log output for actionable issues.

## Usage

- Training, evaluation, and visualization scripts are in the `scripts/` folder.
- Experiment scripts for each dataset/task are in the `experiments/` folder.

## Citation

If you use this project, please cite the original datasets and models as referenced in the documentation. 

**Can We Predict the Unpredictable? Leveraging DisasterNet-LLM for Multimodal Disaster Classification**  
*Kulahara, Manaswi, Gautam Siddharth Kashyap, Nipun Joshi, and Arpita Soni*  
arXiv preprint: [arXiv:2506.23462](https://arxiv.org/abs/2506.23462) (2025)  
✅ Accepted at the **2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)**,  
Brisbane, Australia (3–8 August 2025)
