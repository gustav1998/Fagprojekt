# Large Datasets

This folder is for larger raw datasets used in the 50k-500k instance
experiment set. The downloaded data files are ignored by Git because several
of them are too large for ordinary repository tracking.

Place each dataset in the folder shown below before running preprocessing.

| Dataset config | Expected local files | Source |
|---|---|---|
| `connect_4` | `connect_4/connect-4.data` | https://archive.ics.uci.edu/dataset/26/connect+4 |
| `shuttle` | `shuttle/shuttle.trn`, `shuttle/shuttle.tst` | https://archive.ics.uci.edu/dataset/148/statlog+shuttle |
| `census_income_kdd` | `census_income_kdd/census-income.data`, `census_income_kdd/census-income.test` | https://archive.ics.uci.edu/dataset/117/census+income+kdd |
| `secondary_mushroom` | `secondary_mushroom/secondary_data.csv` | https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset |
| `phiusiil_phishing` | `phiusiil_phishing/PhiUSIIL_Phishing_URL_Dataset.csv` | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset |
| `rt_iot2022` | `rt_iot2022/RT_IOT2022` | https://archive.ics.uci.edu/dataset/942/rt-iot2022 |
| `skin_segmentation` | `skin_segmentation/Skin_NonSkin.txt` | https://archive.ics.uci.edu/dataset/229/skin+segmentation |
| `sensorless_drive` | `sensorless_drive/Sensorless_drive_diagnosis.txt` | https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis |
| `aps_failure` | `aps_failure/aps_failure_training_set.csv`, `aps_failure/aps_failure_test_set.csv` | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks |

Example preprocessing command:

```bash
uv run python -m src.data_pipeline.make_dataset2 --dataset connect_4 --representation both
```
