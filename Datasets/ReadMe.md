<h1 align="center">
  <img src="../assets/fig/logo.png" alt="mTSBench Icon" width="32" style="vertical-align: middle; margin-right: 8px;">
  <b>mTSBench</b>
</h1>

**mTSBench** contains a collection of 344 multivariate time series from 19 datasets commonly used in anomaly detection research. Each subfolder corresponds to a specific dataset (e.g., `MSL`, `SMAP`, `CalIt2`), and contains `.csv` files such as `*_train.csv`, `*_test.csv`, and `*_val.csv`. Check `data_summary.csv` for details on each time series. 


### Overview of Multivariate Time-Series Datasets, details in `data_summary.csv`

| **Dataset**     | **Domain**                 | **#TS** | **#Dims** | **Length** | **#AnomPts** | **#AnomSeqs** |
|-----------------|----------------------------|--------:|----------:|------------|--------------:|--------------:|
| CalIt2          | Smart Building             |      1  |        3  | >5K         |             0 |            21 |
| CreditCard      | Finance / Fraud Detection  |      1  |       30  | >100K       |           219 |            10 |
| Daphnet         | Healthcare                 |     26  |       10  | >50K        |             0 |         1–16  |
| Exathlon        | Cloud Computing            |     30  |       21  | >50K        |          0–4  |         0–6   |
| GECCO           | Water Quality Monitoring   |      1  |       10  | >50K        |             0 |            37 |
| GHL             | Industrial Process         |     14  |       17  | >100K       |             0 |         1–4   |
| Genesis         | Industrial Automation      |      1  |       19  | >5K         |             0 |             2 |
| GutenTAG        | Synthetic Benchmark        |     30  |       21  | >10K        |             0 |         1–3   |
| MITDB           | Healthcare                 |     47  |        3  | >500K       |             0 |      1–720    |
| MSL             | Spacecraft Telemetry       |     26  |       56  | >5K         |             0 |         1–3   |
| OPPORTUNITY     | Human Activity Recognition |     13  |       33  | >25K        |             0 |             1 |
| Occupancy       | Smart Building             |      2  |        6  | >5K         |          1–3  |        9–13   |
| PSM             | IT Infrastructure          |      1  |       27  | >50K        |             0 |            39 |
| SMAP            | Spacecraft Telemetry       |     48  |       26  | >5K         |             0 |         1–3   |
| SMD             | IT Infrastructure          |     18  |       39  | >10K        |             0 |        4–24   |
| SVDB            | Healthcare                 |     78  |        3  | >100K       |             0 |      2–678    |
| CIC-IDS-2017    | Cybersecurity              |      5  |       73  | >100K       |      0–8656   |    0–2546     |
| Metro           | Transportation             |      1  |        6  | >10K        |            20 |             5 |
| SWAN-SF         | Industrial Process         |      1  |       39  | >50K        |          5233 |          1382 |
