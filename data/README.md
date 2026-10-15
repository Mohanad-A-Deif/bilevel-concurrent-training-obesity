# NHANES 2013-2014 input data

`raw/` contains CSV exports of three NHANES 2013-2014 public-use components, which were used to produce the reported results:

| File | NHANES component | Variables used |
|---|---|---|
| `demographic.csv` | Demographics (DEMO_H) | SEQN, RIAGENDR, RIDAGEYR, SDDSRVYR |
| `examination.csv` | Body Measures (BMX_H) | SEQN, BMXHT, BMXWT, BMXBMI, BMXWAIST |
| `questionnaire.csv` | Physical Activity (PAQ_H) | SEQN, PAQ605-PAD675 (GPAQ domains) |

NHANES data are released by the U.S. National Center for Health Statistics (NCHS) as public-use files: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013

The official transport files can be used instead: place `DEMO_H.XPT`, `BMX_H.XPT`, and `PAQ_H.XPT` (from the URL above) in `raw/`, and `src/cohort.py` will read them in preference to the CSV files. The script checks that the cohort flow equals the published flow (10,175 screened; 5,769 aged 20 years or older; 2,083 with BMI of at least 30 kg/m2; 1,963 complete cases) and warns otherwise.

Only the variables listed above are read. NHANES survey weights are not used.
