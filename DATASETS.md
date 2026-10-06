# Datasets (S³ paper)

This file lists the data behind the S³ paper, where each public source can be obtained, and which
parts are team-authored. Please check each source page for its current license terms before reuse.

## Calibration set (eight use cases, public sources)

Each use case is a 30-item random sample drawn from the public source below. The samples are in
`data/gold_sets/`. Model outputs and summaries are in `data/raw_outputs/newuc*` and
`data/results/newuc*`.

| Use case | Sample file | Source dataset | Location | License shown on the source page |
|---|---|---|---|---|
| Phishing detection | `uc1_phishing_detection.csv` | Phishing Email Dataset | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset | not stated |
| Receipt extraction | `uc2_receipt_extraction.csv` | SROIE (ICDAR 2019) | https://github.com/zzzDavid/ICDAR-2019-SROIE | MIT |
| IT ticket classification | `uc3_it_ticket_classification.csv` | Classification of IT Support Tickets | https://zenodo.org/records/7648117 | CC BY 4.0 |
| Product review sentiment | `uc4_amazon_review_sentiment.csv` | Amazon Fine Food Reviews | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews | not stated |
| CVE-to-CWE classification | `uc5_cve_cwe_classification.csv` | CVE and CWE Mapping Dataset | https://www.kaggle.com/datasets/krooz0/cve-and-cwe-mapping-dataset | not stated |
| Symptom-to-disease | `uc6_symptom_disease_classification.csv` | Disease-Symptom Knowledge Database (Columbia) | http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html | not stated |
| Contract clause classification | `uc7_contract_clause_classification.csv` | CUAD | https://github.com/TheAtticusProject/cuad | not stated |
| Financial news sentiment | `uc8_financial_news_sentiment.csv` | Financial PhraseBank (Kaggle copy) | https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news | not stated |

The symptom-to-disease samples were taken from a Kaggle copy of the Columbia database:
https://www.kaggle.com/datasets/wisnuafifuddin/disease-symptom-knowledge-database-by-columbia-edu

"Not stated" means the source page did not show a license when checked; it does not mean the data is
unrestricted. The public datasets themselves are not stored in this repository, only the 30-item samples
used in the experiments.

## Validation set (the original eight use cases)

These gold sets were authored by the research team and are not drawn from a public dataset. Each has 100
items (70 train, 30 test); the 30-item test split is used for benchmarking. Table 8 of the paper lists the
public sources used as domain references.

| Use case | File |
|---|---|
| SMS threat detection | `data/gold_sets/uc1_sms_threat_detection.csv` |
| Invoice field extraction | `data/gold_sets/uc2_invoice_extraction.csv` |
| Support ticket routing | `data/gold_sets/uc3_support_ticket_routing.csv` |
| Product review sentiment | `data/gold_sets/uc4_product_review_sentiment.csv` |
| Automated code review | `data/gold_sets/uc5_code_review.csv` |
| Healthcare clinical triage | `data/gold_sets/uc6_clinical_triage.csv` |
| Legal contract risk analysis | `data/gold_sets/uc7_legal_contract_analysis.csv` |
| Financial report drafting | `data/gold_sets/uc8_financial_report_drafting.csv` |
