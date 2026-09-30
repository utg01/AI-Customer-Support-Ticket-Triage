# AI Customer Support Ticket Triage

An NLP service that predicts ticket `Type` (`Incident`, `Problem`, `Request`, or `Change`) and `Priority` (`high`, `medium`, or `low`).  
It auto-routes high-confidence Type predictions and flags low-confidence tickets for human review.

Live demo: <!-- TODO: add Streamlit Cloud URL -->

## Results at a glance

- **Type (4 classes):** macro-F1 about `0.76-0.77`, compared with a `0.14` majority baseline, on a leakage-controlled split.
- At Type confidence `>= 0.7`, the model auto-routes `47%` of tickets at `93%` accuracy; the rest go to human review.
- **Priority (3 classes)** is only weakly predictable from text: macro-F1 about `0.41` versus a `0.20` baseline. Review flags therefore use Type confidence only.
- Evaluation leakage was found and fixed; the evidence is documented below.

## How it works

**Data.** English rows from three source CSVs are combined, missing or repeated ticket text is removed, and subject plus body become one text field. Near-duplicate tickets are grouped before the final split.

**Embeddings.** Each ticket is represented with `BAAI/bge-small-en-v1.5`, producing vectors that can be used by the classifiers.

**Models.** Logistic Regression, LinearSVC, and RandomForest are compared with a dummy majority-class baseline. The final deployed models use balanced class weights where applicable.

**Split and evaluation.** Near-duplicate groups are kept on one side of a `StratifiedGroupKFold` split. Accuracy and macro-F1 are reported for both Type and Priority.

**Triage.** The API and Streamlit app embed new text, call the saved classifiers, return the predicted labels and confidences, and set `needs_review` when Type confidence is below `0.7`. Priority confidence is reported separately with `priority_reliable` set at `>= 0.6`, but it never triggers review.

## The leakage finding

On a random split, RandomForest Priority macro-F1 was `0.713`, suspiciously high. A similarity check found a median nearest-train similarity of `0.966`; `62.5%` of test rows had a train row with cosine similarity above `0.95`. Plain 1-NN label copying scored Type accuracy `0.894` and Priority accuracy `0.81` without a model.

The fix groups near-duplicates using cosine similarity `> 0.95` and connected components. This produced `13,564` groups, with a largest group of `499`, and the split uses `StratifiedGroupKFold` so each group stays on one side. After the fix, maximum test-to-train similarity was `0.95` and 1-NN Priority accuracy fell to `0.461`.

### Random split (leaky, for comparison only)

| Model | Type acc | Type macro-F1 | Priority acc | Priority macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.399 | 0.143 | 0.412 | 0.194 |
| Logistic Regression | 0.757 | 0.777 | 0.449 | 0.443 |
| LinearSVC | 0.783 | 0.784 | 0.478 | 0.457 |
| RandomForest | 0.861 | 0.853 | 0.724 | 0.713 |

`Logistic Regression`, `LinearSVC`, and `RandomForest` use `class_weight="balanced"`.

### Group-aware split (honest, used for final numbers)

| Model | Type acc | Type macro-F1 | Priority acc | Priority macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.399 | 0.143 | 0.424 | 0.198 |
| Logistic Regression | 0.745 | 0.764 | 0.416 | 0.408 |
| LinearSVC | 0.774 | 0.773 | 0.434 | 0.408 |
| RandomForest | 0.765 | 0.739 | 0.469 | 0.393 |

## Confidence threshold and human review

Type only, Logistic Regression, grouped split:

| Threshold | Auto-handled | Accuracy on auto-handled |
|---:|---:|---:|
| 0.0 | 100% | 0.745 |
| 0.5 | 95% | 0.763 |
| 0.6 | 67% | 0.841 |
| 0.7 | 47% | 0.930 |
| 0.8 | 35% | 0.987 |

**Deployed threshold:** `0.7`. Priority confidence is shown with `priority_reliable` when confidence is `>= 0.6`, but it never triggers review: only `11%` of tickets reach `0.6`, at `0.662` accuracy.

## Preprocessing experiment

Logistic Regression on the grouped split produced these macro-F1 scores:

| Preprocessing | Type | Priority |
|---|---:|---:|
| Raw text | 0.764 | 0.408 |
| Lowercase + punctuation removed | 0.765 | 0.404 |
| Plus stop-words removed | 0.746 | 0.411 |

There is no meaningful gain from preprocessing. Raw text is used because the embedding model handles casing and punctuation, while stop-word removal slightly hurts Type performance.

## Limitations

- `Problem` and `Incident` are the hardest pair. `class_weight="balanced"` raised Problem recall from `0.29` to `0.65` on the random split.
- Priority is weakly predictable from text alone.
- Evaluation uses one train/test split with about `5,000` test rows, so differences below about `0.02` are noise.
- The project supports English only.
- The dataset contains many near-duplicate tickets.
- Type labels (`Incident`, `Problem`, `Request`, `Change`) are somewhat subjective.

## Final model

Logistic Regression is deployed because it provides `predict_proba` and performs within noise of LinearSVC. The saved classifiers are `clf_type.joblib` and `clf_priority.joblib`.

## Run it

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

Start the API:

```bash
uvicorn app:app --reload
```

Send `POST /triage` with:

```json
{"text": "Subject: Feature change. Body: Could you please add an export to CSV option in the reports dashboard?"}
```

Example response:

```json
{"type": "Change", "type_confidence": 0.922, "priority": "low", "priority_confidence": 0.593, "priority_reliable": false, "needs_review": false}
```

## Dataset

The project uses the [Multilingual Customer Support Tickets dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) from Kaggle, by `tobiasbueck`. The raw CSVs are not included in this repository; English rows from the dataset files were combined for the experiment.

`scikit-learn` is pinned to `1.9.1` in `requirements.txt`; use the same environment version when loading the `.joblib` files.

## Repo layout

```text
.
|-- app.py
|-- clf_priority.joblib
|-- clf_type.joblib
|-- data_processing.ipynb
|-- LinearSVC_RandomForest.ipynb
|-- requirements.txt
|-- streamlit_app.py
`-- README.md
```

<details>
<summary>Learning notes and notebook guide</summary>

### Notebook workflow

Run `data_processing.ipynb` to load the source data, keep English rows, combine subject and body, remove incomplete or repeated text, and create embeddings. Run `LinearSVC_RandomForest.ipynb` to compare models, measure similarity, create grouped splits, evaluate the models, and save the final classifiers.

### Metric definitions

- **Accuracy:** correct predictions divided by all predictions.
- **Precision:** of the tickets predicted as a class, the share that belongs to that class.
- **Recall:** of the tickets that belong to a class, the share the model finds.
- **F1:** the harmonic mean of precision and recall.
- **Macro-F1:** F1 calculated per class and averaged equally, so smaller classes matter as much as larger ones.

### Reading a confusion matrix

Rows are actual labels and columns are predicted labels. The diagonal contains correct predictions. A class's recall is its diagonal value divided by the total of its actual-label row; its precision is the diagonal value divided by the total of its predicted-label column.

### Important code lessons

- Keep embeddings and labels in the same row order.
- Use `index=False` when saving tabular outputs unless the index is intentional.
- Group near-duplicates before evaluation when similar records can cross a random split.

</details>
