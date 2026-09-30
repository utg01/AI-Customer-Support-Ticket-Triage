# AI Customer Support Ticket Triage

This project uses natural language processing and machine learning to classify customer-support tickets.

For each ticket, the system predicts:

- `type`: `Change`, `Incident`, `Problem`, or `Request`
- `priority`: `high`, `medium`, or `low`

The project also includes a Streamlit app that accepts one ticket or a CSV file and returns predictions, confidence scores, and a human-review flag.

## Project Flow

The complete flow is:

```text
Three CSV datasets
        |
        v
Keep English tickets
        |
        v
Combine the datasets
        |
        v
Remove missing and repeated ticket text
        |
        v
Split data into training and test sets
        |
        v
Convert text into numerical embeddings
        |
        v
Train classification models
        |
        v
Evaluate accuracy, precision, recall, F1, and confusion matrices
        |
        v
Save final models and use them in the Streamlit app
```

In simple Hinglish: pehle datasets ko clean aur combine kiya, phir ticket text ko numbers mein convert kiya, model ko examples se train kiya, aur end mein naye tickets ka type aur priority predict ki.

## Repository Structure

```text
.
|-- datasets/
|   |-- dataset-tickets-multi-lang-4-20k.csv
|   |-- dataset-tickets-multi-lang3-4k.csv
|   `-- aa_dataset-tickets-multi-lang-5-2-50-version.csv
|-- data_processing.ipynb
|-- LinearSVC_RandomForest.ipynb
|-- streamlit_app.py
|-- app.py
|-- clf_type.joblib
|-- clf_priority.joblib
|-- requirements.txt
|-- create_ml_guide_pdf.py
`-- NLP_Machine_Learning_Guide.pdf
```

The CSV datasets, NumPy embedding files, logs, and generated PDF are ignored by Git where appropriate. The two `.joblib` model files are kept because the Streamlit app needs them to make predictions without retraining.

## What We Did in `data_processing.ipynb`

### 1. Loaded the datasets

The notebook reads the three CSV files from the `datasets` folder:

```python
df1 = pd.read_csv("datasets/dataset-tickets-multi-lang-4-20k.csv")
df2 = pd.read_csv("datasets/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df3 = pd.read_csv("datasets/dataset-tickets-multi-lang3-4k.csv")
```

Each CSV is loaded into a pandas DataFrame. A DataFrame is a table in Python, similar to a spreadsheet.

### 2. Kept English rows

The original datasets contain multiple languages. The project keeps English tickets because the chosen model and experiments were designed for English text:

```python
df1 = df1[df1["language"] == "en"]
```

The same filtering idea was applied to the other datasets.

### 3. Combined the datasets

The three DataFrames were placed one below the other:

```python
fdf = pd.concat([df1, df2, df3], ignore_index=True)
```

`ignore_index=True` creates a fresh row number for the combined DataFrame.

### 4. Checked data quality

The notebook checked:

- Duplicate ticket bodies
- Missing ticket bodies
- Whether identical text had different `type` values
- Whether identical text had different `priority` values
- Class distributions for `type` and `priority`

These checks matter because duplicates can make a model appear stronger than it really is. If almost the same ticket appears in both training and testing data, the model may look as if it learned a general rule when it actually recognized a very similar example.

### 5. Created one text column

The subject and body were combined:

```python
fdf["text"] = fdf["subject"].fillna("") + " " + fdf["body"]
```

Missing subjects are replaced with an empty string. This gives the model one complete text field.

### 6. Removed incomplete and repeated examples

```python
fdf = fdf.dropna(subset=["body"]).copy()
fdf = fdf.drop_duplicates(subset="text").reset_index(drop=True)
```

Rows without a body were removed. Repeated complete text was reduced to one row.

The final machine-learning table contains:

```python
df = fdf[["text", "type", "priority"]].copy()
```

The other dataset columns are not needed for this experiment.

### 7. Created the initial train/test split

```python
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["type"],
    random_state=42
)
```

The result was:

- Training rows: `20,109`
- Test rows: `5,028`
- Total rows: `25,137`

`stratify=df["type"]` tries to keep the same ticket-type proportions in both sets. `random_state=42` makes the split reproducible.

### 8. Created text embeddings

The project used:

```python
SentenceTransformer("BAAI/bge-small-en-v1.5")
```

An embedding is a list of numbers representing the meaning of a piece of text. Texts with similar meaning tend to have similar vectors.

The embeddings were saved as NumPy files so they would not need to be generated again every time:

- `X_train.npy`
- `X_test.npy`
- `X_train_text_basic.npy`
- `X_test_text_basic.npy`
- `X_train_text_stop.npy`
- `X_test_text_stop.npy`

## What We Did in `LinearSVC_RandomForest.ipynb`

### 1. Reloaded the prepared data and embeddings

The second notebook loads the three datasets, rebuilds the cleaned text table, and loads the saved embedding arrays.

It checks that the number of embedding rows matches the number of DataFrame rows. This is important because the labels must stay in exactly the same order as the embeddings.

### 2. Compared models

The notebook compared:

- Dummy baseline
- Logistic Regression
- LinearSVC
- RandomForest

Each model was tested on both targets:

- Ticket `type`
- Ticket `priority`

`class_weight="balanced"` was used for several models so that less common classes receive more attention.

### 3. Initial random-split results

The first random split produced these saved results:

| Model | Target | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Baseline | Type | 0.399 | 0.143 |
| Baseline | Priority | 0.424 | 0.198 |
| Logistic Regression | Type | 0.745 | 0.764 |
| Logistic Regression | Priority | 0.416 | 0.408 |
| LinearSVC | Type | 0.774 | 0.773 |
| LinearSVC | Priority | 0.434 | 0.408 |
| RandomForest | Type | 0.765 | 0.739 |
| RandomForest | Priority | 0.469 | 0.393 |

A separate earlier run on the same general random-split setup reported:

- RandomForest type accuracy: `0.861`
- RandomForest type macro-F1: `0.853`
- RandomForest priority accuracy: `0.724`
- RandomForest priority macro-F1: `0.713`

The exact score depends on which split and model run is being viewed, but the important observation is the same: the random split looked much stronger than the stricter grouped evaluation.

## Why the Initial Result Was Suspicious

The initial split randomly divided rows into training and test sets. This is often acceptable when every row is independent. Here, however, many tickets were extremely similar.

The notebook calculated embedding similarities between test tickets and training tickets. The saved output showed:

```text
similarity percentiles (5,25,50,75,95): [0.857 0.929 0.966 0.985 0.996]
test rows with sim > 0.95: 0.625
```

This means:

- The median maximum similarity was about `0.966`.
- The 95th percentile was about `0.996`.
- About `62.5%` of test rows had a training row with similarity above `0.95`.

So the guess that the similarity was around `0.96` or `0.97` is confirmed. The notebook contains both `0.966` as the median percentile and the `0.95` threshold analysis.

In simple words: the model was often tested on tickets that looked very similar to tickets it had already seen. Is model ne kuch patterns rat liye the, especially repeated or near-repeated ticket wording. This can make random-split accuracy look artificially high. This is a form of evaluation leakage or over-optimistic evaluation, even if the labels themselves were not directly copied.

## Grouping Similar Embeddings

To reduce this problem, the notebook grouped highly similar embeddings.

The `make_groups()` function:

1. Compares embeddings in chunks so the full similarity matrix does not need to be held in one huge operation.
2. Finds pairs whose similarity is above a threshold.
3. Builds a graph where similar tickets are connected.
4. Uses connected components to turn connected tickets into groups.

At a similarity threshold of `0.95`, the saved output was:

```text
groups: 13564 | largest group: 499 | singletons: 7766
```

At thresholds of `0.95` and `0.93`, the output was:

```text
0.95 | groups: 13564 | largest: 499 | singletons: 7766
0.93 | groups: 9975  | largest: 1042 | singletons: 5063
```

A lower threshold creates larger groups because more tickets are considered similar.

### Why grouping helps

Without grouping, one group of near-duplicate tickets can be split like this:

```text
same ticket family -> training set and test set
```

That gives the model a very familiar example during testing.

With grouping, the goal is:

```text
same ticket family -> only one side of the split
```

This tests whether the model can handle a new ticket family rather than simply recognizing a close copy.

The notebook then used:

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

`StratifiedGroupKFold` tries to preserve class proportions while keeping each similarity group on one side of the split.

## Group-Aware Results

The grouped split produced:

- Training rows: `20,110`
- Test rows: `5,027`
- Test type proportions:
  - Incident: `0.399`
  - Request: `0.289`
  - Problem: `0.207`
  - Change: `0.106`
- Test priority proportions:
  - medium: `0.424`
  - high: `0.382`
  - low: `0.194`

The nearest-neighbour results also dropped:

- Type accuracy: `0.739`
- Priority accuracy: `0.461`

This is evidence that the original random split was easier because of similar examples crossing the split.

The grouped model comparison was:

| Model | Type Accuracy | Type Macro-F1 | Priority Accuracy | Priority Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.399 | 0.143 | 0.424 | 0.198 |
| Logistic Regression | 0.745 | 0.764 | 0.416 | 0.408 |
| LinearSVC | 0.774 | 0.773 | 0.434 | 0.408 |
| RandomForest | 0.765 | 0.739 | 0.469 | 0.393 |

The RandomForest priority result on the grouped split is only `0.469`, compared with the earlier roughly `0.724` priority accuracy from the easier random split. That large drop is why the earlier result should be treated as over-optimistic rather than as proof that the model generalizes well.

For ticket type, LinearSVC was strongest on the grouped split. For priority, all models were much weaker, which suggests that priority may require information not present in the ticket text alone or may have noisy labels.

## Evaluation Metrics

### Accuracy

Accuracy is:

```text
correct predictions / all predictions
```

It is easy to understand, but it can hide poor performance on smaller classes.

### Precision

Precision answers:

> When the model predicts a class, how often is it correct?

For one class:

```text
precision = true positives / (true positives + false positives)
```

Hinglish: model ne jis class ka naam liya, un predictions mein kitne actually sahi the?

### Recall

Recall answers:

> Of all the real examples of a class, how many did the model find?

```text
recall = true positives / (true positives + false negatives)
```

Hinglish: jo tickets actually is class ke the, unmein se model ne kitne pakde?

### F1 score

F1 combines precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

A model cannot receive a high F1 if either precision or recall is very low.

### Macro-F1

Macro-F1 calculates F1 separately for every class, then averages the class scores equally.

This is useful here because a model might perform very well on `Request` but poorly on `Problem`. Macro-F1 prevents the large or easy class from hiding the weak class.

Weighted-F1 gives more weight to classes with more examples. It can therefore look better even when a smaller class performs poorly.

## Confusion Matrices

A confusion matrix compares actual labels with predicted labels.

- Rows represent actual labels.
- Columns represent predicted labels.
- The diagonal contains correct predictions.
- Off-diagonal cells contain mistakes.

For example, if a real `Problem` ticket appears in the `Incident` column, the model predicted `Incident` for that Problem ticket.

The first notebook plotted a confusion matrix for ticket type. The PDF guide generated from this project also includes matrices for the model comparisons. A matrix helps explain *which* classes are being confused instead of only showing one overall score.

To read a class row:

```text
class recall = correct diagonal cell / total actual examples in that row
```

To read a class column:

```text
class precision = correct diagonal cell / total predictions in that column
```

## Final Model and Triage Function

The second notebook saves final Logistic Regression models:

- `clf_type.joblib`
- `clf_priority.joblib`

They are loaded by the Streamlit app.

The `triage()` function:

1. Receives ticket text.
2. Creates embeddings using the same SentenceTransformer model.
3. Gets class probabilities from both saved classifiers.
4. Chooses the class with the largest probability.
5. Stores type and priority confidence.
6. Marks low type confidence as `needs_review`.
7. Shows whether priority confidence is reliable.

The current threshold settings are:

```python
THRESHOLDS = {
    "type": 0.7,
    "priority": 0.0
}
```

A type confidence below `0.7` triggers review. Priority threshold `0.0` means priority confidence by itself does not trigger the review flag. The app still displays `priority_reliable` using a `0.6` threshold.

In the 100-ticket test sample:

- 100 tickets were checked.
- 58 tickets were flagged for review.
- The output displayed predicted labels, actual labels, confidence values, and `needs_review`.

Confidence is not a guarantee. A model can be confidently wrong, so the review queue is important.

## Running Locally

### 1. Open the project folder

```powershell
cd "C:\Users\utg18\OneDrive\Desktop\NLP Project"
```

### 2. Create a virtual environment if needed

```powershell
python -m venv venv
```

### 3. Activate the environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If Windows runs Streamlit from a different installation, use the project interpreter explicitly:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

This is safer than calling `streamlit run` directly because it guarantees that Streamlit and `sentence_transformers` come from the same environment.

### 5. Start the app

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

### 6. Use the app

The app has four areas:

1. **Single ticket**: paste a ticket and analyze it.
2. **Batch CSV**: upload a CSV, choose its text column, and process up to 500 rows.
3. **Review queue**: inspect tickets marked for human review during the current session.
4. **Model report**: view saved evaluation results and threshold behavior.

## Running the Notebooks

Open the notebooks in VS Code or Jupyter using the project virtual environment.

Run `data_processing.ipynb` first when rebuilding the data pipeline. It creates the cleaned data and embeddings.

Run `LinearSVC_RandomForest.ipynb` after that to:

- Compare classifiers.
- Check embedding similarity.
- Create grouped splits.
- Evaluate models on the stricter split.
- Save final classifiers.
- Test triage confidence behavior.

The embedding step can take time because the language model processes every ticket. Reusing the saved `.npy` files avoids repeating that work unnecessarily.

## Important Code Lessons

### Dataset paths

All three dataset paths use the `datasets/` folder. If the notebook is run from another working directory, use absolute paths or change the working directory first.

### Saving filtered data

When filtering a DataFrame, save the filtered variable, not the original variable:

```python
df1 = df1[df1["language"] == "en"]
df1.to_csv("datasets/dataset-tickets-multi-lang-4-20k.csv", index=False)
```

### Avoiding index columns

Always use `index=False` when saving a cleaned DataFrame unless the pandas index is intentionally part of the data:

```python
df.to_csv("output.csv", index=False)
```

### Keeping embeddings and labels aligned

The row order of `X_train` must match the row order of `train_df`. If rows are shuffled or filtered after embeddings are created, the labels can be paired with the wrong vectors.

## Main Conclusion

The project successfully builds an end-to-end ticket triage pipeline, but the evaluation shows an important limitation.

The initial random split looked strong, especially for RandomForest. The earlier result of roughly 72% priority accuracy and 86% type accuracy was over-optimistic because many test embeddings were extremely similar to training embeddings. The notebook confirmed a median similarity of about `0.966`, with `62.5%` of test rows having similarity above `0.95`.

After grouping similar embeddings and keeping groups separate between training and testing, performance became more realistic. Type prediction remained useful, especially with LinearSVC, but priority prediction was much harder.

The correct lesson is not that the model learned nothing. The model learned useful patterns, especially for ticket type. The lesson is that evaluation must reflect the real use case. If production tickets are new variations of known ticket families, grouped evaluation is a more honest test than a purely random row split.

In Hinglish: initial split mein model ne similar examples dekh kar achha score diya, lekin grouped split ne test kiya ki model genuinely new ticket families par kaisa perform karta hai. Isliye grouped result ko zyada realistic maanna chahiye.
