import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

OUT = "NLP_Machine_Learning_Guide.pdf"

# Read the same three source files used by the notebooks.
df1 = pd.read_csv("datasets/dataset-tickets-multi-lang-4-20k.csv")
df2 = pd.read_csv("datasets/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df3 = pd.read_csv("datasets/dataset-tickets-multi-lang3-4k.csv")
fdf = pd.concat([df1, df2, df3], ignore_index=True)
fdf = fdf.dropna(subset=["body"]).copy()
fdf["text"] = fdf["subject"].fillna("") + " " + fdf["body"]
fdf = fdf.drop_duplicates(subset="text").reset_index(drop=True)
df = fdf[["text", "type", "priority"]].copy()

# Recreate the first notebook split so matrices match the saved embeddings.
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["type"], random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

x_train = np.load("X_train.npy")
x_test = np.load("X_test.npy")

# Train the two models from the second notebook for matrix generation.
models = {
    "LinearSVC": {
        "type": LinearSVC(class_weight="balanced", max_iter=5000),
        "priority": LinearSVC(class_weight="balanced", max_iter=5000),
    },
    "RandomForest": {
        "type": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=2, random_state=42),
        "priority": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=2, random_state=42),
    },
}
predictions = {}
for model_name, targets in models.items():
    predictions[model_name] = {}
    for target, classifier in targets.items():
        classifier.fit(x_train, train_df[target])
        predictions[model_name][target] = classifier.predict(x_test)

metrics = [
    ("Baseline", "type", 0.399, 0.143),
    ("Baseline", "priority", 0.424, 0.198),
    ("Logistic Regression", "type", 0.745, 0.764),
    ("Logistic Regression", "priority", 0.416, 0.408),
    ("LinearSVC", "type", 0.774, 0.773),
    ("LinearSVC", "priority", 0.434, 0.408),
    ("RandomForest", "type", 0.765, 0.739),
    ("RandomForest", "priority", 0.469, 0.393),
]

BLUE = "#174A68"
TEAL = "#138A8A"
ORANGE = "#E07A3F"
DARK = "#1F2933"
LIGHT = "#F2F6F8"


def setup_page(title, subtitle=None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=22, fontweight="bold", color=BLUE, va="top")
    if subtitle:
        fig.text(0.08, 0.905, subtitle, fontsize=10.5, color=TEAL, va="top")
    fig.text(0.08, 0.035, "NLP ticket classification guide | generated from the project notebooks", fontsize=7.5, color="#667085")
    return fig


def add_paragraphs(fig, paragraphs, y=0.86, width=76, size=10.5, leading=0.026):
    for paragraph in paragraphs:
        lines = []
        for line in paragraph.split("\n"):
            lines.extend(textwrap.wrap(line, width=width) or [""])
        block = "\n".join(lines)
        fig.text(0.09, y, block, fontsize=size, color=DARK, va="top", linespacing=1.35)
        y -= leading * (len(lines) + 1.2)
    return y


def add_bullets(fig, bullets, y, width=76, size=10.2, leading=0.025):
    for bullet in bullets:
        lines = textwrap.wrap(bullet, width=width) or [""]
        text = "\n".join(["- " + lines[0]] + ["  " + line for line in lines[1:]])
        fig.text(0.10, y, text, fontsize=size, color=DARK, va="top", linespacing=1.3)
        y -= leading * (len(lines) + 0.75)
    return y


def add_callout(fig, text, y, size=10.5, color=ORANGE, style="normal", width=76):
    wrapped = textwrap.fill(text, width=width)
    fig.text(0.10, y, wrapped, fontsize=size, color=color, style=style, va="top", linespacing=1.3)


def table_page(title, subtitle, headers, rows, col_widths=None, note=None):
    fig = setup_page(title, subtitle)
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.65])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F7FAFC" if row % 2 else "white")
            cell.get_text().set_color(DARK)
    if note:
        fig.text(0.09, 0.12, note, fontsize=9, color="#475467", va="top")
    return fig


with PdfPages(OUT) as pdf:
    fig = setup_page("NLP Ticket Classification", "A beginner-friendly guide to the two project notebooks")
    fig.text(0.09, 0.77, "What this project does", fontsize=17, color=TEAL, fontweight="bold")
    add_paragraphs(fig, [
        "This project reads customer-support tickets, combines three CSV datasets, removes unusable or repeated text, converts each ticket into a numerical embedding, and trains machine-learning models.",
        "The models predict two things: the ticket type (Change, Incident, Problem, or Request) and the ticket priority (high, medium, or low). The final notebook also tests confidence and marks uncertain predictions for human review.",
        "Hinglish summary: pehle data ko saaf aur combine karte hain, phir text ko numbers mein convert karte hain, model ko examples se train karte hain, aur last mein naye ticket ka type aur priority predict karte hain.",
    ], y=0.71, size=12, leading=0.034)
    fig.text(0.09, 0.30, "The journey", fontsize=16, color=TEAL, fontweight="bold")
    add_bullets(fig, [
        "data_processing.ipynb: prepare and inspect the dataset, create embeddings, compare an initial Logistic Regression model, and study duplicate-like tickets.",
        "LinearSVC_RandomForest.ipynb: compare models, make a stricter group-aware split, save final classifiers, and run an uncertainty-aware triage function.",
        "Important idea: a model can be accurate on familiar or highly similar tickets but weaker on genuinely new ticket patterns. The notebooks investigate this difference.",
    ], 0.25, size=11)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("1. The project in plain English", "The complete flow from CSV files to a prediction")
    y = add_paragraphs(fig, [
        "Step 1 - Load: pandas reads each CSV into a DataFrame. A DataFrame is like a spreadsheet in Python: rows are tickets and columns are ticket properties.",
        "Step 2 - Combine: pd.concat stacks the three DataFrames vertically. Each ticket becomes one row in fdf.",
        "Step 3 - Clean: rows without a body are removed. Subject and body are joined into one text field. Exact repeated text is removed.",
        "Step 4 - Split: the cleaned data is divided into training data and test data. The model learns from training rows; the test rows are kept aside for an honest check.",
        "Step 5 - Embed: SentenceTransformer converts each ticket into a vector of numbers. Similar meanings should produce nearby vectors.",
        "Step 6 - Learn: classifiers learn a boundary between labels using those vectors.",
        "Step 7 - Evaluate: accuracy, precision, recall, F1, classification reports, and confusion matrices show what the model gets right and wrong.",
        "Step 8 - Use: triage() predicts labels for new text, estimates confidence, and logs rows needing review.",
    ], y=0.85, width=76, size=10.6, leading=0.023)
    add_callout(fig, "Simple analogy: embeddings are a map of meaning; the classifier learns areas on that map.", max(y - 0.01, 0.10), size=11)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("2. Notebook 1: data_processing.ipynb", "Preparing the ticket data")
    y = add_paragraphs(fig, [
        "The notebook starts with import pandas as pd and import numpy as np. pandas handles tables; NumPy handles numerical arrays. Later, scikit-learn and sentence-transformers provide splitting, metrics, and text embeddings.",
        "The first dataset is loaded into df. Then df1, df2, and df3 represent the three source datasets. The code filters language == 'en' so the later model works on English tickets.",
        "The three English DataFrames are combined with fdf = pd.concat([df1, df2, df3], ignore_index=True). ignore_index=True creates a fresh row numbering instead of retaining three old indexes.",
        "The notebook checks duplicate bodies, missing bodies, and whether the same body has different labels. These checks matter because repeated or conflicting tickets can make evaluation look better or confuse the model.",
        "The final preparation creates text = subject + body, drops repeated text, and keeps only text, type, and priority. The comment about the other columns means they are not needed for this experiment.",
    ], y=0.85, width=76, size=10.5, leading=0.024)
    add_callout(fig, "Hinglish: yahan hum raw spreadsheet ko model ke liye simple table mein convert kar rahe hain - ek text column aur do answers/labels.", 0.20, style="italic")
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("Notebook 1: split, baseline, and embeddings", "Why each stage is needed")
    add_paragraphs(fig, [
        "train_test_split(..., test_size=0.2, stratify=df['type'], random_state=42) keeps 80 percent for training and 20 percent for testing. stratify tries to preserve the same type proportions in both parts. random_state=42 makes the split repeatable.",
        "The DummyClassifier is a baseline. It does not understand text; it mostly predicts the most common label. If a real model cannot beat this baseline, the real model is not useful.",
        "SentenceTransformer('BAAI/bge-small-en-v1.5') loads a pretrained language model. encode() turns each ticket into a vector. normalize_embeddings=True makes vector lengths comparable and makes dot products behave like cosine similarity.",
        "The vectors are saved as X_train.npy and X_test.npy. Saving avoids repeating the expensive embedding step every time the notebook is reopened.",
        "Logistic Regression is first used as a simple classifier on top of embeddings. The notebook prints accuracy, macro-F1, and a classification report, then plots a confusion matrix for ticket type.",
    ], y=0.85, width=76, size=10.5, leading=0.024)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("3. Notebook 2: LinearSVC_RandomForest.ipynb", "Model comparison and production-style testing")
    add_paragraphs(fig, [
        "This notebook reloads the three CSVs and reconstructs the cleaned text table. It loads the saved embeddings and checks that their number of rows matches train_df and test_df.",
        "The models dictionary contains LinearSVC and RandomForest. A loop trains each model for both targets: type and priority. clone(m) gives a fresh model for each experiment.",
        "LinearSVC finds separating boundaries in the embedding space. It is often strong for text classification and is relatively fast. Its decision scores are not probabilities by default.",
        "RandomForest builds many decision trees and combines their votes. n_estimators=200 means 200 trees. class_weight='balanced' gives more influence to less common classes.",
        "The later part checks near-duplicate tickets using embedding similarity. It creates connected groups of very similar rows and then uses StratifiedGroupKFold so one similarity group does not leak into both train and test.",
        "Finally, the notebook trains final Logistic Regression models, saves them as clf_type.joblib and clf_priority.joblib, and defines triage() for future ticket text.",
    ], y=0.85, width=76, size=10.5, leading=0.023)
    add_callout(fig, "Why the second split matters: a random split can put near-copies in both train and test. That makes the score optimistic. Group-aware splitting is a harder and more realistic test.", 0.14, size=10, style="italic")
    pdf.savefig(fig); plt.close(fig)

    metric_rows = [[m, t, f"{a:.3f}", f"{f:.3f}"] for m, t, a, f in metrics]
    fig = table_page("4. Results from the saved notebook output", "Accuracy and macro-F1; higher is better", ["Model", "Target", "Accuracy", "Macro-F1"], metric_rows, [0.35, 0.25, 0.18, 0.18], "These values are the outputs saved in LinearSVC_RandomForest.ipynb. The group-aware split is reported separately in the next interpretation.")
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("How to read the results", "The important lesson is not only the biggest number")
    add_paragraphs(fig, [
        "On the initial random split, RandomForest is strongest in the saved comparison: type accuracy 0.861 and macro-F1 0.853; priority accuracy 0.724 and macro-F1 0.713.",
        "On the stricter group-aware split, scores drop: LinearSVC type accuracy is 0.774 and macro-F1 is 0.773; RandomForest type accuracy is 0.765 and macro-F1 is 0.739. Priority is much harder, with the best saved priority accuracy 0.469 and macro-F1 0.408.",
        "This drop is meaningful. The similarity experiment found that 62.5 percent of the original test rows had a training row with similarity above 0.95. A nearest-neighbour type accuracy of 0.894 on that split suggests that many tickets are very close to known examples.",
        "Hinglish: random split mein model ko milte-julte examples train aur test dono mein mil jaate hain. Group split mein un copies ko alag rakha gaya, isliye score zyada realistic aur lower hai.",
    ], y=0.85, width=76, size=10.8, leading=0.029)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("5. Evaluation terms: precision, recall, and F1", "The three words students most often mix up")
    add_paragraphs(fig, [
        "For one class, such as Incident, imagine the model says 'Incident' or 'not Incident'. True positive (TP) means it predicted Incident and it really was Incident. False positive (FP) means it predicted Incident but the real label was something else. False negative (FN) means it missed a real Incident.",
        "Precision = TP / (TP + FP). Of the tickets the model called Incident, how many were actually Incident? High precision means few false alarms.",
        "Recall = TP / (TP + FN). Of all real Incident tickets, how many did the model find? High recall means few missed cases.",
        "F1 = 2 * precision * recall / (precision + recall). F1 balances precision and recall. It becomes low when either one is low.",
        "Example: if the model labels 100 tickets as Incident and 80 are truly Incident, precision is 0.80. If there were 100 real Incident tickets and it found 80, recall is 0.80, so F1 is also 0.80.",
        "Hinglish: precision ka matlab 'jo maine positive bola, usme kitne sahi the'; recall ka matlab 'jo actually positive the, unme se kitne pakde'.",
    ], y=0.85, width=76, size=10.6, leading=0.026)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("Macro-F1 and weighted-F1", "Why the report shows more than one average")
    add_paragraphs(fig, [
        "A multi-class problem has one precision, recall, and F1 for every label. The classification report then averages them.",
        "Macro-F1 calculates the F1 for each class first, then gives every class equal weight. If Problem is difficult but Request is easy, macro-F1 still makes Problem visible. This is useful when minority classes matter.",
        "Weighted-F1 also calculates each class F1, but weights each class by its number of examples. A large class can dominate the score. Weighted-F1 can look good even when a small class performs poorly.",
        "Accuracy is the fraction of all predictions that are correct. It is easy to understand, but it can hide weak minority-class performance. That is why this project reports macro-F1 as well.",
        "In the saved results, RandomForest type macro-F1 is 0.853 on the initial split, but the class-level report shows Problem F1 is 0.68 while Request F1 is 0.96. The average alone is not enough; inspect each class.",
    ], y=0.85, width=76, size=10.7, leading=0.029)
    pdf.savefig(fig); plt.close(fig)

    # Confusion matrices for both models and both targets.
    matrix_specs = [
        ("LinearSVC: ticket type", "LinearSVC", "type", ["Change", "Incident", "Problem", "Request"]),
        ("LinearSVC: priority", "LinearSVC", "priority", ["high", "low", "medium"]),
        ("RandomForest: ticket type", "RandomForest", "type", ["Change", "Incident", "Problem", "Request"]),
        ("RandomForest: priority", "RandomForest", "priority", ["high", "low", "medium"]),
    ]
    for title, model_name, target, labels in matrix_specs:
        actual = test_df[target]
        predicted = predictions[model_name][target]
        cm = confusion_matrix(actual, predicted, labels=labels)
        fig = setup_page("6. Confusion matrix - " + title, "Rows are actual labels; columns are predicted labels")
        ax = fig.add_axes([0.18, 0.28, 0.64, 0.52])
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Correct predictions are on the diagonal", fontsize=12, color=DARK, pad=12)
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > cm.max() * 0.55 else DARK
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.text(0.10, 0.18, "Read one row at a time. For example, the Problem row shows how many real Problem tickets were predicted as Change, Incident, Problem, or Request. Off-diagonal numbers are mistakes.", fontsize=10, color=DARK, va="top")
        pdf.savefig(fig); plt.close(fig)

    fig = setup_page("How to interpret a confusion matrix", "A practical reading method")
    add_paragraphs(fig, [
        "The diagonal cells are correct predictions. A large diagonal value is good. Values outside the diagonal are errors.",
        "For a particular class, add its row to get the number of actual examples. The diagonal cell divided by that row total is recall for that class.",
        "For a particular class, add its column to get the number of predictions made for that class. The diagonal cell divided by that column total is precision for that class.",
        "If Problem is often predicted as Incident, the two classes may have similar language or the training data may not define them clearly. If low priority is missed and predicted as medium, the model may be conservative about calling tickets low.",
        "The matrices in this PDF are generated from the saved X_train.npy and X_test.npy embeddings and the same model settings in the second notebook. Small differences can occur if the CSVs or embeddings are regenerated.",
    ], y=0.85, width=76, size=10.7, leading=0.029)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("7. The triage() function", "How a new ticket becomes a prediction and a review decision")
    add_paragraphs(fig, [
        "triage(texts) receives a list of ticket strings. model.encode() creates embeddings in the same way used during training. The function creates an output DataFrame with the original text.",
        "For each target, final[target].predict_proba() returns class probabilities. The largest probability becomes the predicted label, and the largest probability becomes the confidence value.",
        "The flag starts as False. If type confidence is below 0.7, the row is marked for review. In the notebook, priority threshold is 0.0, so priority confidence alone never triggers review. priority_reliable is a separate flag requiring priority confidence >= 0.6.",
        "Rows marked needs_review are appended to low_confidence_log.csv. mode='a' means append; header is written only when the file does not already exist.",
        "In the 100-row sample, 58 rows were flagged. The output has predicted type, actual type, predicted priority, actual priority, confidence values, and needs_review. That is a useful manual audit sample.",
    ], y=0.85, width=76, size=10.6, leading=0.026)
    add_callout(fig, "Important: probabilities are model confidence estimates, not a guarantee of correctness. A high confidence can still be wrong.", 0.15)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("8. Code issues and careful improvements", "Things a student should notice in the current notebooks")
    add_bullets(fig, [
        "In data_processing.ipynb, df1 is filtered into English rows but the next line saves df, not df1. If the intention is to overwrite that file with English rows, use df1.to_csv(..., index=False).",
        "In the df3 save line, index=False is missing. That can create an extra unnamed index column when the CSV is read again.",
        "The first notebook initially uses a simple random split, while the second notebook later uses a group-aware split. Always report which split produced a metric.",
        "The first notebook plots a confusion matrix for ticket type, but does not plot one for priority or save the matrix values. The PDF adds both targets for LinearSVC and RandomForest.",
        "The second notebook fits final Logistic Regression models after comparing models, even though RandomForest performed best on the initial saved comparison. This can be intentional for probability-based triage, because LogisticRegression provides predict_proba; LinearSVC does not provide probabilities directly.",
        "When using the actual labels in the 100-row audit, keep sample_rows and sample text together. The current code does this correctly by sampling rows first and then adding actual_type and actual_priority in the same order.",
    ], 0.85, width=88, size=10.3, leading=0.028)
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("9. A student checklist", "A repeatable way to understand and run this project")
    add_bullets(fig, [
        "Run data preparation and verify row counts, missing values, language values, duplicates, and class distributions.",
        "Create the train/test split before fitting any model. Do not let test labels influence training.",
        "Create embeddings once, save them, and confirm their row order still matches the DataFrames.",
        "Compare against a dummy baseline. Then inspect accuracy, macro-F1, each class report, and confusion matrices.",
        "Check whether near-duplicate tickets cross the split. If they do, use grouped splitting for a stricter estimate.",
        "Choose the final model based on the real goal. For type, the best score may matter; for priority, reliable confidence and human review may matter more.",
        "Save the trained model and test triage on known examples and genuinely new examples. Keep a review log and inspect mistakes.",
    ], 0.85, width=88, size=10.6, leading=0.031)
    fig.text(0.10, 0.16, "One-line summary: clean data -> split honestly -> embed text -> train -> evaluate deeply -> review uncertain predictions.", fontsize=12, color=TEAL, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    fig = setup_page("Glossary", "Short definitions for the main terms")
    glossary = [
        "DataFrame: a table in pandas.",
        "Label/target: the answer the model must predict, such as type or priority.",
        "Embedding: a numeric representation of text meaning.",
        "Training set: examples used to learn.",
        "Test set: held-back examples used to evaluate.",
        "Classification: choosing one label from several categories.",
        "Accuracy: all correct predictions divided by all predictions.",
        "Precision: how trustworthy positive predictions are.",
        "Recall: how many real examples of a class were found.",
        "F1: a balance of precision and recall.",
        "Macro-F1: average F1 where every class counts equally.",
        "Confusion matrix: a table of actual labels versus predicted labels.",
        "Data leakage: test information accidentally helping training.",
        "Confidence: the model's probability-like estimate for its chosen class.",
        "Triage: automatically route a ticket and send uncertain cases to a person.",
    ]
    add_bullets(fig, glossary, 0.85, width=88, size=10.8, leading=0.032)
    pdf.savefig(fig); plt.close(fig)

print(f"Created {OUT}")
print(f"Rows used: {len(df)}; train: {len(train_df)}; test: {len(test_df)}")
