import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

TYPE_THRESHOLD = 0.7         # isse neeche Type confidence hua to human review
PRIORITY_RELIABLE_AT = 0.6   # isse upar priority pe zyada bharosa
LOG_FILE = 'api_review_log.csv'

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
embedder.max_seq_length = 256          # encoding wali same setting
clf_type = joblib.load('clf_type.joblib')
clf_priority = joblib.load('clf_priority.joblib')

app = FastAPI(title='Ticket Triage API')


class Ticket(BaseModel):
    text: str


@app.get('/')
def health():
    return {'status': 'ok'}


@app.post('/triage')
def triage(ticket: Ticket):
    emb = embedder.encode([ticket.text], normalize_embeddings=True)

    type_proba = clf_type.predict_proba(emb)[0]
    prio_proba = clf_priority.predict_proba(emb)[0]

    result = {
        'type': str(clf_type.classes_[type_proba.argmax()]),
        'type_confidence': round(float(type_proba.max()), 3),
        'priority': str(clf_priority.classes_[prio_proba.argmax()]),
        'priority_confidence': round(float(prio_proba.max()), 3),
    }
    result['priority_reliable'] = result['priority_confidence'] >= PRIORITY_RELIABLE_AT
    result['needs_review'] = result['type_confidence'] < TYPE_THRESHOLD

    if result['needs_review']:
        row = {'time': datetime.now().isoformat(timespec='seconds'),
               'text': ticket.text, **result}
        pd.DataFrame([row]).to_csv(LOG_FILE, mode='a',
                                   header=not os.path.exists(LOG_FILE), index=False)
    return result