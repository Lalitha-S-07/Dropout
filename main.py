from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://127.0.0.1:8000"]  # replace with your frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow ALL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load('model3.pkl')
X_train = joblib.load('trainexp.pkl')
print(X_train.shape)

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    mode='classification',
    random_state=42
)

class inputdata(BaseModel):
    Family_Income_Monthly: int
    Current_CGPA: float
    Physical_Verification_Score: int
    Siblings_Count: int
    Parents_Employed: int
    School_Type_Public: int


def correct_feature(X_train, exp):
    feature_names = X_train.columns.tolist()
    clean_result = []

    for feat_cond, val in exp.as_list():
        for feat in feature_names:
            if feat in feat_cond:
                clean_result.append({feat: round(val, 2)})
                break
    return clean_result


# ⭐ NEW CODE ADDED — Generate explanation sentences
# ⭐ UPDATED generate_explanation — interprets LIME contributions correctly
def generate_explanation(feature_scores):
    # feature_scores maps feature name -> contribution (float)
    parts = []

    # helper to describe magnitude
    def mag_text(v):
        a = abs(v)
        if a >= 0.30:
            return "strongly"
        if a >= 0.10:
            return "moderately"
        return "slightly"

    # Family income: positive contribution increases dropout probability
    v = feature_scores.get("Family_Income_Monthly", 0)
    if v > 0:
        parts.append(f"Family income {mag_text(v)} increases the predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"Family income {mag_text(v)} reduces the predicted dropout risk (contribution {v}).")
    else:
        parts.append("Family income has negligible contribution to the prediction.")

    # CGPA: higher positive value means it increases dropout probability in your model (interpretation follows sign)
    v = feature_scores.get("Current_CGPA", 0)
    if v > 0:
        parts.append(f"CGPA {mag_text(v)} increases predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"CGPA {mag_text(v)} reduces predicted dropout risk (contribution {v}).")
    else:
        parts.append("CGPA has negligible contribution to the prediction.")

    # Physical verification
    v = feature_scores.get("Physical_Verification_Score", 0)
    if v > 0:
        parts.append(f"Physical verification score {mag_text(v)} increases predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"Physical verification score {mag_text(v)} reduces predicted dropout risk (contribution {v}).")
    else:
        parts.append("Physical verification score has negligible contribution.")

    # Parents employed
    v = feature_scores.get("Parents_Employed", 0)
    if v > 0:
        parts.append(f"Parents being employed {mag_text(v)} increases predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"Parents being employed {mag_text(v)} reduces predicted dropout risk (contribution {v}).")
    else:
        parts.append("Parents' employment status has negligible contribution.")

    # Siblings count
    v = feature_scores.get("Siblings_Count", 0)
    if v > 0:
        parts.append(f"Siblings count {mag_text(v)} increases predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"Siblings count {mag_text(v)} reduces predicted dropout risk (contribution {v}).")
    else:
        parts.append("Siblings count has negligible contribution.")

    # School type
    v = feature_scores.get("School_Type_Public", 0)
    if v > 0:
        parts.append(f"Public school type {mag_text(v)} increases predicted dropout risk (contribution {v}).")
    elif v < 0:
        parts.append(f"Public school type {mag_text(v)} reduces predicted dropout risk (contribution {v}).")
    else:
        parts.append("School type has negligible contribution.")

    return " ".join(parts)


@app.get("/")
def home():
    return {"message": "welcome to dropout prediction"}


@app.post('/predict')
def predict(data: inputdata):

    income = data.Family_Income_Monthly
    cgpa = data.Current_CGPA
    pvscore = data.Physical_Verification_Score
    sibilings_count = data.Siblings_Count
    parents_employeed = data.Parents_Employed
    school = data.School_Type_Public

    df = pd.DataFrame(
        [[income, cgpa, pvscore, sibilings_count, parents_employeed, school]],
        columns=['Family_Income_Monthly', 'Current_CGPA', 'Physical_Verification_Score',
                 'Siblings_Count', 'Parents_Employed', 'School_Type_Public']
    )

    prob = model.predict_proba(df)[:, 1][0]
    dropout_likelihood = int(prob * 100)
    retension_score = 100 - dropout_likelihood

    exp = explainer.explain_instance(
        data_row=df.loc[0].values,
        predict_fn=model.predict_proba
    )

    result = correct_feature(X_train, exp)

    # ⭐ NEW CODE ADDED — Convert list of dicts into a single dict
    feature_scores = {list(item.keys())[0]: list(item.values())[0] for item in result}

    # ⭐ NEW CODE ADDED — Generate natural explanation sentence
    explanation = generate_explanation(feature_scores)

    # ⭐ CHANGED CODE — Add explanation to API output
    result.extend([
        {"dropout_likelihood": dropout_likelihood},
        {"retension_score": retension_score},
        {"explanation": explanation}
    ])

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

