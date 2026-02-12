import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# ১. ডেটা লোড
df = pd.read_excel('data_cleaned.xlsx')
drop_cols = ['Institute', 'Session ', 'Session']
df = df.drop(columns=drop_cols, errors='ignore')

# ২. জেন্ডার অনুযায়ী ডেটা আলাদা করা
male_data = df[df['Gender'] == 'Male'].drop(columns=['Gender'])
female_data = df[df['Gender'] == 'Female'].drop(columns=['Gender'])

# ৩. মডেল ও ব্যাকগ্রাউন্ড ডেটা সেভ করার ফাংশন
def train_shap_model(data, gender_type, force_include=None):
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # ফিচার ইম্পরট্যান্স বের করা
    temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_model.fit(X, y)
    feature_importance = pd.Series(temp_model.feature_importances_, index=X.columns)
    
    # ফিচার সিলেকশন লজিক
    if force_include:
        remaining_count = 15 - len(force_include)
        remaining_features = feature_importance.drop(labels=force_include, errors='ignore')
        top_remaining = remaining_features.nlargest(remaining_count).index.tolist()
        final_features = force_include + top_remaining
    else:
        final_features = feature_importance.nlargest(15).index.tolist()
    
    print(f"\n✅ Features for {gender_type}: {final_features}")
    
    # ফাইনাল মডেল ট্রেইন
    X_final = X[final_features]
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_final, y)
    
    # সেভ করা (মডেল + ফিচার + ব্যাকগ্রাউন্ড ডেটা)
    joblib.dump(final_model, f'{gender_type}_shap_model.pkl')
    joblib.dump(final_features, f'{gender_type}_shap_features.pkl')
    
    # SHAP এর জন্য ১০০টি স্যাম্পল ডেটা সেভ রাখা
    background_data = X_final.sample(100, random_state=42)
    joblib.dump(background_data, f'{gender_type}_shap_background.pkl')
    
    print(f"🎉 {gender_type} Model & Data Saved!")

# ৪. রান করা
train_shap_model(male_data, 'male')
train_shap_model(female_data, 'female', force_include=['Weekly Study Time', 'Weekly Library Time'])