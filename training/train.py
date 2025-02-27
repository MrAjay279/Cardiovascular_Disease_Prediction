import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load dataset with the correct delimiter
data = pd.read_csv('cardio_train.csv', delimiter=';')

# Drop the 'id' column
data = data.drop('id', axis=1)

# Convert age from days to years
data['age'] = data['age'] / 365

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['gender', 'cholesterol', 'gluc'], drop_first=True)

# Split into features and target
X = data.drop('cardio', axis=1)
y = data['cardio']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize/Standardize features if necessary
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler and feature names
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

# Train and save XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.pkl')

# Generate and save confusion matrix for XGBoost
conf_matrix_xgb = confusion_matrix(y_test, xgb_model.predict(X_test))
joblib.dump(conf_matrix_xgb, 'conf_matrix_xgb.pkl')

# Generate and save ROC curve for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure()
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='XGBoost ROC curve (area = %0.2f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('newroc_curve_xgb.png')

# Feature Importance for XGBoost
feature_importance_xgb = xgb_model.feature_importances_
joblib.dump(feature_importance_xgb, 'feature_importance_xgb.pkl')

# Train and save Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.pkl')

# Generate and save confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, rf_model.predict(X_test))
joblib.dump(conf_matrix_rf, 'conf_matrix_rf.pkl')

# Generate and save ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve_rf.png')

# Feature Importance for Random Forest
feature_importance_rf = rf_model.feature_importances_
joblib.dump(feature_importance_rf, 'feature_importance_rf.pkl')

# Print accuracy for XGBoost and Random Forest models
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f'XGBoost Accuracy: {xgb_accuracy:}')
print(f'Random Forest Accuracy: {rf_accuracy:}')
