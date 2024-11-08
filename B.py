import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# مسیر فایل داده‌ها
data_path = 'I:\\IOT\\HW2\\For mostafa\\data.txt'
data = pd.read_csv(data_path, delimiter=";")

# حذف ویرگول‌ها از تمامی ستون‌های عددی و تبدیل به float
for column in data.select_dtypes(include=['object']).columns:
    data[column] = data[column].replace({',': ''}, regex=True)  # حذف ویرگول‌ها
    data[column] = pd.to_numeric(data[column], errors='coerce')  # تبدیل به عدد

# انتخاب ویژگی‌ها و هدف
X = data[['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005']]  # ویژگی‌های جدید از دیتابیس
y = data['MT_006']  # هدف جدید

# تقسیم داده‌ها به داده‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# مدل‌های مختلف
models = {
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# نتایج ارزیابی مدل‌ها
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ذخیره نتایج در دیکشنری
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R^2 Score": r2
    }

    # چاپ نتایج در کنسول
    print(f"\n{name} Results:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R^2 Score: {r2}")

# نمودار مقایسه پیش‌بینی‌ها با داده‌های واقعی
plt.figure(figsize=(14, 6))

for i, (name, model) in enumerate(models.items(), start=1):
    y_pred = model.predict(X_test)

    plt.subplot(1, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # خط ایده‌آل
    plt.xlabel("Actual MT_006")
    plt.ylabel("Predicted MT_006")
    plt.title(f"{name} - Actual vs Predicted")

plt.tight_layout()
plt.show()

# نمودار ارزیابی مدل‌ها
metrics_names = ["MAE", "MSE", "R^2 Score"]
metrics_values = {metric: [results[model][metric] for model in results] for metric in metrics_names}

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics_names):
    plt.bar([name for name in results.keys()], metrics_values[metric], alpha=0.6, label=metric)

plt.xlabel("Models")
plt.ylabel("Metrics")
plt.title("Model Evaluation Metrics")
plt.legend()
plt.show()
