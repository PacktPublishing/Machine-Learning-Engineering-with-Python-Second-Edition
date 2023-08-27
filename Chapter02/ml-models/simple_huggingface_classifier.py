from huggingface_hub import hf_hub_download, hf_hub_url, cached_download
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

REPO_ID = "electricweegie/mlewp-sklearn-wine"
FILENAME = "rfc.joblib"

# model = joblib.load(cached_download(
#     hf_hub_url(REPO_ID, FILENAME)
# ))

model = joblib.load(hf_hub_download(REPO_ID, FILENAME))

# Load the dataset
X, y = load_wine(return_X_y=True)
# create an array of True for 2 and False otherwise
y = y == 2

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(model.predict(X_test))

