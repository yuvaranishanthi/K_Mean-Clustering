import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# 1. Load dataset (replace with your own file)
df = pd.read_csv("top10s.csv", encoding="ISO-8859-1")

# 2. Select numeric features for clustering
features = ["bpm", "nrgy", "dnce", "val", "pop"]
X = df[features]

# 3. Train the KMeans model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Assign clusters
df["cluster"] = kmeans.predict(X)

# 5. Create cluster descriptions
cluster_descriptions = {}
for cluster_id in sorted(df["cluster"].unique()):
    examples = df[df["cluster"] == cluster_id][["title", "artist"]].head(3)
    example_list = [f"{row['title']} - {row['artist']}" for _, row in examples.iterrows()]
    cluster_descriptions[cluster_id] = (
        f"Cluster {cluster_id}: {len(df[df['cluster'] == cluster_id])} songs. "
        f"Examples: {', '.join(example_list)}"
    )

# 6. Save model, features, descriptions, and ranges
os.makedirs("model", exist_ok=True)
joblib.dump(kmeans, "model/kmeans_model.pkl")  # ✅ Save the actual model

joblib.dump(features, "model/features.pkl")
joblib.dump(cluster_descriptions, "model/cluster_descriptions.pkl")

feature_ranges = {f: (X[f].min(), X[f].max()) for f in features}
joblib.dump(feature_ranges, "model/feature_ranges.pkl")

print("✅ K-Means model trained and saved successfully with descriptions.")
