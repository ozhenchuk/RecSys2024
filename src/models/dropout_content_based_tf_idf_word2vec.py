import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

import gensim.downloader as api

word2vec = api.load("glove-wiki-gigaword-50")


def get_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def preprocess_data(ratings, movies):
    data = pd.merge(ratings, movies, on="movie_id")

    data["genres"] = data["genres"].apply(lambda x: x.split("|"))
    data["title"] = data["title"].str.replace(r"\(\d+\)", "").str.strip()
    data["genres"] = data["genres"].apply(lambda x: " ".join(x))

    data["genre_embeddings"] = data["genres"].apply(
        lambda x: get_embedding(x, word2vec)
    )
    data["title_embeddings"] = data["title"].apply(lambda x: get_embedding(x, word2vec))
    embeddings = np.hstack(
        (
            data["genre_embeddings"].values.tolist(),
            data["title_embeddings"].values.tolist(),
        )
    )

    scaler = MinMaxScaler()

    item_profiles = scaler.fit_transform(embeddings)

    user_profiles = data.groupby("user_id").apply(
        lambda x: np.mean(item_profiles[x.index], axis=0)
    )
    user_profiles = scaler.fit_transform(user_profiles.tolist())

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=5)

    train_user_profiles = torch.tensor(
        user_profiles[train_data["user_id"].values - 1], dtype=torch.float32
    )
    train_item_profiles = torch.tensor(
        item_profiles[train_data["movie_id"].values - 1], dtype=torch.float32
    )
    train_ratings = torch.tensor(train_data["rating"].values, dtype=torch.float32)

    test_user_profiles = torch.tensor(
        user_profiles[test_data["user_id"].values - 1], dtype=torch.float32
    )
    test_item_profiles = torch.tensor(
        item_profiles[test_data["movie_id"].values - 1], dtype=torch.float32
    )
    test_ratings = torch.tensor(test_data["rating"].values, dtype=torch.float32)
    return (
        data,
        (train_user_profiles, train_item_profiles, train_ratings),
        (
            test_user_profiles,
            test_item_profiles,
            test_ratings,
        ),
    )


class DropoutContentBasedModel(nn.Module):
    def __init__(self, input_dim):
        super(DropoutContentBasedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, user_profile, item_profile):
        x = torch.cat([user_profile, item_profile], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def train_and_evaluate(model, data, num_epochs=10, lr=0.001):
    (
        _,
        (train_user_profiles, train_item_profiles, train_ratings),
        (
            test_user_profiles,
            test_item_profiles,
            test_ratings,
        ),
    ) = data
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_user_profiles, train_item_profiles).squeeze()
        loss = criterion(outputs, train_ratings)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_user_profiles, test_item_profiles).squeeze()
        test_loss = mean_squared_error(test_ratings, test_outputs)
        print(f"Test MSE: {test_loss}")

    return train_losses, test_loss


def get_user_profile(user_id, data, item_profiles):
    user_ratings = data[data["UserID"] == user_id]
    user_profile = np.mean(item_profiles[user_ratings["MovieID"].values - 1], axis=0)
    return torch.tensor(user_profile, dtype=torch.float32).unsqueeze(0)


def get_recommendations(user_id, model, data, item_profiles, top_n=10):
    user_profile = get_user_profile(user_id, data, item_profiles)
    item_profiles_tensor = torch.tensor(item_profiles, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scores = model(
            user_profile.repeat(item_profiles_tensor.shape[0], 1), item_profiles_tensor
        ).squeeze()
    top_items_indices = torch.topk(scores, top_n).indices.numpy()
    return top_items_indices, scores.numpy()


def display_recommendations(user_id, top_items_indices, scores, data, movies, top_n=10):
    top_movies = movies.iloc[top_items_indices]
    top_scores = scores[top_items_indices]
    recommendations = pd.DataFrame(
        {
            "MovieID": top_movies["MovieID"],
            "Title": top_movies["Title"],
            "Score": top_scores,
        }
    )
    recommendations = recommendations.sort_values(by="Score", ascending=False).head(
        top_n
    )
    print(f"Recommendations for User {user_id}:")
    print(recommendations)
