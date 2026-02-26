# Film Recommendation System

An interactive movie recommendation app that suggests films based on user behavior and movie similarity.

## Overview
This project uses collaborative filtering to generate personalized movie recommendations and similar-movie suggestions from historical interaction data.

## Features
- Recommend movies by user ID or username
- Recommend similar movies based on a selected title
- Filter recommendations by genre
- Display a user's interacted movies
- Show top trending movies by interaction count

## Tech Stack
- Python
- Streamlit
- Pandas
- scikit-surprise (KNN-based collaborative filtering)

## Dataset Files
- `interactions.csv`
- `item_features.csv`
- `user_data.csv`

## Installation
```bash
pip install -r requirements.txt
```

## Run Locally
```bash
streamlit run FilmRec.py
```

## Resume Summary
Built a movie recommendation web app using Python, Streamlit, Pandas, and scikit-surprise, implementing item-based collaborative filtering, top-N personalized recommendations, genre filtering, and trending movie analytics.

## Notes
The current script contains absolute local file paths. For portability, replace them with relative paths before sharing or deploying.
