# GameTaste: Personal Video Game Rating Analysis

## 1. Project Proposal

I want to analyze **my own video game taste** using the ratings of 200+ games I have on **RAWG.io** and combine that with **publicly available game metadata** (genres, platforms, release year, Metacritic score, tags, popularity, etc.). The goal is to see:

- what kinds of games I actually like (not just what I say I like),
- how my ratings compare to “global”/public ratings,
- and whether we can predict how much I would like a game based only on its features.

So this will be a personalized mini recommender / taste analysis project.

## 2. Data to Be Used

I will use **two types of data**:

### 2.1. My Personal Data
- Source: **my RAWG.io profile** (list of games I played/rated).
- Fields I will try to get:
  - game name / slug
  - my rating (target variable)
  - date added / played (if available)
  - platform (if visible)
- This is the “extra / own” data the project requires.

### 2.2. Public Data
For the same games, I will fetch public game info from:
**IGDB API** (it is a public database): genres, tags, release date, Metacritic score, popularity / added count, playtime.
I will then **merge** my personal ratings with the public metadata using game name.

## 3. Data Collection Plan

2. **Export / scrape** my game list from RAWG.io and save as `data/my_ratings.csv`.  
   - If RAWG API gives my games directly, I will write a small Python script (`src/fetch_rawg.py`) to pull them.
   - If not, I will do one manual export and clean it in a notebook.  
3. For every game in `my_ratings.csv`, **query IGDB API** for extra info (genre, platforms, release year, Metacritic, tags) and save as `data/games_raw_public.csv`.  
4. **Join** the two CSVs in a notebook (`notebooks/01_eda.ipynb`) and start exploratory data analysis.
