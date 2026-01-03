# GameTaste: Personal Video Game Rating Analysis

For the full report, please look at the "dsa210.pdf" file in the files.

For recreating the project: 
1. Run fetch_rawg.py to get the general data. You will need to get a (free) API key from RAWG.io. 
2. OR, you may use my already fetched data.
2. Export your personal game list from RAWG.io, using its export function.
3. Run my_rawgdata_to_cleancsv.py to make the data usable, then put your 1-10 ratings in the csv. 
4. OR, make your own data, it will work provided you match your data fields to a rawg export.
5. Now you have everything you need. Run genre_preferences_finder.py for objective 1.
6. Then run exploratory_analysis.py for objective 2.
7. Then run ml_methods.py and ml_recommender for ml methods and the recommendation engine.
8. The other scripts were for early tests, they are not required now.
8. That is all!





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
**RAWG API** (it is a public database): genres, tags, release date, Metacritic score, popularity / added count, playtime.
I will then **merge** my personal ratings with the public metadata using game name.

## 3. Data Collection Plan

2. **Export / scrape** my game list from RAWG.io and save as `data/my_ratings.csv`.  
   - If RAWG API gives my games directly, I will write a small Python script (`src/fetch_rawg.py`) to pull them.
   - If not, I will do one manual export and clean it in a notebook.  
3. For every game in `my_ratings.csv`, **query RAWG API** for extra info (genre, platforms, release year, Metacritic, tags) and save as `data/games_raw_public.csv`.  
4. **Join** the two CSVs in a notebook (`notebooks/01_eda.ipynb`) and start exploratory data analysis.
