# Mental Health Subreddit Classification

This project implements a confidence-based ensemble model for classifying mental health-related subreddit posts.
bert model and voting model are too large，i didn,t upload


## Project Structure
```bash
.
├── data/                    # Data files (not included due to size)
│   ├── features_data.csv    # Features extracted from posts
│   └── post_data.csv        # Original Reddit posts
├── base_models/             # Base models
│   ├── bert/               # BERT model files
│   ├── xgboost_22cls/      # XGBoost model for 22 categories
│   └── targeted_xgboost/   # XGBoost model for 4 specific categories
├── final_model/            # Final ensemble model
├── bert-training.py        # BERT training script
├── voting_model.py         # Model combination script
├── xgb_22cls.py           # 22-class XGBoost training
└── xgb_4cls.py            # 4-class XGBoost training
```

# First prepare the data
- Place the post_data.csv and features_data.csv in the data/ directory
- Ensure all model files are in their respective directories

# Training order:
  python bert-training.py   
  python xgb_4cls.py          
  python xgb_22cls.py      
  python voting_model.py

# Model Structure
  BERT: Base text classification model
  XGBoost (4 classes): Specifically for socialanxiety, mentalhealth, COVID19_support, and lonely
  XGBoost (22 classes): For remaining categories
  Voting Model: Combines predictions based on BERT confidence
