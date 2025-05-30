# NBA MVP Prediction Model

A machine learning project that scrapes NBA data from Basketball Reference and predicts MVP winners using Ridge Regression and Random Forest models.

## Project Overview

This project builds a comprehensive NBA MVP prediction system by:
- Scraping historical NBA data (1991-2024) including player stats, team standings, and MVP voting results
- Processing and cleaning the data to create meaningful features
- Training machine learning models to predict MVP voting shares
- Backtesting model performance across multiple seasons

## Features

- **Data Scraping**: Automated scraping of NBA data from Basketball Reference
- **Data Processing**: Handles multiple teams per player, creates ratio features, and merges datasets
- **Machine Learning**: Implements Ridge Regression and Random Forest models
- **Backtesting**: Evaluates model performance across historical seasons
- **Feature Analysis**: Identifies most important predictive features

## Dataset

The project uses three main data sources from Basketball Reference:
- **Player Statistics**: Per-game stats for all NBA players (1991-2024)
- **Team Standings**: Win-loss records and team performance metrics
- **MVP Voting**: Historical MVP voting results and vote shares

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nba-mvp-prediction.git
cd nba-mvp-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download ChromeDriver and update the path in `datascraping.py`:
```python
chromedriver_path = "/path/to/your/chromedriver"
```

4. Create a `nicknames.csv` file with team abbreviations and full names:
```csv
ATL,Atlanta Hawks
BOS,Boston Celtics
BRK,Brooklyn Nets
...
```

## Usage

### Complete Pipeline

Run the entire pipeline with these three commands:

```bash
# 1. Scrape data from Basketball Reference
python datascraping.py

# 2. Process and clean the data
python predictors.py

# 3. Train models and make predictions
python machine_learning.py
```

### Individual Scripts

**Data Scraping** (`datascraping.py`):
- Scrapes MVP voting, player stats, and team standings
- Creates HTML files and CSV outputs
- Requires ChromeDriver for dynamic content

**Data Processing** (`predictors.py`):
- Cleans and merges datasets
- Handles players with multiple teams
- Creates final `player_mvp_stats.csv`

**Machine Learning** (`machine_learning.py`):
- Trains Ridge Regression and Random Forest models
- Performs backtesting across seasons
- Generates predictions and feature importance analysis

## Model Performance

The models are evaluated using Average Precision for top-5 MVP candidates:

- **Ridge Regression**: Baseline model with standard features
- **Ridge with Ratios**: Enhanced with year-normalized ratio features
- **Random Forest**: Uses categorical encodings for recent years

## Key Features Used

The model considers 33+ predictor variables including:
- **Basic Stats**: Points, assists, rebounds, steals, blocks
- **Advanced Stats**: Shooting percentages, efficiency metrics
- **Team Performance**: Wins, losses, win percentage, strength of schedule
- **Ratio Features**: Year-normalized performance metrics

## Output Files

After running the complete pipeline:
- `mvps.csv`: MVP voting data
- `players.csv`: Player statistics
- `teams.csv`: Team standings
- `player_mvp_stats.csv`: Final merged dataset
- `predictions_[year].csv`: Model predictions for each year
- `feature_importance.csv`: Feature importance rankings
- `correlations.csv`: Correlation analysis

## Model Insights

The analysis reveals key predictive factors for MVP selection:
- Team success (wins, win percentage) is crucial
- Individual performance metrics (points, assists, efficiency)
- Advanced stats like Player Efficiency Rating
- Ratio features that normalize for league-wide changes

## Requirements

- Python 3.7+
- ChromeDriver (for Selenium web scraping)
- Internet connection (for data scraping)
- See `requirements.txt` for full package list

## Data Sources

All data is scraped from [Basketball Reference](https://www.basketball-reference.com/), including:
- MVP voting results by year
- Player per-game statistics
- Team standings and performance metrics

## Limitations

- Requires manual ChromeDriver setup
- Dependent on Basketball Reference website structure
- Historical data only (predictions are retrospective)
- Does not account for narrative factors in MVP voting

## Future Improvements

- Add playoff performance metrics
- Incorporate advanced analytics (BPM, VORP, etc.)
- Include media sentiment analysis
- Real-time prediction capabilities
- Web interface for predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Basketball Reference for providing comprehensive NBA data
- The NBA for creating an amazing sport with rich statistical data
- The open-source community for the tools and libraries used

## Contact

For questions or suggestions, please open an issue or contact [your-email@example.com]