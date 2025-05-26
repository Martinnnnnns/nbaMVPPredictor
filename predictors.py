import pandas as pd

def load_data():
    """Load MVP and player data"""
    print("Loading data...")
    mvps = pd.read_csv("mvps.csv")
    players = pd.read_csv("players.csv")
    teams = pd.read_csv("teams.csv")
    return mvps, players, teams

def clean_mvps_data(mvps):
    """Clean MVP data"""
    print("Cleaning MVP data...")
    # Keep only relevant columns
    mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]
    return mvps

def clean_players_data(players):
    """Clean player data"""
    print("Cleaning player data...")
    # Remove unnamed columns and rank
    if "Unnamed: 0" in players.columns:
        del players["Unnamed: 0"]
    if "Rk" in players.columns:
        del players["Rk"]
    
    # Remove asterisks from player names
    players["Player"] = players["Player"].str.replace("*", "", regex=False)
    
    return players

def handle_multiple_teams(df):
    """Handle players who played for multiple teams in one season"""
    print("Handling players with multiple teams...")
    
    def single_team(df_group):
        if df_group.shape[0] == 1:
            return df_group
        else:
            # If player has TOT (total) row, use that but keep the last team
            tot_row = df_group[df_group["Tm"] == "TOT"]
            if not tot_row.empty:
                tot_row = tot_row.copy()
                tot_row["Tm"] = df_group.iloc[-1]["Tm"]  # Use last team
                return tot_row
            else:
                return df_group.iloc[-1:]  # Just take the last team
    
    # Group by player and year, then apply single_team function
    players_clean = df.groupby(["Player", "Year"]).apply(single_team)
    
    # Reset index
    players_clean.index = players_clean.index.droplevel()
    players_clean.index = players_clean.index.droplevel()
    
    return players_clean

def merge_data(players, mvps):
    """Merge player and MVP data"""
    print("Merging player and MVP data...")
    combined = players.merge(mvps, how="outer", on=["Player", "Year"])
    
    # Fill NaN values in MVP columns with 0
    combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)
    
    return combined

def add_team_names(combined):
    """Add full team names using nicknames mapping"""
    print("Adding team names...")
    
    # Load nicknames mapping
    nicknames = {}
    with open("nicknames.csv") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.replace("\n", "").split(",")
                if len(parts) >= 2:
                    abbrev = parts[0].strip()
                    name = parts[1].strip()
                    nicknames[abbrev] = name
    
    combined["Team"] = combined["Tm"].map(nicknames)
    
    return combined

def merge_team_data(combined, teams):
    """Merge team statistics"""
    print("Merging team data...")
    
    # Clean teams data
    if "Unnamed: 0" in teams.columns:
        del teams["Unnamed: 0"]
    
    # Remove asterisks and playoff indicators from team names
    teams["Team"] = teams["Team"].str.replace("*", "", regex=False)
    
    # Filter out division headers
    teams = teams[~teams["W"].str.contains("Division", na=False)]
    
    # Convert data types
    teams = teams.apply(pd.to_numeric, errors='ignore')
    
    # Handle GB column (Games Behind) - replace "—" with 0
    teams["GB"] = pd.to_numeric(teams["GB"].str.replace("—", "0"))
    
    # Merge with combined data
    train = combined.merge(teams, how="outer", on=["Team", "Year"])
    
    return train

def main():
    """Main function to process all data"""
    # Load data
    mvps, players, teams = load_data()
    
    # Clean MVP data
    mvps = clean_mvps_data(mvps)
    
    # Clean player data
    players = clean_players_data(players)
    
    # Handle multiple teams per player per season
    players = handle_multiple_teams(players)
    
    # Merge player and MVP data
    combined = merge_data(players, mvps)
    
    # Add team names
    combined = add_team_names(combined)
    
    # Merge team data
    train = merge_team_data(combined, teams)
    
    # Convert data types
    train = train.apply(pd.to_numeric, errors='ignore')
    
    # Save final dataset
    train.to_csv("player_mvp_stats.csv", index=False)
    print("Final dataset saved to player_mvp_stats.csv")
    
    # Print some basic info
    print(f"Final dataset shape: {train.shape}")
    print(f"Years covered: {train['Year'].min()} to {train['Year'].max()}")
    print(f"Number of players: {train['Player'].nunique()}")
    print(f"MVP winners in dataset: {len(train[train['Share'] > 0.5])}")

if __name__ == "__main__":
    main()