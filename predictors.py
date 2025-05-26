import pandas as pd

def load_data():
    """Load MVP and player data"""
    print("Loading data...")
    mvps = pd.read_csv("mvps.csv")
    players = pd.read_csv("players.csv")
    teams = pd.read_csv("teams.csv")
    print(f"MVP data: {mvps.shape}")
    print(f"Player data: {players.shape}")
    print(f"Team data: {teams.shape}")
    return mvps, players, teams

def clean_mvps_data(mvps):
    """Clean MVP data"""
    print("Cleaning MVP data...")
    # Keep only relevant columns
    mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]
    print(f"MVP data after cleaning: {mvps.shape}")
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
    
    print(f"Player data after cleaning: {players.shape}")
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
    
    print(f"Player data after handling multiple teams: {players_clean.shape}")
    return players_clean

def merge_data(players, mvps):
    """Merge player and MVP data"""
    print("Merging player and MVP data...")
    combined = players.merge(mvps, how="outer", on=["Player", "Year"])
    
    # Fill NaN values in MVP columns with 0
    combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)
    
    print(f"Combined data shape: {combined.shape}")
    return combined

def add_team_names(combined):
    """Add full team names using nicknames mapping"""
    print("Adding team names...")
    
    # Load nicknames mapping
    nicknames = {}
    try:
        with open("nicknames.csv") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():  # Skip empty lines
                    parts = line.replace("\n", "").split(",")
                    if len(parts) >= 2:
                        abbrev = parts[0].strip()
                        name = parts[1].strip()
                        nicknames[abbrev] = name
        print(f"Loaded {len(nicknames)} team mappings")
    except FileNotFoundError:
        print("Warning: nicknames.csv not found. Please create this file.")
        return combined
    
    combined["Team"] = combined["Tm"].map(nicknames)
    
    # Check for unmapped teams
    unmapped = combined[combined["Team"].isnull()]["Tm"].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped teams found: {unmapped}")
    
    return combined

def merge_team_data(combined, teams):
    """Merge team statistics"""
    print("Merging team data...")
    
    # Clean teams data
    if "Unnamed: 0" in teams.columns:
        del teams["Unnamed: 0"]
    
    # Remove asterisks and playoff indicators from team names
    teams["Team"] = teams["Team"].str.replace("*", "", regex=False)
    
    # Filter out division headers and other non-team rows
    teams = teams[~teams["W"].astype(str).str.contains("Division", na=False)]
    teams = teams[~teams["W"].astype(str).str.contains("Conference", na=False)]
    
    # Convert data types
    teams = teams.apply(pd.to_numeric, errors='ignore')
    
    # Handle GB column (Games Behind) - replace "—" with 0
    if "GB" in teams.columns:
        teams["GB"] = teams["GB"].astype(str).str.replace("—", "0")
        teams["GB"] = pd.to_numeric(teams["GB"], errors='coerce').fillna(0)
    
    # Merge with combined data
    before_merge = combined.shape[0]
    train = combined.merge(teams, how="left", on=["Team", "Year"])
    after_merge = train.shape[0]
    
    print(f"Rows before team merge: {before_merge}")
    print(f"Rows after team merge: {after_merge}")
    print(f"Final dataset shape: {train.shape}")
    
    return train

def main():
    """Main function to process all data"""
    print("="*60)
    print("STARTING DATA PROCESSING FOR NBA MVP PREDICTION")
    print("="*60)
    
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
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA PROCESSING SUMMARY")
    print("="*50)
    print(f"Final dataset shape: {train.shape}")
    print(f"Years covered: {train['Year'].min()} to {train['Year'].max()}")
    print(f"Number of unique players: {train['Player'].nunique()}")
    print(f"MVP winners in dataset: {len(train[train['Share'] > 0.5])}")
    print(f"Players with MVP votes: {len(train[train['Share'] > 0])}")
    
    # Show years with data
    years_available = sorted(train['Year'].unique())
    print(f"Years with data: {years_available}")
    
    print("\nData processing completed successfully!")

if __name__ == "__main__":
    main()