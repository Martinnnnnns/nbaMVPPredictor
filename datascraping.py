import requests
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def setup_directories():
    """Create necessary directories for storing HTML files"""
    directories = ['mvp', 'player', 'team']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def scrape_mvp_data(years):
    """Scrape MVP voting data for given years"""
    print("Scraping MVP data...")
    url_start = "https://www.basketball-reference.com/awards/awards_{}.html"
    
    for year in years:
        url = url_start.format(year)
        print(f"Scraping MVP data for {year}...")
        
        data = requests.get(url)
        
        with open(f"mvp/{year}.html", "w+") as f:
            f.write(data.text)

def scrape_player_data_with_selenium(years, chromedriver_path):
    """Scrape player statistics using Selenium (needed for full page loading)"""
    print("Scraping player data with Selenium...")
    
    # Setup Chrome driver
    driver = webdriver.Chrome(executable_path=chromedriver_path)
    
    player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"
    
    try:
        for year in years:
            url = player_stats_url.format(year)
            print(f"Scraping player data for {year}...")
            
            driver.get(url)
            driver.execute_script("window.scrollTo(1,10000)")
            time.sleep(2)
            
            with open(f"player/{year}.html", "w+") as f:
                f.write(driver.page_source)
    finally:
        driver.quit()

def scrape_team_data(years):
    """Scrape team standings data"""
    print("Scraping team data...")
    team_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"
    
    for year in years:
        url = team_stats_url.format(year)
        print(f"Scraping team data for {year}...")
        
        data = requests.get(url)
        
        with open(f"team/{year}.html", "w+") as f:
            f.write(data.text)

def process_mvp_data(years):
    """Process scraped MVP HTML files into DataFrame"""
    print("Processing MVP data...")
    dfs = []
    
    for year in years:
        with open(f"mvp/{year}.html") as f:
            page = f.read()
        
        soup = BeautifulSoup(page, 'html.parser')
        
        # Remove header row that interferes with parsing
        over_header = soup.find('tr', class_="over_header")
        if over_header:
            over_header.decompose()
        
        mvp_table = soup.find_all(id="mvp")[0]
        mvp_df = pd.read_html(str(mvp_table))[0]
        mvp_df["Year"] = year
        dfs.append(mvp_df)
    
    return pd.concat(dfs)

def process_player_data(years):
    """Process scraped player HTML files into DataFrame"""
    print("Processing player data...")
    dfs = []
    
    for year in years:
        with open(f"player/{year}.html") as f:
            page = f.read()
        
        soup = BeautifulSoup(page, 'html.parser')
        
        # Remove header row that interferes with parsing
        thead = soup.find('tr', class_="thead")
        if thead:
            thead.decompose()
        
        player_table = soup.find_all(id="per_game_stats")[0]
        player_df = pd.read_html(str(player_table))[0]
        player_df["Year"] = year
        dfs.append(player_df)
    
    return pd.concat(dfs)

def process_team_data(years):
    """Process scraped team HTML files into DataFrame"""
    print("Processing team data...")
    dfs = []
    
    for year in years:
        with open(f"team/{year}.html") as f:
            page = f.read()
        
        soup = BeautifulSoup(page, 'html.parser')
        
        # Remove header row that interferes with parsing
        thead = soup.find('tr', class_="thead")
        if thead:
            thead.decompose()
        
        # Process Eastern Conference
        e_table = soup.find_all(id="divs_standings_E")[0]
        e_df = pd.read_html(str(e_table))[0]
        e_df["Year"] = year
        e_df["Team"] = e_df["Eastern Conference"]
        del e_df["Eastern Conference"]
        dfs.append(e_df)
        
        # Process Western Conference
        w_table = soup.find_all(id="divs_standings_W")[0]
        w_df = pd.read_html(str(w_table))[0]
        w_df["Year"] = year
        w_df["Team"] = w_df["Western Conference"]
        del w_df["Western Conference"]
        dfs.append(w_df)
    
    return pd.concat(dfs)

def main():
    # Configuration
    years = list(range(1991, 2022))  # 1991 to 2021
    chromedriver_path = "/path/to/your/chromedriver"  # UPDATE THIS PATH
    
    # Create directories
    setup_directories()
    
    # Scrape data
    scrape_mvp_data(years)
    scrape_team_data(years)
    
    # Note: You need to update the chromedriver path above
    print("IMPORTANT: Update chromedriver_path variable with your actual ChromeDriver path")
    print("You can download ChromeDriver from: https://chromedriver.chromium.org/downloads")
    print("On Mac, you may need to run: xattr -d com.apple.quarantine chromedriver")
    
    # Uncomment the line below after updating chromedriver_path
    # scrape_player_data_with_selenium(years, chromedriver_path)
    
    # Process scraped data into DataFrames and save as CSV
    mvps = process_mvp_data(years)
    mvps.to_csv("mvps.csv", index=False)
    print("MVP data saved to mvps.csv")
    
    # Uncomment these lines after scraping player data
    # players = process_player_data(years)
    # players.to_csv("players.csv", index=False)
    # print("Player data saved to players.csv")
    
    teams = process_team_data(years)
    teams.to_csv("teams.csv", index=False)
    print("Team data saved to teams.csv")
    
    print("Data scraping completed!")

if __name__ == "__main__":
    main()