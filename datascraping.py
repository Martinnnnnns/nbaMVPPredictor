import requests
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def setup_directories():
    """Create necessary directories for storing HTML files"""
    directories = ['mvp', 'player', 'team']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def setup_chrome_driver(chromedriver_path):
    """Setup Chrome driver with modern Selenium syntax"""
    options = Options()
    options.add_argument('--headless')  # Run in background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_mvp_data(years):
    """Scrape MVP voting data for given years"""
    print("Scraping MVP data...")
    url_start = "https://www.basketball-reference.com/awards/awards_{}.html"
    
    for year in years:
        url = url_start.format(year)
        print(f"Scraping MVP data for {year}...")
        
        try:
            data = requests.get(url)
            data.raise_for_status()
            
            with open(f"mvp/{year}.html", "w+", encoding='utf-8') as f:
                f.write(data.text)
            print(f"✓ MVP data for {year} saved")
        except Exception as e:
            print(f"✗ Error scraping MVP data for {year}: {e}")

def scrape_player_data_with_selenium(years, chromedriver_path):
    """Scrape player statistics using Selenium (needed for full page loading)"""
    print("Scraping player data with Selenium...")
    
    driver = setup_chrome_driver(chromedriver_path)
    player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"
    
    try:
        for year in years:
            url = player_stats_url.format(year)
            print(f"Scraping player data for {year}...")
            
            try:
                driver.get(url)
                driver.execute_script("window.scrollTo(1,10000)")
                time.sleep(3)  # Wait for dynamic content to load
                
                with open(f"player/{year}.html", "w+", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"✓ Player data for {year} saved")
            except Exception as e:
                print(f"✗ Error scraping player data for {year}: {e}")
    finally:
        driver.quit()

def scrape_team_data(years):
    """Scrape team standings data"""
    print("Scraping team data...")
    team_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"
    
    for year in years:
        url = team_stats_url.format(year)
        print(f"Scraping team data for {year}...")
        
        try:
            data = requests.get(url)
            data.raise_for_status()
            
            with open(f"team/{year}.html", "w+", encoding='utf-8') as f:
                f.write(data.text)
            print(f"✓ Team data for {year} saved")
        except Exception as e:
            print(f"✗ Error scraping team data for {year}: {e}")

def process_mvp_data(years):
    """Process scraped MVP HTML files into DataFrame"""
    print("Processing MVP data...")
    dfs = []
    
    for year in years:
        try:
            with open(f"mvp/{year}.html", encoding='utf-8') as f:
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
            print(f"✓ Processed MVP data for {year}")
        except Exception as e:
            print(f"✗ Error processing MVP data for {year}: {e}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

def process_player_data(years):
    """Process scraped player HTML files into DataFrame"""
    print("Processing player data...")
    dfs = []
    
    for year in years:
        try:
            with open(f"player/{year}.html", encoding='utf-8') as f:
                page = f.read()
            
            soup = BeautifulSoup(page, 'html.parser')
            
            # Remove header rows that interfere with parsing
            for thead in soup.find_all('tr', class_="thead"):
                thead.decompose()
            
            player_table = soup.find_all(id="per_game_stats")[0]
            player_df = pd.read_html(str(player_table))[0]
            player_df["Year"] = year
            dfs.append(player_df)
            print(f"✓ Processed player data for {year}")
        except Exception as e:
            print(f"✗ Error processing player data for {year}: {e}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

def process_team_data(years):
    """Process scraped team HTML files into DataFrame"""
    print("Processing team data...")
    dfs = []
    
    for year in years:
        try:
            with open(f"team/{year}.html", encoding='utf-8') as f:
                page = f.read()
            
            soup = BeautifulSoup(page, 'html.parser')
            
            # Remove header rows that interfere with parsing
            for thead in soup.find_all('tr', class_="thead"):
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
            
            print(f"✓ Processed team data for {year}")
        except Exception as e:
            print(f"✗ Error processing team data for {year}: {e}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

def main():
    # Configuration - UPDATED FOR 2021-2024
    years = list(range(1991, 2025))  # 1991 to 2024
    chromedriver_path = "/Users/bernardoguterrres/Desktop/DS/nbaMVP"
    
    print(f"Scraping data for years: {min(years)} to {max(years)}")
    
    # Create directories
    setup_directories()
    
    # Check if chromedriver path exists
    if not os.path.exists(chromedriver_path):
        print(f"WARNING: ChromeDriver not found at {chromedriver_path}")
        print("Please check the chromedriver path")
        return
    
    # Scrape data
    scrape_mvp_data(years)
    scrape_team_data(years)
    scrape_player_data_with_selenium(years, chromedriver_path)
    
    # Process scraped data into DataFrames and save as CSV
    print("\n" + "="*50)
    print("PROCESSING DATA")
    print("="*50)
    
    mvps = process_mvp_data(years)
    if not mvps.empty:
        mvps.to_csv("mvps.csv", index=False)
        print("MVP data saved to mvps.csv")
    
    players = process_player_data(years)
    if not players.empty:
        players.to_csv("players.csv", index=False)
        print("Player data saved to players.csv")
    
    teams = process_team_data(years)
    if not teams.empty:
        teams.to_csv("teams.csv", index=False)
        print("Team data saved to teams.csv")
    
    print("\n" + "="*50)
    print("DATA SCRAPING COMPLETED!")
    print("="*50)
    print(f"Years scraped: {min(years)} to {max(years)}")
    print("Files created: mvps.csv, players.csv, teams.csv")

if __name__ == "__main__":
    main()