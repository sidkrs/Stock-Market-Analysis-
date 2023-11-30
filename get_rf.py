import requests
from bs4 import BeautifulSoup
import re

# URL of the page to scrape
URL = 'https://www.cnbc.com/quotes/US3M'

def extract_treasury_rate(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Use regex to find the rate
        pattern = r'"US TREASURY-CURRENT 3 MONTH","last":"(.*?)%"'
        match = re.search(pattern, str(soup))

        if match:
            # Convert the percentage to a decimal
            rate_percentage = match.group(1)
            rate_decimal = round(float(rate_percentage) / 100, 4)
            return rate_decimal
        else:
            return 'Rate not found'
    except requests.RequestException as e:
        return f'Error fetching data: {e}'
    except re.error as ree:
        return f'Regex error: {ree}'

# Extract and print the 3-month Treasury Bill rate

def main():
    treasury_rate = extract_treasury_rate(URL)
    print('3-Month Treasury Bill Rate (as decimal):', treasury_rate)

if __name__ == '__main__':
    main()
