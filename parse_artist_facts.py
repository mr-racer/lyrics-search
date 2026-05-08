# artist  should be in format like "The Weeknd" -> "the-weeknd", "Kanye West" - "kanye-west" (.lower().split() and "-".join(...)

html_content = requests.get(f'https://www.songfacts.com/facts/{artist}').text

def extract_artist_facts(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Target the specific list containing the facts
    facts_container = soup.find('ul', class_='artistfacts-results')
    if not facts_container:
        return ""

    artistfacts = []
    for li in facts_container.find_all('li'):
        inner_div = li.find('div', class_='inner')
        if inner_div:
            # Extract raw text, replace HTML tags with spaces, strip leading/trailing whitespace
            raw_text = inner_div.get_text(separator=' ', strip=True)
            
            # Collapse multiple spaces/newlines into a single space
            cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
            
            # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
            cleaned_text = html.unescape(cleaned_text)
            
            if cleaned_text:
                artistfacts.append(cleaned_text)

    # Join all facts into a single string separated by double newlines
    return "\n\n".join(artistfacts)

# Run extraction
artistfacts = extract_artist_facts(html_content)
print(artistfacts)