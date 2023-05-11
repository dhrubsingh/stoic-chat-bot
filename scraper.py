import requests
from bs4 import BeautifulSoup


"""

url = "https://www.gutenberg.org/cache/epub/2680/pg2680-images.html"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all the div elements that have a class attribute with the value "chapter"
chapter_divs = soup.find_all("div", {"class": "chapter"})


with open("meditations.txt", "w") as f:
    # Loop through the chapter divs and write their text to the file
    for chapter_div in chapter_divs:
        f.write(chapter_div.get_text())
        f.write("\n")

"""

url = "https://standardebooks.org/ebooks/epictetus/discourses/george-long/text/single-page#book-1"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Open the output file in write mode
with open('discourses.txt', 'w') as f:
    # Loop through each book section and extract the paragraph contents
    for i in range(1, 5): # up to book-4
        book_section = soup.find('section', {'id': f'book-{i}'})
        paragraphs = book_section.find_all('p')
        
        # Write the contents of each paragraph to the output file
        for paragraph in paragraphs:
            f.write(f"Book {i}: {paragraph.text}\n")