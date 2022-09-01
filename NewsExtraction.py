from bs4 import BeautifulSoup
import requests


def cnn_extract(link):
    found = False
    while not found:
        page = requests.get(link)
        if page is not None:
            found = True
    soup = BeautifulSoup(page.text, "html.parser")
    # This was added because I ran into an error where seemingly randomly, I would get back a different soup object, and
    # parsing it with the code in the try block would return None.
    try:
        body_text = soup.find('p', class_="zn-body__paragraph speakable").getText()
        body_html = soup.find_all('p', class_="zn-body__paragraph" )
    except AttributeError:
        body_html = soup.select('.article__content p')
        body_text = ""
    for i in body_html:
        body_text += i.getText()
    return body_text


def fox_extract(link):
    found = False
    while not found:
        page = requests.get(link)
        if page is not None:
            found = True
    soup = BeautifulSoup(page.text, "html.parser")
    paragraphs = soup.select('.article-body p')
    body_text = ""
    for i in paragraphs:
        body_text += "\n" + i.getText()
    return body_text


def cnbc_extract(link):
    found = False
    while not found:
        page = requests.get(link)
        if page is not None:
            found = True
    soup = BeautifulSoup(page.text, "html.parser")
    body_html = soup.select('.group p')
    body_text = ""
    for i in body_html:
        body_text += "\n"+i.getText()
    return body_text


def nbc_extract(link):
    found = False
    while not found:
        page = requests.get(link)
        if page is not None:
            found = True
    soup = BeautifulSoup(page.text, "html.parser")
    body_html = soup.select('.article-body__content p')
    body_text = ""
    for i in body_html:
        body_text += "\n" + i.getText()
    return body_text

