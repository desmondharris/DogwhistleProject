from bs4 import BeautifulSoup
import urllib.request


def cnn_extract(link):
    page = urllib.request.urlopen(link)
    soup = BeautifulSoup(page, "html.parser")
    all_paragraphs = soup.select("div.zn-body__paragraph")
    text = ""
    for i in all_paragraphs:
        text += i.getText() + "  \n"
    if text is None:
        raise Exception('Error parsing HTML')
    return text


def fox_extract(link):
    page = urllib.request.urlopen(link)
    soup = BeautifulSoup(page, "html.parser")
    paragraphs = soup.select('.article-body p')
    body_text = ""
    for i in paragraphs:
        body_text += i.getText() + "\n"
    return body_text


def cnbc_extract(link):
    page = urllib.request.urlopen(link)
    soup = BeautifulSoup(page, "html.parser")
    body_html = soup.select('.group p')
    body_text = ""
    for i in body_html:
        body_text += i.getText() + "\n"
    return body_text


def nbc_extract(link):
    page = urllib.request.urlopen(link)
    soup = BeautifulSoup(page, "html.parser")
    body_html = soup.select('.article-body__content p')
    body_text = ""
    for i in body_html:
        body_text += i.getText() + "\n"
    return body_text

