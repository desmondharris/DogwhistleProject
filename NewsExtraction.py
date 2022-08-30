from bs4 import BeautifulSoup
import requests


def cnn_extract(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.text, "html.parser")
    top_paragraph = soup.find('p',  class_="zn-body__paragraph speakable").getText()
    body_html = soup.find_all('div', class_="zn-body__paragraph")
    body_text = ""
    for i in body_html:
        body_text += i.getText()
    top_paragraph += body_text
    return top_paragraph


def fox_extract(link):
    page = requests.get(link)

