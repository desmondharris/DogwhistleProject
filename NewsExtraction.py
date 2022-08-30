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
    soup = BeautifulSoup(page.text, "html.parser")
    paragraphs = soup.select('.article-body p')
    body_text = ""
    for i in paragraphs:
        body_text += "\n" + i.getText()
    return body_text


def cnbc_extract(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.text, "html.parser")
    body_html = soup.select('.group p')
    body_text = ""
    for i in body_html:
        body_text += "\n"+i.getText()
    return body_text


def nbc_extract(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.text, "html.parser")
    body_html = soup.select('.article-body__content p')
    body_text = ""
    for i in body_html:
        body_text += "\n" + i.getText()
    return body_text

