import requests
from bs4 import BeautifulSoup
import time
import datetime

i = 0
j = 0
k = 0
p = 1

f = open('captions.txt', 'w')

date_ref = (time.strptime("December 1, 2014", "%B %d, %Y"))
date_raw = []
title_suffix = []
album_url = []
caption_duplicate_ref = ''

while i < 26:
    if i == 0:
        url = "http://www.newyorksocialdiary.com/party-pictures"
        response = requests.get(url)
        soup = BeautifulSoup(response.text)
        title_div = soup.find_all('span', attrs={'class': 'views-field views-field-title'})
        date_div = soup.find_all('span', attrs={'class': 'views-field views-field-created'})
        for date_divs in date_div:
            date_span = date_divs('span', attrs={'class': 'field-content'})
            date_raw.append(time.strptime(str(date_span[0].text), "%A, %B %d, %Y"))
        for title_divs in title_div:
            title_span = title_divs('a')
            title_suffix.append(str(title_span[0]['href']))
        i += 1
    elif i > 0:
        url = "http://www.newyorksocialdiary.com/party-pictures" + "?page=" + str(i)
        response = requests.get(url)
        soup = BeautifulSoup(response.text)
        title_div = soup.find_all('span', attrs={'class': 'views-field views-field-title'})
        date_div = soup.find_all('span', attrs={'class': 'views-field views-field-created'})
        for date_divs in date_div:
            date_span = date_divs('span', attrs={'class': 'field-content'})
            date_raw.append(time.strptime(str(date_span[0].text), "%A, %B %d, %Y"))
        for title_divs in title_div:
            title_span = title_divs('a')
            title_suffix.append(str(title_span[0]['href']))
        i += 1

while j < len(date_raw):
    if date_raw[j] >= date_ref:
        j += 1
    else:
        album_url.append("http://www.newyorksocialdiary.com" + title_suffix[j])
        j += 1

while k < len(album_url):
    url = album_url[k]
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    caption_div_1 = soup.find_all('div', attrs={'class': 'photocaption'})
    caption_div_2 = soup.find_all('font', attrs={'size': '1', 'face': 'Verdana, Arial, Helvetica, sans-serif'})
    caption_div_3 = soup.find_all('td', attrs={'valign': 'top', 'class': 'photocaption'})
    for caption_divs in caption_div_1:
        caption_divs = (caption_divs.text).strip()
        if len(caption_divs) == 0:
            pass
        else:
            if caption_divs == caption_duplicate_ref:
                pass
            else:
                #f.write(str(p) + '     ' + (caption_divs).encode('utf-8') + '     div_1' + '\n')
                f.write((caption_divs).encode('utf-8') + '\n')
                caption_duplicate_ref = caption_divs
            #p += 1
    for caption_divs in caption_div_2:
        caption_divs = (caption_divs.text).strip()
        if len(caption_divs) == 0:
            pass
        else:
            if caption_divs == caption_duplicate_ref:
                pass
            else:
                #f.write(str(p) + '     ' + (caption_divs).encode('utf-8') + '     div_2' + '\n')
                f.write((caption_divs).encode('utf-8') + '\n')
                caption_duplicate_ref = caption_divs
            #p += 1
    for caption_divs in caption_div_3:
        caption_divs = (caption_divs.text).strip()
        if len(caption_divs) == 0:
            pass
        else:
            if caption_divs == caption_duplicate_ref:
                pass
            else:
                #f.write(str(p) + '     ' + (caption_divs).encode('utf-8') + '     div_3' + '\n')
                f.write((caption_divs).encode('utf-8') + '\n')
                caption_duplicate_ref = caption_divs
            #p += 1
    k += 1

f.close()
