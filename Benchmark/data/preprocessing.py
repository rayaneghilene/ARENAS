import re

def clean_text(text):
    if type(text) != str:
        return ""

    temp = text.lower()
    temp = re.sub("'", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub(r'www\S+', '', temp)
    temp = re.sub('[()!?]', '', temp)
    temp = re.sub('\[.*?\]','', temp)
    temp = re.sub("[^a-z]"," ", temp)
    temp = re.sub("rt","", temp)
    temp = re.sub("url","", temp)
    temp = re.sub(" +"," ", temp)

    return temp.strip()


def relabel(label, map):
    return map[label]