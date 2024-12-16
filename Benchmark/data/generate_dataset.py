import pandas as pd
from preprocessing import clean_text, relabel
from ast import literal_eval


def get_davidson():
    davidson_df = pd.read_csv("1 Davidson et al/data/labeled_data.csv")

    davidson_df.drop(davidson_df.columns.difference(['tweet', 'class']), 1, inplace=True)

    davidson_df['text'] = davidson_df['tweet'].apply(lambda x: clean_text(x))
    davidson_df['class'] = davidson_df['class'].apply(lambda x: relabel(x, {0: "hate", 1: "offensive", 2: "neither"}))

    davidson_df = davidson_df[['text', 'class']]

    davidson_df = davidson_df[davidson_df['text'] != ""]
    davidson_df = davidson_df[davidson_df['text'].notna()]

    return davidson_df


def get_gab_reddit(dataset = "gab"):
    if dataset == "gab":
        origina_df = pd.read_csv("3 4 Reddit and Gab/data/gab.csv")

    elif dataset == "reddit":
        origina_df = pd.read_csv("3 4 Reddit and Gab/data/reddit.csv")

    else:
        return

    origina_df.drop(origina_df.columns.difference(['text', 'hate_speech_idx']), 1, inplace=True)

    output_df = pd.DataFrame(columns = ['text', 'class'])

    for _, line in origina_df.iterrows():

        post = line['text'].split("\n")[:-1]

        for text in post:

            if(text == ""):
                continue

            if(not text[0].isdigit()):
                continue

            text_id = int(text[0])

            output_df = output_df.append({'text': clean_text(text), 'class': 'neither'}, ignore_index = True)

            if(type(line['hate_speech_idx']) != str):
                continue

            for id in literal_eval(line['hate_speech_idx']):

                if(int(id) == text_id):
                    output_df.loc[output_df.index[-1], 'class'] = "hate"

    output_df = output_df[output_df['text'] != ""]
    output_df = output_df[output_df['text'].notna()]

    return output_df


def get_fox():
    fox_df = pd.read_json("5 Fox News/full-comments-u.json", lines=True)

    fox_df.drop(fox_df.columns.difference(['text', 'label']), 1, inplace=True)

    fox_df['text'] = fox_df['text'].apply(lambda x: clean_text(x))
    fox_df['class'] = fox_df['label'].apply(lambda x: relabel(x, {0: "neither", 1: "hate"}))

    fox_df = fox_df[['text', 'class']]

    fox_df = fox_df[fox_df['text'] != ""]
    fox_df = fox_df[fox_df['text'].notna()]

    return fox_df


def get_hasoc2019(dataset = "train"):
    if dataset == "train":
        hasoc_df = pd.read_table("6 HASOC 2019/english_dataset/english_dataset.tsv")

    elif dataset == "test":
        hasoc_df = pd.read_table("6 HASOC 2019/english_dataset/hasoc2019_en_test-2919.tsv")

    else:
        return
    
    hasoc_df.drop(hasoc_df.columns.difference(['text', 'task_2']), 1, inplace=True)

    hasoc_df['text'] = hasoc_df['text'].apply(lambda x: clean_text(x))
    hasoc_df['class'] = hasoc_df['task_2'].apply(lambda x: relabel(x, {"NONE": "neither", "HATE": "hate", "PRFN": "profane", "OFFN": "offensive"}))

    hasoc_df = hasoc_df[['text', 'class']]

    hasoc_df = hasoc_df[hasoc_df['text'] != ""]
    hasoc_df = hasoc_df[hasoc_df['text'].notna()]

    return hasoc_df


def get_hasoc2020(dataset = "train"):
    if dataset == "train":
        hasoc_df = pd.read_excel("14 HASOC 2020/hasoc_2020_en_train_new.xlsx")

    elif dataset == "test":
        hasoc_df = pd.read_excel("14 HASOC 2020/hasoc_2020_en_test_new.xlsx")

    else:
        return
    
    hasoc_df.drop(hasoc_df.columns.difference(['text', 'task2']), 1, inplace=True)

    hasoc_df['text'] = hasoc_df['text'].apply(lambda x: clean_text(x))
    hasoc_df['class'] = hasoc_df['task2'].apply(lambda x: relabel(x, {"NONE": "neither", "HATE": "hate", "PRFN": "profane", "OFFN": "offensive"}))

    hasoc_df = hasoc_df[hasoc_df['text'] != ""]
    hasoc_df = hasoc_df[hasoc_df['text'].notna()]

    hasoc_df = hasoc_df[['text', 'class']]

    return hasoc_df


def get_stormfront():
    original_df = pd.read_csv("7 Stormfront/annotations_metadata.csv")

    original_df.drop(original_df.columns.difference(['file_id', 'label']), 1, inplace=True)

    output_df = pd.DataFrame(columns = ['text', 'class'])

    for _, line in original_df.iterrows():

        if line['label'] == "idk/skip" or line['label'] == "relation":
            continue

        with open(f"7 Stormfront/all_files/{line['file_id']}.txt", encoding="utf8") as file:

            text = file.read()

            output_df = output_df.append({'text': clean_text(text), 'class': relabel(line['label'], {"hate": "hate", "noHate": "neither"})}, ignore_index = True)

    output_df = output_df[output_df['text'] != ""]
    output_df = output_df[output_df['text'].notna()]

    return output_df


def get_hateval(dataset = "train"):
    if dataset == "train":
        hateval_df = pd.read_csv("8 Hateval 2019/hateval2019_en_train.csv")

    elif dataset == "test":
        hateval_df = pd.read_csv("8 Hateval 2019/hateval2019_en_test.csv")

    elif dataset == "dev":
        hateval_df = pd.read_csv("8 Hateval 2019/hateval2019_en_dev.csv")

    else:
        return

    hateval_df.drop(hateval_df.columns.difference(['text', 'HS']), 1, inplace=True)

    hateval_df['text'] = hateval_df['text'].apply(lambda x: clean_text(x))
    hateval_df['class'] = hateval_df['HS'].apply(lambda x: relabel(x, {0: "neither", 1: "hate"}))

    hateval_df = hateval_df[hateval_df['text'] != ""]
    hateval_df = hateval_df[hateval_df['text'].notna()]

    hateval_df = hateval_df[['text', 'class']]

    return hateval_df


def get_grimminger(dataset = "train"):
    if dataset == "train":
        grimminger_df = pd.read_table("9 Grimminger and Klinger/train.tsv")

    elif dataset == "test":
        grimminger_df = pd.read_table("9 Grimminger and Klinger/test.tsv")

    else:
        return

    grimminger_df.drop(grimminger_df.columns.difference(['text', 'HOF']), 1, inplace=True)

    grimminger_df['text'] = grimminger_df['text'].apply(lambda x: clean_text(x))
    grimminger_df['class'] = grimminger_df['HOF'].apply(lambda x: relabel(x, {"Non-Hateful": "neither", "Hateful": "hate"}))

    grimminger_df = grimminger_df[grimminger_df['text'] != ""]
    grimminger_df = grimminger_df[grimminger_df['text'].notna()]

    grimminger_df = grimminger_df[['text', 'class']]

    return grimminger_df


def get_trac():
    trac_df1 = pd.read_csv("13 TRAC/trac-1/english/agr_en_dev.csv")
    trac_df2 = pd.read_csv("13 TRAC/trac-1/english/agr_en_train.csv")

    trac_df = pd.concat([trac_df1, trac_df2])

    trac_df.drop(trac_df.columns.difference(['text', 'class']), 1, inplace=True)

    trac_df['text'] = trac_df['text'].apply(lambda x: clean_text(x))
    trac_df['class'] = trac_df['class'].apply(lambda x: relabel(x, {"OAG": "aggressive", "CAG": "aggressive", "NAG": "neither"}))

    trac_df = trac_df[trac_df['text'] != ""]
    trac_df = trac_df[trac_df['text'].notna()]

    return trac_df


def get_founta():
    founta_df = pd.read_csv("12 Founta/hatespeech_text_label_vote_RESTRICTED_100K.csv", sep = "	")

    founta_df.drop(founta_df.columns.difference(['Tweet text', 'Label']), 1, inplace=True)

    founta_df['text'] = founta_df['Tweet text'].apply(lambda x: clean_text(x))
    founta_df['class'] = founta_df['Label'].apply(lambda x: relabel(x, {"hateful": "hate", "abusive": "abusive", "spam": "neither", "normal": "neither"}))

    founta_df = founta_df[founta_df['text'].notna()]
    founta_df = founta_df[founta_df['text'] != ""]

    founta_df = founta_df[['text', 'class']]

    return founta_df


def get_olid(dataset = "train"):
    if dataset == "train":
        olid_df = pd.read_table("10 OLID/olid-training-v1.0.tsv")

    elif dataset == "test":

        olid_df = pd.read_table("10 OLID/testset-levela.tsv")

        olid_df["subtask_a"] = ""

        labels_df = pd.read_csv("10 OLID/labels-levela.csv")

        for index, line in olid_df.iterrows():

            if line["id"] != labels_df["id"][index]:
                raise Exception("Files are not in a consistent order")

            olid_df["subtask_a"][index] = labels_df["subtask_a"][index]

    else:
        return

    olid_df.drop(olid_df.columns.difference(['tweet', 'subtask_a']), 1, inplace=True)

    olid_df['text'] = olid_df['tweet'].apply(lambda x: clean_text(x))
    olid_df['class'] = olid_df['subtask_a'].apply(lambda x: relabel(x, {"OFF": "offensive", "NOT": "neither"}))

    olid_df = olid_df[olid_df['text'].notna()]
    olid_df = olid_df[olid_df['text'] != ""]

    olid_df = olid_df[['text', 'class']]

    return olid_df


def get_jigsaw(dataset = "train"):
    class_preference_map = ["threat", "severe_toxic", "identity_hate", "insult", "obscene", "toxic"]

    if dataset == "train":
        jigsaw_df = pd.read_csv("11 Jigsaw Toxic Comment Classification Challenge/train.csv")

        jigsaw_df["class"] = "neither"

        for index, line in jigsaw_df.iterrows():

            for c in class_preference_map:

                if line[c] == 1:
                    jigsaw_df["class"][index] = c
                    break

    elif dataset == "test":
        jigsaw_df = pd.read_csv("11 Jigsaw Toxic Comment Classification Challenge/test.csv")
        labels_df = pd.read_csv("11 Jigsaw Toxic Comment Classification Challenge/test_labels.csv")

        jigsaw_df["class"] = "unlabeled"

        for index, line in labels_df.iterrows():

            if line["id"] != jigsaw_df["id"][index]:
                raise Exception("Files are not in a consistent order")
            
            if line["toxic"] == -1:
                continue

            for c in class_preference_map:
                if line[c] == 1:
                    jigsaw_df["class"][index] = c
                    break

                if c == "toxic":
                    jigsaw_df["class"][index] = "neither"

        jigsaw_df.drop(jigsaw_df[jigsaw_df["class"] == "unlabeled"].index, inplace=True)


    jigsaw_df.drop(jigsaw_df.columns.difference(['comment_text', 'class']), 1, inplace=True)

    jigsaw_df['text'] = jigsaw_df['comment_text'].apply(lambda x: clean_text(x))

    jigsaw_df = jigsaw_df[jigsaw_df['text'].notna()]
    jigsaw_df = jigsaw_df[jigsaw_df['text'] != ""]

    jigsaw_df = jigsaw_df[['text', 'class']]

    return jigsaw_df