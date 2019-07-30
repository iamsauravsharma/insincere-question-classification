from zipfile import ZipFile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
import nltk
import re
from nltk.tokenize import word_tokenize


# extract zip file
def extract_zip(file):
    with ZipFile(file, "r") as zip:
        zip.extractall()
        print("Done extracting", file)


# remove zip file after successfully extraction of file
def remove_zip(file):
    os.remove(file)
    print("Successfully deleted ", file)


zips = ["train.csv.zip"]
for zip in zips:
    extract_zip(zip)
    remove_zip(zip)

# read csv file
train = pd.read_csv("train.csv")

# count out a target number
target_count = train["target"].value_counts()

# bar plot a target distribution
plt.figure(figsize=(8, 6))
ax = sns.barplot(target_count.index, target_count.values)
ax.set_title("Question distribution of train dataset")
ax.set_xlabel("Question type")
ax.set_ylabel("No of questions")

# split question to count out a question length
train["quest_len"] = train["question_text"].apply(lambda x: len(x.split()))

# create list of question insincere and sincere separately
sincere = train[train["target"] == 0]
insincere = train[train["target"] == 1]

# Plot question length distribution of both type of questions in single graph
plt.figure(figsize=(15, 8))
sns.distplot(sincere["quest_len"], hist=True, label="sincere")
sns.distplot(insincere["quest_len"], hist=True, label="insincere")
plt.legend()
plt.xlabel("Question length")
plt.title("Questions Length Distribution of both question")

# Plot question length distribution of sincere questions
plt.figure(figsize=(15, 8))
sns.distplot(sincere["quest_len"], hist=True, label="sincere")
plt.legend()
plt.xlabel("Question length")
plt.title("Questions Length Distribution of sincere question")

# Plot question length distribution of insincere questions
plt.figure(figsize=(15, 8))
sns.distplot(insincere["quest_len"], hist=True, label="insincere")
plt.legend()
plt.xlabel("Question length")
plt.title("Questions Length Distribution of insincere question")

# pre processing

puncts = [
    ",",
    ".",
    '"',
    ":",
    ")",
    "(",
    "-",
    "!",
    "?",
    "|",
    ";",
    "'",
    "$",
    "&",
    "/",
    "[",
    "]",
    ">",
    "%",
    "=",
    "#",
    "*",
    "+",
    "\\",
    "•",
    "~",
    "@",
    "£",
    "·",
    "_",
    "{",
    "}",
    "©",
    "^",
    "®",
    "`",
    "<",
    "→",
    "°",
    "€",
    "™",
    "›",
    "♥",
    "←",
    "×",
    "§",
    "″",
    "′",
    "Â",
    "█",
    "½",
    "à",
    "…",
    "“",
    "★",
    "”",
    "–",
    "●",
    "â",
    "►",
    "−",
    "¢",
    "²",
    "¬",
    "░",
    "¶",
    "↑",
    "±",
    "¿",
    "▾",
    "═",
    "¦",
    "║",
    "―",
    "¥",
    "▓",
    "—",
    "‹",
    "─",
    "▒",
    "：",
    "¼",
    "⊕",
    "▼",
    "▪",
    "†",
    "■",
    "’",
    "▀",
    "¨",
    "▄",
    "♫",
    "☆",
    "é",
    "¯",
    "♦",
    "¤",
    "▲",
    "è",
    "¸",
    "¾",
    "Ã",
    "⋅",
    "‘",
    "∞",
    "∙",
    "）",
    "↓",
    "、",
    "│",
    "（",
    "»",
    "，",
    "♪",
    "╩",
    "╚",
    "³",
    "・",
    "╦",
    "╣",
    "╔",
    "╗",
    "▬",
    "❤",
    "ï",
    "Ø",
    "¹",
    "≤",
    "‡",
    "√",
]


# clean some punctuation
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, "")
    return x


one_letter_word = [
    " a ",
    " b ",
    " c ",
    " d ",
    " e ",
    " f ",
    " g ",
    " h ",
    " i ",
    " j ",
    " k ",
    " l ",
    " m ",
    " n ",
    " o ",
    " p ",
    " q ",
    " r ",
    " s ",
    " t ",
    " u ",
    " v ",
    " w ",
    " x ",
    " y ",
    " z ",
]


# clean one letter words
def clean_one_letter_word(x):
    x = str(x)
    for punct in one_letter_word:
        x = x.replace(punct, "")
    return x


# clean numbers
def clean_numbers(x):
    x = re.sub("[0-9]{5,}", "", x)
    x = re.sub("[0-9]{4}", "", x)
    x = re.sub("[0-9]{3}", "", x)
    x = re.sub("[0-9]{2}", "", x)
    return x


mispell_dict = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "colour": "color",
    "centre": "center",
    "favourite": "favorite",
    "travelling": "traveling",
    "counselling": "counseling",
    "theatre": "theater",
    "cancelled": "canceled",
    "labour": "labor",
    "organisation": "organization",
    "wwii": "world war 2",
    "citicise": "criticize",
    "youtu ": "youtube ",
    "Qoura": "Quora",
    "sallary": "salary",
    "Whta": "What",
    "narcisist": "narcissist",
    "howdo": "how do",
    "whatare": "what are",
    "howcan": "how can",
    "howmuch": "how much",
    "howmany": "how many",
    "whydo": "why do",
    "doI": "do I",
    "theBest": "the best",
    "howdoes": "how does",
    "mastrubation": "masturbation",
    "mastrubate": "masturbate",
    "mastrubating": "masturbating",
    "pennis": "penis",
    "Etherium": "Ethereum",
    "narcissit": "narcissist",
    "bigdata": "big data",
    "2k17": "2017",
    "2k18": "2018",
    "qouta": "quota",
    "exboyfriend": "ex boyfriend",
    "airhostess": "air hostess",
    "whst": "what",
    "watsapp": "whatsapp",
    "demonitisation": "demonetization",
    "demonitization": "demonetization",
    "demonetisation": "demonetization",
}


def _get_mispell(mispell_dict):
    mispell_re = re.compile("(%s)" % "|".join(mispell_dict.keys()))
    return mispell_dict, mispell_re


# replace a misspell text
mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


# Pre process a text
def preprocess(text):
    text = text.lower()
    text = clean_text(text)
    text = clean_one_letter_word(text)
    text = clean_numbers(text)
    text = replace_typical_misspell(text)
    return text


# preprocess a text
tqdm.pandas()
sincere["question_text"] = sincere["question_text"].progress_apply(preprocess)
insincere["question_text"] = insincere["question_text"].progress_apply(preprocess)


# define a word cloud print function
def create_wordcloud(data, title):
    nltk.download("punkt")
    question_text = data.question_text.str.cat(
        sep=" "
    )  # function to split text into word
    tokens = word_tokenize(question_text)
    vocabulary = set(tokens)
    print(len(vocabulary))
    stop_words = set(STOPWORDS)
    tokens = [w for w in tokens if w not in stop_words]
    frequency_dist = nltk.FreqDist(tokens)
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stop_words,
        min_font_size=10,
    ).generate_from_frequencies(frequency_dist)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")
    plt.show()


# create insincere question word cloud
create_wordcloud(insincere, "Insincere question word cloud")

# create sincere question word cloud
create_wordcloud(sincere, "Sincere question word cloud")
