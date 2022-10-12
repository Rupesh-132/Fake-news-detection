import streamlit as st
import pickle
import requests
from streamlit_lottie import st_lottie
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

# ---- LOAD ASSETS ----
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_iv4dsx3q.json")
lottie_news = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_KLaN10ftkY.json")
lottie_scanner = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_qQRyVB.json")


# Header section
with st.container():

    st.subheader("Hi, We Are Coders_Grid_4.0:wave:")
    st.title("We are here to help you in classifying the news as Fake or Real")
    st.write(
        "Don't worry with the fake news.............."
    )

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st_lottie(lottie_coding, height=300, key="coding")
    with right_column:
        st_lottie(lottie_news, height=300, key="news")
    st_lottie(lottie_scanner, height=300, key="scanner")
    st.write("---")




ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
       y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizerrupesh.pkl','rb'))
model = pickle.load(open('modelrupesh.pkl','rb'))
st.title("Fake News Detector")
input_news = st.text_area("Enter the news")


if st.button('Predict'):
    # 1. preprocess
    transformed_news = transform_text(input_news)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_news])
    # 3. predict
    result = model.predict(vector_input)[0]
   # st.header(result)
    # 4. Display
    if result == 1:
        st.header(":neutral_face:Fake")
    else:
        st.header("Not Fake")







# 2.vectorize
# 3.Predict
# 4.Display

