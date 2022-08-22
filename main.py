from transformers import pipeline
import streamlit as st
import time


classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")


@st.experimental_memo
def predict(text):
    result = classifier(text)
    return result[0]['label']


def main():
    st.title("Banki RU AI Review")
    review = st.text_input("Your review")
    pred = ""

    if st.button("To Analyse"):
        pred = predict(review)
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            # Update the progress bar with each iteration.
            bar.progress(i + 1)
            time.sleep(0.1)

    st.success(pred)


if __name__ == "__main__":
    main()




