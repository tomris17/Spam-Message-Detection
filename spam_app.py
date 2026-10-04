import joblib
import streamlit as st

st.set_page_config(page_title="Spam Message Detection App", layout="centered")

st.title("Spam Message Detection App")
st.write(
    "Bu uygulama, Doğal Dil İşleme (NLP) ve Multinomial Naive Bayes modeli kullanarak girilen bir mesajın spam olup olmadığını tespit eder."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("spam_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

st.subheader("Mesajinizi Giriniz:")
user_message = st.text_area("Mesaj Metni", "Congratulations! You have won a free ticket. Call now!")

if st.button("Siniflandir", type="primary"):
    if user_message.strip() == "":
        st.warning("Lutfen gecerli bir mesaj metni giriniz.")
    else:
        try:
            # Metni modelin egitildigi TF-IDF vektorlerine donusturme
            message_vec = vectorizer.transform([user_message])
            prediction = model.predict(message_vec)
            
            result = "Spam Mesaj" if prediction[0] == 1 else "Normal Mesaj (Ham)"
            st.success(f"Analiz Sonucu: **{result}**")
        except Exception as e:
            st.error(f"Tahmin sirasinda bir hata olustu: {e}")