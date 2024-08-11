import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# To make sure there are no file encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.set_page_config(
    page_title = "Cassava Leaf Disease Prediction",
    layout = "centered",
    page_icon= ":four_leaf_clover:"
    )

    menu = ['Home', 'About', 'Contact']
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.title('🍀 Cassava Leaf Disease Prediction')

        st.subheader("This app predicts diseases affecting cassava leaves.")
        st.write("")

        uploaded_file = st.file_uploader("Choose a leaf image", type = ["jpg", "jpeg", "png"])

        if st.button("Predict"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                
                st.write("")
                st.write("Uploaded image")

                st.image(image, use_column_width = True)
                st.write("")

                try:
                    with st.spinner("**Predicting...**"):

                        np.set_printoptions(suppress = True)
    
                        model = tensorflow.keras.models.load_model('model/EffNetB0_512_16.h5')

                        image = image.resize((512, 512))
                        image = np.expand_dims(image, axis = 0)

                        labels = {0: "Cassava Bacterial Blight (CBB)", 1: "Cassava Brown Streak Disease (CBSD)", 2: "Cassava Green Mottle (CGM)", 3: "Cassava Mosaic Disease (CMD)", 4: "Healthy"}
                        
                        predictions = np.argmax(model.predict(image))

                        print(labels[predictions])

                        label = labels[predictions]

                    if(label == 'Healthy'):
                        st.success("Above leaf image is predicted to be **" + label + "**.")
                    
                    else:
                        st.success("Above leaf image is predicted to be affected by **" + label + "**.")
                        st.write("")

                        if(label == 'Cassava Bacterial Blight (CBB)'):
                            st.write("The pathogen *Xanthomonas axonopodis pv. Manihotis* causes **Cassava Bacterial Blight (CBB)**, a bacterial disease affecting cassava. \
                                Bacterial blight is the disease that causes the most yield losses in cassava across the world.")
                        
                        elif(label == 'Cassava Brown Streak Disease (CBSD)'):
                            st.write("**Cassava Brown Streak Disease (CBSD)** is a viral disease caused by two different ipomovirus species, Cassava Brown Streak Virus (CBSV) and Ugandan Cassava Brown Streak Virus (UCBSV), both of which belong to the *Potyviridae* family.")
                        
                        elif(label == 'Cassava Green Mottle (CGM)'):
                            st.write("**Cassava Green Mottle (CGM)** is a viral disease caused by Cassava Green Mottle Virus (CGMV), which belongs to the *Secoviridae* family.")
                        
                        elif(label == 'Cassava Mosaic Disease (CMD)'):
                            st.write("**Cassava Mosaic Disease (CMD)** is a viral disease caused by Cassava Mosaic Virus (CMV), a generic term for many virus species in the *Geminiviridae* family.")

                except:
                    st.error("Apologies! Something went wrong! 🙇🏽‍♂️")
            else:
                st.error("Could you please upload an image? 🙇🏽‍♂️")

    elif choice == "About":
        st.title("About the project")
        st.write("")

        cassava_with_leaves = Image.open("./assets/cassava_with_leaves.jpg")
        st.image(cassava_with_leaves, use_column_width = True, caption = "Cassava roots with leaves.")

        st.write("")
        st.write("As the second-largest provider of carbohydrates in Africa, **Cassava** (*Manihot esculenta*) is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. \
            At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. \
            Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. \
            This suffers from being labour-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth. \
            With the help of data science, it may be possible to identify common diseases so they can be treated.")

        st.write("")
        st.write("This application is aimed at classifying a cassava leaf image, using a [**Kaggle Research Code Competition**](https://www.kaggle.com/c/cassava-leaf-disease-classification) dataset consisting of 21,397 labeled images collected during a regular survey in Uganda, into four disease categories (**Cassava Bacterial Blight**, **Cassava Brown Streak Disease**, **Cassava Green Mottle** and **Cassava Mosaic Disease**) or a fifth category indicating a **Healthy** leaf. \
            Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. \
            This is in a format that most realistically represents what farmers would need to diagnose in real life.")

        st.write("")
        st.write("For this, [**EfficientNet-B0**](https://arxiv.org/abs/1905.11946), a state-of-the-art convolutional neural network model, was trained using **Transfer Learning**. \
            EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. \
            Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.")

        st.write("")
        efficientnetb0_architecture = Image.open("./assets/efficientnetb0-architecture.png")
        st.image(efficientnetb0_architecture, use_column_width = True, caption = "EfficientNet-B0 baseline network")

        st.write("")
        st.write("The trained model was also made as a submission to [**Cassava Leaf Disease Classification**](https://www.kaggle.com/c/cassava-leaf-disease-classification/), the Kaggle Research Code Competition. \
        The model achieved an overall **accuracy score** of **0.8887** and **0.8851** on the **private** and **public competition leaderboard** respectively, indicating the efficiency of the approach.")

    elif choice == "Contact":

        st.title("Hi there 👋, I'm Srinath K R")
    
        st.write("")
        st.write("I'm a student of School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India.")

        st.write("")
        st.write("Feel free to reach out to me through: [LinkedIn](https://www.linkedin.com/in/srinathkr) | [Email](mailto:k.r.srinath07@gmail.com)")
        st.write("Check out my other works: [GitHub](https://github.com/srinathkr07)")

        st.write("----")

if __name__ == "__main__":
    main()