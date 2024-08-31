# Text Classification

Using GPT-2 model to solve Text Classification problem. Then, deploy this to the website by using streamlit method.

### What is Text Classification?

Text classification is a machine learning technique that assigns a set of predefined categories to open-ended text. Text classifiers can be used to organize, structure, and categorize pretty much any kind of text – from documents, medical studies and files, and all over the web.

Text classification is one of the fundamental tasks in natural language processing with broad applications such as sentiment analysis, topic labeling, spam detection, and intent detection.

### How about the dataset is used?

Data is in CSV format and has two columns: text and category. It contains 2226 different texts, each labeled under one in 5 categories: entertainment, sport, tech, business, and politics.

![Screenshot 2024-07-24 204249](https://github.com/user-attachments/assets/6a5ae1e7-57bf-4f8d-aab7-75b3b09f049b)

![Screenshot 2024-07-24 204142](https://github.com/user-attachments/assets/e9810ef0-48db-490f-b26b-65c49e45ec9c)

### How do I partition data to train, validate, and test?

Divide the dataset into 3 parts:
- df_train: training sessions, containing 80% of the data
- df_val: validation set, which contains the next 10% of the data
- df_test: test set, which contains the remaining 10% of the data

![Screenshot 2024-07-24 204357](https://github.com/user-attachments/assets/498b1b1e-31ad-4945-bf34-a4703261cbeb)

### Accuracy

The model has an accuracy of around 91%.

![Screenshot 2024-07-24 204506](https://github.com/user-attachments/assets/799c7d26-301e-428d-a89d-ef7d1bcafdc0)

### Deploy the model on the website

By using the model trained, this website allows you to use it as a tool to classify texts into one of five categories: entertainment, sport, tech, business, and politics.

Paste a paragraph of an article that you need to classify in the left area, press `Predict` button and the result will be displayed in the right area.

![snapedit_1721828995359](https://github.com/user-attachments/assets/76364e53-abfa-462e-9bde-488d72e05ff5)

Thanks to [Dao Thi Ngoc Giau](https://github.com/Giau39) - my friend assisted me in finishing this assignment.

For more information, you can also read the report (Vietnamese version) to understand this session.
