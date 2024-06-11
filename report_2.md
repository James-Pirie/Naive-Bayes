# 361 Assignment 2 - James Pirie

## Data Representation and Pre-Processing
For how words are represneted, I decided it would be best to count frequency, rather than purely a binary exists or does not exist. I believe that frequency would allow me to get a lot more information from the data. To store the frequency in my program, originaly I calculated the vocabulary, then the probability per class, then the bayes score for every word, then stored that so I could preform processess, but in practice this was a very slow method and took about 40 minutes simply to pre-processe. I then decided on a new method, first, the vocabulary would be retrieved, then I would generate a dictionary, the keys in the dictionary would be each word in the vocabulary, then each key would have another dict with the keys being every unique class. Each key would associate with the number of times the word appears in each class.

## Naive Bayes Implementation
My implementation first gets all the classes, it then calculates the probability of one class existing based on the proportion it makes up in the data set, n of class rows / total rows. The get_word_score method takes in a word and uses the bayes formula count(w|c) / len(documents of class(c)). Then predict_abstract_class iterates over all words in an abstract using the get_word_score n of words in abstract times for every class in the dataframe. Multiplying the scores together and with the class probability using the logarithm method to avoid underflow, then whichever value for each class is bigger is the prediciton.

## Naive Bayes Improvement

### N-Grams
My first attemp at Naive Bayes extension was with N-Grams. Instead of calculating the vocabulary with each word being a token, I wrote a method to group n words together for every combination found in the data-set. When performing bayes on these the accuracy actually decreased from about 94% on basic naive bayes to about 92%, if n=2. If n was equal to anything else accuracy would drop to less than 60%, which is not ideal. I did not end up keeping Naive Bayes in my final model.

### Data Over Sampling
My secocond attempt cat Naive Bayes extension was by improving the IID of the data-set by oversampling minority classes. To help me with this I imported RandomOverSampler from the imblearn library. I then created a function called over_sample, it takes in a dataframe and splits it into two parts, one with just the abstract and one for the target variable. It then uses RandomOverSampler fit_resample to fit the features to assciated classes to balance it so that there are equal amounts of all 4 classes. Returning as a reconstructed data frame. This method was quite succesful and score around 98% accuracy on my validation, which is explained later, and 98% on Kaggle.

### Stop Word Removal
My third extension I implemented was stop word removal. Stop words are words suchs as "the" or "and", which often don't provide actual information to the bayes classifier, and might mislead it. To achieve this, first I downloaded a file from nltk which contains a large amount of stop words for the English language. I then wrote a method called remove_stop_words, which takes in a string, splits it by blank space and then iterates through it, saving the word only if it is not in the stop words file. For internal testing and validation this did not make much of a difference, but on submission to Kaggle accuracy increased by .3%, so I kept it in my final model.

### Noteable Attempts
- Before Oversampling I attempted to manualy weight classes by adjusting multiplications of class probabilities, this method was less succesful and very prone to optimisation bias, so was not kept.
- I also attempted to weight words by summing their frequencyies from the dictionary and the lower the frequency the higher the weighting, this decreased accuracy so was not kept.

## Validation 
### Method
- For validation I primarily used the accuracy score. Accuracy is measured by number of correctly classified rows divided by total number of rows multiplied by 100, given as a percentage. This is calculate in the method test_training, which takes in two data frames, training data and testing data, it then takes in a dictionary as described above in data-representation and another dictionary which contains pre-processed calculated values for the numnber of total words in each class. The accuracy is then calculated then returned as a fstring: f"Accuracy: {accuracy}%".
- To ensure the best validation I also used k-fold cross validation. I wrote a method called k_fold_cross_validation, which takes in the k, in this case I used 10, and the data as a data frame. The data frame is then split into k equal pieces, and then k times k - 1 splits of the data are concatonated to be used for training, which involves calculating the word_count_dict and number_of_word_per_class_dict dictionary, then test_training is run k times with the testing data being the left over k slice, not used in training.
### Resutls
My extended Naive Bayes model scored between 3 and 5% better on average, getting about 98-99% accuracy on my k-fold testing, and 97.6% accuracy on kaggle. This compared to the original Naive Bayes which scored 91% on Kaggle and about 92-94% on k-fold. These results are shown at the bottom of this document. This clearly shows that my extensions are an improvemnt over base Naive Bayes and, as shown by the K-fold cross validation, quite significant, nearing 99%.


```python
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def get_training_data():
    """Get training data"""
    training_file = 'data/trg.csv'
    return pd.read_csv(training_file)


def get_testing_data():
    """Get testing data"""
    testing_file = 'data/tst.csv'
    return pd.read_csv(testing_file)


def remove_stop_words(text: str):
    """Use nltk database to remove stop words from a row's abstract"""
    with open('data/stopwords/english', 'r') as f:
        custom_stopwords = set(f.read().splitlines())

    # filter out stopwords
    text_split = text.split()
    filtered_words = [word for word in text_split if word.lower() not in custom_stopwords]
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def remove_stop_words_data_frame(data_frame: pd.DataFrame):
    """For every row use remove_stop_words to remove stop words from an entire data frame"""
    for index, row in data_frame.iterrows():
        row['abstract'] = remove_stop_words(row['abstract'])
    return data_frame


def over_sample(data_frame: pd.DataFrame):
    """Over sample minority classes until they are equally sampled."""
    x = data_frame.drop(columns=['class', 'id'])
    y = data_frame['class']

    # split data frame on target class
    over_sampler = RandomOverSampler()
    x_re_sampled, y_re_sampled = over_sampler.fit_resample(x, y)
    re_sampled_df = pd.DataFrame(x_re_sampled, columns=x.columns)
    re_sampled_df['class'] = y_re_sampled

    return re_sampled_df


def get_vocabulary(data_frame: pd.DataFrame):
    """Get all unique words in entire data frame"""
    vocabulary = set()
    for abstract in data_frame['abstract']:
        words = abstract.split()
        vocabulary.update(words)
    return list(vocabulary)


def get_classes(data_frame: pd.DataFrame):
    """Get all possible classes in entire data frame"""
    all_classes = set()
    for class_value in data_frame['class']:
        all_classes.update(class_value)
    return list(all_classes)


def probability_per_class(classes: list, data_frame: pd.DataFrame):
    """Get the probability that a randomly chosen row in the data frame will be of each class"""
    probabilities = {}
    # loop through provided class values
    for class_value in classes:
        current_class_count = 0
        for row in data_frame['class']:
            # count number of times a class appears
            if row == class_value:
                current_class_count += 1
    
        class_probability = current_class_count / data_frame.shape[0]

        probabilities[class_value] = class_probability

    return probabilities


def word_in_data_frame_count(word: str, data_frame: pd.DataFrame):
    """Counts the number of times a word appears in the 'abstract' column of a DataFrame."""
    total_count = 0

    abstracts = data_frame['abstract']
    if word in abstracts.split():
        word_counts = abstracts.apply(lambda x: x.count(word))
        total_count = word_counts.sum()
    return total_count


def get_dictionary(vocabulary: list,  data_frame: pd.DataFrame):
    """Calculate the number of times a word appears in each class, store in dictionary with word as key"""
    final_dict = {}
    for unique_word in vocabulary:
        final_dict[unique_word] = {'E': 0, 'B': 0, 'V': 0, 'A': 0}
    for index, row in data_frame.iterrows():
        for word in row['abstract'].split():
            final_dict[word][row['class']] += 1
    return final_dict


def get_word_score(word: str, word_count_dictionary: dict, class_name: str, word_count_dict: dict):
    """Use bayes formula to calcualte probabulity word appears in a class"""
    # numerator: number of times the word appears in the class with Laplace smoothing
    numerator = word_count_dictionary[word][class_name] + 1
    # denominator: number of words in a class
    denominator = word_count_dict[class_name]
    return np.log(numerator) - np.log(denominator)


def predict_abstract_class(abstract: str, word_count_dictionary: dict, class_probabilities: dict, class_names: list, number_of_word_per_class_dict: dict):
    """For every row in data frame, use bayes formula to calculate most likely class abstract belongs to"""
    score_list = []
    for current_class in class_names:
        bayes_score = np.log(class_probabilities[current_class])
        for word in abstract.split():
            # only calculate if words has been trained on
            if word in word_count_dictionary.keys():
                bayes_score += get_word_score(word, word_count_dictionary, current_class, number_of_word_per_class_dict)
        score_list.append(bayes_score)
    return class_names[score_list.index(max(score_list))]


def test_training(testing_data: pd.DataFrame, training_data: pd.DataFrame, word_count_dict: dict, number_of_word_per_class_dict: dict):
    count_total = 0
    count_correct = 0

    classes = get_classes(training_data)
    class_probabilities = probability_per_class(classes, training_data)

    for i, row in testing_data.iterrows():
        count_total += 1
        if predict_abstract_class(row['abstract'], word_count_dict, class_probabilities, classes, number_of_word_per_class_dict) == row['class']:
            count_correct += 1

    return f"Accuracy: {(count_correct / count_total) * 100}%"

```

## Validation

### Improved Model Validation with 10-Fold Validation


```python

def test_training(testing_data: pd.DataFrame, training_data: pd.DataFrame, word_count_dict: dict, number_of_word_per_class_dict: dict):
    count_total = 0
    count_correct = 0

    classes = get_classes(training_data)
    class_probabilities = probability_per_class(classes, training_data)

    for i, row in testing_data.iterrows():
        count_total += 1
        if predict_abstract_class(row['abstract'], word_count_dict, class_probabilities, classes, number_of_word_per_class_dict) == row['class']:
            count_correct += 1

    return f"Accuracy: {(count_correct / count_total) * 100}%"

def k_fold_cross_validation(k: int, data_input: pd.DataFrame):
    """Divide the dataset into k partitions, """
    k_data_frames = np.array_split(data_input, k)

    for i in range(len(k_data_frames)):
        # use 1/kth of the data for testing
        testing_frame = pd.DataFrame(k_data_frames[i])
        # use the rest of the data for training
        training_frame = pd.concat([frame for j, frame in enumerate(k_data_frames) if j != i])

        vocab = get_vocabulary(training_frame)

        word_count_dict = get_dictionary(vocab, training_frame)

        number_of_word_per_class_dict = {
            "B": sum(word_count_dict[key]["B"] for key in word_count_dict) + len(word_count_dict),
            "A": sum(word_count_dict[key]["A"] for key in word_count_dict) + len(word_count_dict),
            "E": sum(word_count_dict[key]["E"] for key in word_count_dict) + len(word_count_dict),
            "V": sum(word_count_dict[key]["V"] for key in word_count_dict) + len(word_count_dict)}

        print(f"======================================= Test {i + 1} =======================================")
        print(test_training(testing_frame, training_frame, word_count_dict, number_of_word_per_class_dict))


if __name__ == '__main__':
    data = get_training_data()
    # improvments
    data = over_sample(remove_stop_words_data_frame(data))
    data = data.sample(frac=1).reset_index(drop=True)

    k_fold_cross_validation(10, data)
```

    ======================================= Test 1 =======================================
    Accuracy: 98.95104895104895%
    ======================================= Test 2 =======================================
    Accuracy: 98.6013986013986%
    ======================================= Test 3 =======================================
    Accuracy: 99.3006993006993%
    ======================================= Test 4 =======================================
    Accuracy: 98.6013986013986%
    ======================================= Test 5 =======================================
    Accuracy: 98.71794871794873%
    ======================================= Test 6 =======================================
    Accuracy: 98.83449883449883%
    ======================================= Test 7 =======================================
    Accuracy: 98.13302217036173%
    ======================================= Test 8 =======================================
    Accuracy: 98.83313885647608%
    ======================================= Test 9 =======================================
    Accuracy: 98.36639439906651%
    ======================================= Test 10 =======================================
    Accuracy: 99.53325554259042%


### Un-extended Naive Bayes 10-Fold Validation


```python
if __name__ == '__main__':
    data = get_training_data()
    # improvements commented out
    # data = over_sample(remove_stop_words_data_frame(data))
    data = data.sample(frac=1).reset_index(drop=True)

    k_fold_cross_validation(10, data)
```

    ======================================= Test 1 =======================================
    Accuracy: 93.0%
    ======================================= Test 2 =======================================
    Accuracy: 94.75%
    ======================================= Test 3 =======================================
    Accuracy: 95.75%
    ======================================= Test 4 =======================================
    Accuracy: 93.5%
    ======================================= Test 5 =======================================
    Accuracy: 92.75%
    ======================================= Test 6 =======================================
    Accuracy: 92.75%
    ======================================= Test 7 =======================================
    Accuracy: 95.5%
    ======================================= Test 8 =======================================
    Accuracy: 93.5%
    ======================================= Test 9 =======================================
    Accuracy: 92.0%
    ======================================= Test 10 =======================================
    Accuracy: 93.25%

