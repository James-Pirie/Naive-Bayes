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


def remove_stop_words(text):
    """Use nltk database to remove stop words from a row's abstract"""
    with open('data/stopwords/english', 'r') as f:
        custom_stopwords = set(f.read().splitlines())

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


def make_predictions(training_data: pd.DataFrame, testing_data: pd.DataFrame, dict):
    submission = pd.DataFrame(columns=['id', 'class'])
    new_rows = []
    classes = get_classes(training_data)
    class_probabilities = probability_per_class(classes, training_data)
    for i, row in testing_data.iterrows():
        new_row = {'id': row['id'], 'class': (predict_abstract_class(row['abstract'], dict, class_probabilities, classes))}
        new_rows.append(new_row)

    submission = pd.concat([submission, pd.DataFrame(new_rows)], ignore_index=True)
    submission.to_csv('submission.csv', index=False)
    return submission


if __name__ == '__main__':
    # get data a train a model, then use 10 fold cross validation
    data = get_training_data()
    data = remove_stop_words_data_frame(data)
    data = data.sample(frac=1).reset_index(drop=True)

    k_fold_cross_validation(10, data)







