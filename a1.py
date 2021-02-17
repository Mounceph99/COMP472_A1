import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# ML METRICS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def read_documents(file_name):
    """Read data set file and return documents with target labels."""

    docs = []
    labels = []

    with open(file_name, encoding='utf8') as f:
        for line in f:
            # Split line to pick only the label and the document
            splitted = line.strip().split(' ')

            # Append document text (as a string) to `docs`
            docs.append(' '.join(splitted[3:]))

            # Append label to `labels`
            labels.append(splitted[1])
    
    return docs, labels

def map_labels_to_indices(labels):
    """Map every label to an index"""

    # Get all unique labels in passed labels
    unique_labels = sorted(list(set(labels)))

    # Create a dictionary mapping a label to its index
    labelToIndexDict = dict()
    for i in range(len(unique_labels)):
        labelToIndexDict[unique_labels[i]] = i
    
    return [labelToIndexDict[label] for label in labels], unique_labels

def split_docs(docs, labels, training_proportion):
    """Split the data set according to training_proportion
    to create a training set and evaluation set"""

    # get index of where to separate training docs and eval docs
    split_point = int(training_proportion*len(docs))

    # split the docs
    train_docs = docs[:split_point]
    train_labels = labels[:split_point]
    eval_docs = docs[split_point:]
    eval_labels = labels[split_point:]

    return train_docs, train_labels, eval_docs, eval_labels

def plot_classes(labels):
    """Plot the number of instances in each class of labels"""

    # Get unique labels as a list
    unique_labels = sorted(list(set(labels)))

    # Get count of each label
    counts = [labels.count(label) for label in unique_labels]

    # Graph count of each label
    barGraph = plt.bar(unique_labels, counts)
    barGraph[0].set_color('r')
    plt.title('Number of instances in each class')
    
    # Add count for each class to the graph
    for i, v in enumerate(counts):
        plt.text(plt.xticks()[0][i] - 0.10, v + 50, str(v))
    
    print('Displaying bar graph of classes. Close graph to continue.')
    plt.show()

def docs_to_tfidf(docs):
    """Transforms the docs into vectors of tf-idf features.
    
    Returns the transformed docs set and the CountVectorizer and TfidfTransformer used."""

    count_vect = CountVectorizer()
    # Transform all documents into feature vectors
    # All words in the documents will be mapped to a unique index
    # If there are `m` documents and the vocabulary set V has length |V| = n,
    # Then the result will be a m x n matrix M
    # Where M[i, j] is the count of the word with index `j` in document `i`
    # It is possible to map a word to its index by doing `docs_word_counts.vocabulary_[u'word']`
    docs_word_counts = count_vect.fit_transform(docs)

    # Transform counts to frequencies (`count/total words in specificdocument`)
    # This is called tf: term frequency
    # Also downscale words that occur a lot in the corpus (i.e. dataset)
    # This is called tf-idf: Term Frequency times Inverse Document Frequency
    tfidf_transformer = TfidfTransformer()
    docs_word_tfidf = tfidf_transformer.fit_transform(docs_word_counts)

    return docs_word_tfidf, count_vect, tfidf_transformer



def train_nb(docs, labels):
    """Returns a Naive Bayes Model fitted to the passed docs and labels."""

    print('Training Naive Bayes Model... ', end='', flush=True)
    model = MultinomialNB().fit(docs, labels)
    print('Done')
    return model

def train_base_DT(docs, labels):
    """Returns a Decision Tree (criterion='entropy') fitted to the passed docs and labels."""

    print('Training Base Decision Tree Model... ', end='', flush=True)
    model = DecisionTreeClassifier(criterion='entropy').fit(docs, labels)
    print('Done')
    return model

def train_best_DT(docs, labels):
    """Returns a Decision Tree fitted to the passed docs and labels."""

    print('Training Best Decision Tree Model... ', end='', flush=True)
    model = DecisionTreeClassifier().fit(docs, labels)
    print('Done')
    return model

def evaluate_and_save(model_name, model, eval_docs, eval_labels, classes, index_offset, dataset_name):
    """Uses input model to evaluate eval_docs and saves results into a file `[model name]-[dataset]`."""

    print('Evaluating {}... '.format(model_name), end='', flush=True)

    prediction = model.predict(eval_docs)
    saved_file_name = '{}-{}'.format(model_name, dataset_name)

    with open(saved_file_name, 'w') as f:
        # Write legend (index and corresponding class)
        f.write('========================\n')
        f.write('Legend:\n')
        for i in range(len(classes)):
            f.write('{}: {}\n'.format(i, classes[i]))
        f.write('\n')

        # Write predictions by model
        f.write('========================\n')
        f.write('Predictions ("{document_row_number}, {prediction}", where "document_row_number" is 0-indexed):\n')
        for i in range(prediction.shape[0]):
            f.write('{}, {}{}\n'.format(i+index_offset, prediction[i], ' [Misclassified]' if prediction[i] != eval_labels[i] else ''))
        f.write('\n')

        # Write confusion matrix
        f.write('========================\n')
        f.write('Confusion matrix (row is expected, column is predicted):\n')
        np.set_printoptions(threshold=np.inf) # Prevent numpy from truncating if numpy matrix is too long
        f.write(str(metrics.confusion_matrix(eval_labels, prediction)))
        f.write('\n\n')

        # Write metrics
        precision, recall, f1_measure, support = precision_recall_fscore_support(eval_labels, prediction)
        f.write('========================\n')
        f.write('Metrics:\n\n')
        
        f.write('Precision\n')
        f.write('----------\n')
        for i in range(len(precision)):
            f.write('{}: {}\n'.format(classes[i], precision[i]))
        f.write('\n')

        f.write('Recall\n')
        f.write('----------\n')
        for i in range(len(recall)):
            f.write('{}: {}\n'.format(classes[i], recall[i]))
        f.write('\n')

        f.write('F1-Measure\n')
        f.write('----------\n')
        for i in range(len(f1_measure)):
            f.write('{}: {}\n'.format(classes[i], f1_measure[i]))
        f.write('\n')

        f.write('Accuracy: {}'.format( accuracy_score(eval_labels, prediction) ))
    
    print('Done [Results saved in {}]'.format(saved_file_name), flush=True)

def main(file_name):
    print()

    # Get all documents and their labels
    all_docs, all_labels = read_documents(file_name)

    # Change labels from names to indices
    all_labels_indexed, unique_labels = map_labels_to_indices(all_labels)

    # Split docs and labels into training and evaluation sets (80% training, 20% evaluation)
    train_docs, train_labels, eval_docs, eval_labels = split_docs(all_docs, all_labels_indexed, 0.80)

    # Plot the count of each classes (pos and neg) in the training set
    plot_classes(all_labels)

    print()

    # Transform training docs into tf-idf docs
    # And save CountVector and TfidTransformer used to transform docs
    train_docs_tfidf, count_vect, tfidf_transformer = docs_to_tfidf(train_docs)

    # Create a models dictionary containing all models (Naive Bayes, Base DT, Best DT)
    models = {
        'NaiveBayes': train_nb(train_docs_tfidf, train_labels),
        'BaseDT': train_base_DT(train_docs_tfidf, train_labels),
        'BestDT': train_best_DT(train_docs_tfidf, train_labels)
    }

    print()

    # Prepare evaluation set to be used in models
    # Transform evaluation set to tfidf feature vector using the CountVector and TfidfTransformer from training set
    eval_counts = count_vect.transform(eval_docs)
    eval_tfidf = tfidf_transformer.transform(eval_counts)

    # Evaluate (and save results) of each model on evaluation set
    for model_name, model in models.items():
        evaluate_and_save(
            model_name=model_name,
            model=model,
            eval_docs=eval_tfidf,
            eval_labels=eval_labels,
            classes=unique_labels,
            index_offset=len(train_docs),
            dataset_name=os.path.splitext(os.path.basename(file_name))[0]
        )
    
    print()
    print('Finished')

if __name__ == '__main__':
    # Make sure only one input file is passed
    if len(sys.argv) < 2:
        print('File expected')
    elif len(sys.argv) > 2:
        print('Too many files passed')
    else:
        main(sys.argv[1])