import numpy as np
import string
import pandas as pd
import stanfordnlp as nlp
from datetime import datetime

DEBUG = False

__version__ = "0.1.1"

config = {
    'processors': 'tokenize,lemma',  # Comma-separated list of processors to use
    'lang': 'he',  # Language code for the language to build the Pipeline in
    'tokenize_model_path': './he_htb_models/he_htb_tokenizer.pt',
    # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
    # 'mwt_model_path': './he_htb_models/he_htb_mwt_expander.pt',
    # 'pos_model_path': './he_htb_models/he_htb_tagger.pt',
    # 'pos_pretrain_path': './he_htb_models/he_htb.pretrain.pt',
    'lemma_model_path': './he_htb_models/he_htb_lemmatizer.pt'
    # 'depparse_model_path': './he_htb_models/he_htb_parser.pt',
    # 'depparse_pretrain_path': './he_htb_models/he_htb.pretrain.pt'
}


class Classification:

    def __init__(self, training_process_only=False, using_stanford_nlp=False):
        """
        Initialize first parameters.
        :param training_process_only:
        :param using_stanford_nlp:
        """

        self.using_stanford_nlp = using_stanford_nlp

        self.hebrew_encoding = "utf8"
        self.hebrew_encode_iso = "ISO-8859-8"
        self.data_frame = None
        self.id_to_category = None
        self.category_id_df = None
        self.category_to_id = None
        self.labels = None
        self.data_for_training = None
        self.data_for_testing = None
        self.data_for_validation = None
        self.total_categories = 0

        heb_stop_words_file = open('static/classification/heb_stop_words.txt', 'r', encoding=self.hebrew_encoding)
        self.hebrew_stop_words = self.remove_punctuation(heb_stop_words_file.readlines(), return_split_document=True)

        if self.using_stanford_nlp:
            self.nlp_pipe = nlp.Pipeline(**config)
        else:
            self.nlp_pipe = None
        pass

    def classification_process(self,
                               split_database=False,
                               save_pre_processed_data=False,
                               using_feature_selection=False):
        """

        :param split_database:
        :param save_pre_processed_data:
        :param using_feature_selection:
        :return:
        """
        try:
            # Reading the data into DataFrame.
            self.read_database()

            # Splitting the database into three parts.
            if split_database:
                # Initial Process - Splitting the data into three parts:
                # 1) Training
                # 2) Validation
                # 3) Testing
                self.split_data(self.data_frame)

            # First Phase - pre processing the data, removing stop words, tokenization, lamentation and more.
            pre_processed_data = self.pre_processing()

            if save_pre_processed_data:
                self.save_data(pre_processed_data)

            # Second Phase -- Feature Extraction / Feature Selection
            tf_idf_features, tfidf, bow_features, count_vectorized = self.feature_extraction(pre_processed_data,
                                                                                             tf_idf=True,
                                                                                             bow=True)
            if using_feature_selection:
                self.feature_selection(pre_processed_data, tf_idf_features, tfidf, bow_features, count_vectorized)

            # Third Phase -- Training the Model
            cv_df, models, best_model = self.training_model(split_database,
                                                            pre_processed_data,
                                                            tf_idf_features,
                                                            tfidf,
                                                            bow_features,
                                                            count_vectorized,
                                                            show_plt=True)
            for model in models:
                model_name = datetime.now().strftime('%H_%M_%S')

                # Forth and Fifth Phase -- Validation the model from phase third and Creating Final Model
                final_model = self.model_evaluation(model, tf_idf_features)

                # Final Phase -- Saving the final Model
                self.save_final_model(final_model=final_model, model_name=model_name)

        except Exception as e:
            print(e)
        pass

    def read_database(self, file_name=None, sheet_name=None):
        """

        :param file_name:
        :param sheet_name:
        :return:
        """
        try:
            if file_name is None and sheet_name is None:
                _temp_data = pd.read_excel('complaints.xlsx', sheet_name='complaints')
            else:
                _temp_data = pd.read_excel(file_name, sheet_name=sheet_name)

            # Entering the data from the excel file into pandas DataFrame.
            self.data_frame = pd.DataFrame(_temp_data, columns=['Department', 'Details'])
            print(str.format("Data size: {0}", self.data_frame.shape))
            # Removing any row that contains null/empty value in the columns Details and Department
            self.data_frame = self.data_frame[pd.notnull(self.data_frame['Details'])]
            print(str.format("Data size after remove null: {0}", self.data_frame.shape))
            self.data_frame = self.data_frame[pd.notnull(self.data_frame['Department'])]
            print(str.format("Data size after remove null: {0}", self.data_frame.shape))

            # Adding a new column of category id - numeric.
            self.data_frame['category_id'] = self.data_frame['Department'].factorize()[0]
            # Sorting the data by category id.
            self.data_frame = self.data_frame.sort_values('category_id')

            # Temporary variables
            self.category_id_df = self.data_frame[['Department', 'category_id']].sort_values('category_id')

            self.category_to_id = dict(self.category_id_df.values)
            self.id_to_category = dict(self.category_id_df[['category_id', 'Department']].values)

            self.total_categories = len(self.category_to_id)

            self.labels = self.data_frame.category_id

        except Exception as e:
            print(e)

        pass

    def split_data(self, data: pd.DataFrame, training=0.6, validation=0.3):
        """

        :param data:
        :param training:
        :param validation:
        :return:
        """
        try:
            print(str.format("Splitting the data to three parts: Training={0}% \t Validation={1}% \t Testing={2}%",
                             training * 100, validation * 100, 10))

            end_training_index = round(len(data) * training)
            end_validation_index = round(len(data) * validation)

            self.data_for_training = data[:end_training_index]
            self.data_for_validation = data[end_training_index:end_training_index + end_validation_index]
            self.data_for_testing = data[end_validation_index + end_training_index:]

            print(str.format("Total rows: Training={0} \t Validation={1} \t Testing={2}",
                             len(self.data_for_training),
                             len(self.data_for_validation),
                             len(self.data_for_testing)))

        except Exception as e:
            print(e)
        pass

    def pre_processing(self, data=None):
        """

        :param data:
        :return:
        """

        print(str.format("Time start pre processing: {0}", datetime.now().strftime('%H:%M:%S')))

        if self.using_stanford_nlp:

            # First Phase -- Pre Processing Phase

            # ['category_id', 'Details']
            data = pd.DataFrame(self.pre_processing(self.data_for_training),
                                columns=['category_id', 'Details'])

            print(str.format("Total sentences converted:{0}", len(data)))

            words_to_sentence = self.pre_processing_standford(data)
            print(str.format("Time end pre processing: {0}", datetime.now().strftime('%H:%M:%S')))
            return words_to_sentence
        else:
            data_raw = self.pre_processing_normal()
            print(str.format("Time end pre processing: {0}", datetime.now().strftime('%H:%M:%S')))
            return data_raw

    def pre_processing_standford(self, raw_data: pd.DataFrame, nlp_before_stop_word=False, lamentation_before_stop=True):
        """
        raw_data = Pandas DataFrame.

        The first phase is pre processing the data.

        Flow
        ----------
        1) Removing hebrew stop words.
        2) Lamentation each word in the data by using stanford nlp.
        3)

        :return: DataFrame of all the sentences and their labels in the database with only the lamentation of the word.
        """
        # First phase is tokenize and remove stop word and lamentation
        sentence_to_words = []
        doc = None
        try:
            for index, complaint in raw_data.iterrows():
                # pipe each sentence (complaint)
                # compliant contains:
                # -------------------
                # *Department
                # *Details
                # *Department Id
                # TODO: Removing any signs from the document before processing it in the pipe.
                # TODO: Add the option to remove hebrew stop words before this process.

                if nlp_before_stop_word:
                    doc = self.nlp_pipe(self.remove_punctuation(complaint.Details))
                else:
                    doc = self.nlp_pipe(self.remove_stop_words(complaint.Details, convert_to_string_category=False))
                # take the words (each words is a class of Word)
                i = 0
                if len(doc.sentences) == 0:
                    continue
                if len(doc.sentences) > 1:
                    max_len = len(doc.sentences[0].words)
                    for sentence in doc.sentences:
                        if max_len < len(sentence.words):
                            max_len = len(sentence.words)
                            i = doc.sentences.index(sentence)

                words = doc.sentences[i].words
                words_to_lemma = self._convert_words_to_lemma(words)

                if lamentation_before_stop:
                    # Removing stop words after each word converted into it bases - lamentation
                    sentence_to_words.append(self.remove_stop_words(words_to_lemma,
                                                                    index,
                                                                    complaint.Department, complaint.category_id))
                else:
                    # TODO: Adding the option to remove stop words before lamentation.
                    # sentence_to_words.append(self.remove_stop_words(words_to_lemma, len(sentence_to_words)))
                    pass

        except Exception as e:
            print(e)

        return pd.DataFrame(sentence_to_words)

    def pre_processing_normal(self, data_raw=None):
        """

        :return:
        """
        try:
            if data_raw is None:

                data_raw_details = [[complaint.Department, self.remove_stop_words(complaint.Details, document_number=index, department=complaint.Department)] for index, complaint in self.data_frame.iterrows()]
                data_raw = pd.DataFrame(data_raw_details, columns=['Department', 'Details'])

                if DEBUG:
                    print(data_raw.head())

        except Exception as e:
            print(e)
        return data_raw

    def save_data(self, data, encoding="utf8"):
        """

        :param data:
        :param encoding:
        :return:
        """

        print(isinstance(data, pd.DataFrame))
        if isinstance(data, pd.DataFrame) is False:
            df = pd.DataFrame({'Details': data})
        else:
            df = data
        df.to_csv("complaint_pre_processing.csv")
        pass

    def feature_extraction(self, sentence_to_words: pd.DataFrame, tf_idf=False, bow=False, max_features=10000):
        """

        :param sentence_to_words:
        :param tf_idf:
        :param bow:
        :param max_features:
        :return:
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        tf_idf_features = None
        tfidf = None
        bow_features = None
        count_vectorized = None

        try:
            if tf_idf:
                tfidf = TfidfVectorizer(sublinear_tf=True,
                                        min_df=5,
                                        # norm=12
                                        encoding=self.hebrew_encoding,
                                        ngram_range=(1, 2),
                                        stop_words=self.hebrew_stop_words,
                                        lowercase=False)
                tf_idf_features = tfidf.fit_transform(sentence_to_words.Details).toarray()
                print(str.format("tf_idf_features: {0} ", tf_idf_features.shape))
        except Exception as e:
            print(e)

        try:
            if bow:
                count_vectorized = CountVectorizer(max_features=max_features,
                                                   lowercase=False,
                                                   stop_words=self.hebrew_stop_words,
                                                   analyzer="char",
                                                   tokenizer=None)
                bow_features = count_vectorized.fit_transform(sentence_to_words.Details).toarray()
                print(str.format("bow_features: {0}", bow_features.shape))
        except Exception as e:
            print(e)
        return tf_idf_features, tfidf, bow_features, count_vectorized

    def feature_selection(self, sentence_to_words: pd.DataFrame, tf_idf_features, tfidf, bow_features,
                          count_vectorized):

        """

        :param sentence_to_words:
        :param tf_idf_features:
        :param tfidf:
        :param bow_features:
        :param count_vectorized:
        :return:
        """

        from sklearn.feature_selection import chi2

        try:
            N = 2
            for Product, category_id in sorted(self.category_to_id.items()):
                # category_id = Product.category_id
                # details = Product.Details
                features_chi2 = chi2(tf_idf_features, self.labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                print("# '{}':".format(Product))
                print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
                print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

        except Exception as e:
            print(e)
        pass

    def training_model(self, split_database: bool,
                       pre_processed_data: pd.DataFrame,
                       tf_idf_features, tfidf,
                       bow_features, count_vectorized,
                       show_plt=False):
        """
        Initial first parameters for the training models.
        Training Models:
        1) RandomForestClassifier
        2) LinearSVC
        3) MultinomialNB
        4) LogisticRegression
        5)



        :param pre_processed_data:
        :param split_database:
        :param tf_idf_features:
        :param tfidf:
        :param bow_features:
        :param count_vectorized:
        :param show_plt:
        :return:
        """

        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC, SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier

        if split_database is False:
            x_train, x_test, y_train, y_test = train_test_split(pre_processed_data['Details'],
                                                                pre_processed_data['Department'],
                                                                random_state=0)

        # All the algorithms that we want to use for training.
        models = [
            # -----------------------------
            # Other NLP Algorithms
            # -----------------------------
            # LogisticRegression(multi_class="multinomial"),
            # LinearSVC(multi_class="crammer_singer"),
            # # LogisticRegressionCV(multi_class="multinomial"),
            # KNeighborsClassifier(n_neighbors=5),
            BernoulliNB(),
            GaussianNB(),
            # -----------------------------
            # Default NLP Algorithms
            # -----------------------------
            MultinomialNB(),
            LinearSVC(),
            LogisticRegression(random_state=0),
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        ]

        # CV = 5
        # cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            print(str.format("Time Start {0}\n Model: {1}", datetime.now().strftime('%H:%M:%S'), model))
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, tf_idf_features, self.labels, scoring='accuracy', cv=5)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))

            print(str.format("Time Ended {0}", datetime.now().strftime('%H:%M:%S')))

        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        best_model = 1
        print(cv_df.groupby('model_name').accuracy.mean())

        if show_plt:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.boxplot(x='model_name', y='accuracy', data=cv_df)
            sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                          size=8, jitter=True, edgecolor="gray", linewidth=2)
            plt.show()

        return cv_df, models, best_model

    def model_evaluation(self, model,  tf_idf_features):

        """

        :param model:
        :param tf_idf_features:
        :return:
        """

        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC, SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split

        x_train, _, y_train, _, _, _ = train_test_split(tf_idf_features,
                                                        self.labels,
                                                        self.data_frame.index,
                                                        test_size=0.33,
                                                        random_state=0)
        model.fit(x_train, y_train)

        return model

    def confusion_matrix(self, final_model, tf_idf_features, show_plot=False, calculate_matrices=False):

        """

        :param final_model:
        :param tf_idf_features:
        :param show_plot:
        :param calculate_matrices:
        :return:
        """

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(tf_idf_features,
                                                                                         self.labels,
                                                                                         self.data_frame.index,
                                                                                         test_size=0.33,
                                                                                         random_state=0)

        y_pred = final_model.predict(x_test)
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(y_test, y_pred)

        if show_plot:
            import seaborn as sns
            import matplotlib as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(conf_mat,
                        annot=True,
                        fmt='d',
                        xticklabels=self.category_id_df.Department.values,
                        yticklabels=self.category_id_df.Department.values)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

        if calculate_matrices:
            from sklearn import metrics
            print(metrics.classification_report(y_test,
                                                y_pred,
                                                target_names=self.data_frame['Department'].unique()))

    def model_validation_test(self):
        pass

    @staticmethod
    def remove_punctuation(document, return_split_document=False):
        """
        Method that removes any punctuation, digits and None value.
        :param document: The document that will processed.
        :param return_split_document: By default it is False.
        If it true the method will return the document as a split list.
        :return: By default return a string of words after the process. Otherwise returns as split list.
        """
        if isinstance(document, list):
            document = str(document)

        document = document.translate(str.maketrans('', '', string.punctuation))
        # document = document.translate(str.maketrans('', '', string.digits))
        document = document.translate(str.maketrans('', '', "None"))
        document = document.replace("'", "").replace('"', '')

        if return_split_document:
            return document.split()
        else:
            return document

    def remove_stop_words(self, document,
                          return_string=True,
                          convert_to_string_category=False,
                          document_number=None,
                          department=None,
                          category_id=None):
        """
        Removing stop words from a given document.
        :param return_string:
        :param category_id: The number id of the current document - The label id.
        :param department: The name {string} of the current document - The label name.
        :param document: The document that will process.
        :param document_number: Number of the document to understand where there is a problem.
        :param convert_to_string_category: True to convert the document back to a sentence. Default is set to True.
        :return: Depends on the convert_to_string parm. By default it will returns a sentence with white spaces. When
        the param will set to False the method will return a list of words and not a full sentence.
        """

        import re
        filtered_words = None
        try:
            if type(document) == str:
                document = document.split()

            document = self.remove_punctuation(document, return_split_document=True)

            # Remove stop words.
            filtered_words = [word for word in document if word not in self.hebrew_stop_words]

            filtered_words = self.remove_punctuation(filtered_words, return_split_document=True)

            # Remove words if they are None value.
            filtered_words = [word for word in filtered_words if word is not None]

            # Remove words if they are digits or contains digits.
            filtered_words = [word for word in filtered_words if word is not word.isdigit()]
            filtered_words = [word for word in filtered_words if bool(re.search(r'\d', word)) is False]

            # Remove a word if the length is less than X.
            filtered_words = [word for word in filtered_words if len(word) > 1]

            if DEBUG:
                print(str.format("Row index {0} - Department {1} - Details {2}",
                                 document_number, department, filtered_words))

            if convert_to_string_category:
                return [" ".join(filtered_words), category_id]
            elif return_string:
                return " ".join(filtered_words)
            else:
                return filtered_words

        except Exception as e:
            print(e)
            return None

    @staticmethod
    def _convert_words_to_lemma(words):
        """
        The function takes a list of word's objects and returns a list of lemma.
        Each lemma type of string.
        :param words: A list of Word's Objects.
        :return: A list of lemma
        """
        return [word.lemma for word in words]

    def model_test_test(self, tf_idf_features):
        """

        :param tf_idf_features:
        :return:
        """

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(tf_idf_features,
                                                                                         self.labels,
                                                                                         self.data_frame.index,
                                                                                         test_size=0.33,
                                                                                         random_state=0)
        pass

    @staticmethod
    def save_final_model(final_model, model_name):
        """

        :param final_model:
        :param model_name:
        :return:
        """
        # Saving the model.
        from sklearn.externals import joblib
        joblib.dump(final_model, str.format("{0}_{1}.pkl", model_name, datetime.now().strftime('%dd_%mm_%yy')))
        pass

    def show_data_count(self):
        """

        :return:
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        self.data_frame.groupby('Department').Details.count().plot.bar(ylim=0)
        plt.show()

    @staticmethod
    def load_final_model(path):
        """

        :param path:
        :return:
        """
        from sklearn.externals import joblib
        final_model = open(path, 'rb')
        clf = joblib.load(final_model)
        return clf


if __name__ == '__main__':
    classification = Classification()
    classification.classification_process(split_database=False)
