import os
import numpy as np

from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer


class DataLoader:
    """
    Handles data loading and preprocessing for spam classification.

    Provides static methods for reading email files and converting them
    to feature vectors suitable for machine learning models.
    """

    @staticmethod
    def read_data(directory: str) -> Tuple[int, List[str]]:
        """Read email data from a directory."""
        emails = []

        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return 0, []

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        emails.append(content)
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")

        return len(emails), emails

    @staticmethod
    def load_spam_data(data_path: str = './data_train') -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess spam classification data."""
        spam_dir = os.path.join(data_path, 'original', 'spam')
        nospam_dir = os.path.join(data_path, 'original', 'nospam')
        vocab_path = os.path.join(data_path, 'original', 'vocabulary.npy')

        # Read email data
        n_spam, spam_emails = DataLoader.read_data(spam_dir)
        n_nospam, nospam_emails = DataLoader.read_data(nospam_dir)

        # Load vocabulary
        vocabulary = np.load(vocab_path, allow_pickle=True)
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        print("Vocabulary loaded")

        # Create feature matrix
        all_emails = spam_emails + nospam_emails
        X = vectorizer.fit_transform(all_emails).toarray()
        X[X > 0] = 1  # Convert to binary features

        # Create labels (1 for spam, 0 for no spam)
        y = np.concatenate([
            np.ones((n_spam, 1)),
            np.zeros((n_nospam, 1))
        ], axis=0)

        print(f"Loaded {len(all_emails)} emails ({n_spam} spam, {n_nospam} no-spam)")
        print(f"Feature matrix shape: {X.shape}")

        return X, y