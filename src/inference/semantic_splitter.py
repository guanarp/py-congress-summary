from typing import List
import spacy

class SemanticSplitter:
    def __init__(self, model_name: str = "es_core_news_sm", max_words: int = 4096):
        """
        Parameters
        ----------
        model_name : str
            spaCy model name for Spanish (default: 'es_core_news_sm').
        max_words : int
            Maximum number of tokens per chunk (estimated by words).
        """
        self.nlp = spacy.load(model_name)
        self.max_words = max_words

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks based on sentences.

        Parameters
        ----------
        text : str
            The input long text to split.

        Returns
        -------
        List[str]
            List of text chunks.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        chunks = []
        current_chunk = ""
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if current_word_count + sentence_word_count > self.max_words:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_word_count
            else:
                current_chunk += " " + sentence
                current_word_count += sentence_word_count

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
