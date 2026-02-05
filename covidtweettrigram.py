
import numpy as np
import nltk

np.random.seed(0)
nltk.download('word2vec_sample')

class NgramLM:
    
    def __init__(self):

        # Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram = {}
        
        # Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
        self.bigram_prefix_to_trigram_weights = {}

    def load_trigrams(self):
        """
        Loads the trigrams from the data file and fills the dictionaries defined above.

        """
        with open("data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
            lines = f.readlines()
            for line in lines:
                word1, word2, word3, count = line.strip().split()
                if (word1, word2) not in self.bigram_prefix_to_trigram:
                    self.bigram_prefix_to_trigram[(word1, word2)] = []
                    self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
                self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
                self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

    def top_next_word(self, word1, word2, n=10):
        
        next_words = []
        probs = []
        bigram = (word1, word2)
        
        if bigram not in self.bigram_prefix_to_trigram:
            return [], []

        words = self.bigram_prefix_to_trigram[bigram]
        counts = self.bigram_prefix_to_trigram_weights[bigram]

        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        top_indices = sorted_indices[:n]
        
        next_words = [words[i] for i in top_indices]
        total_count = sum(counts)
        probs = [counts[i] / total_count for i in top_indices]

        return next_words, probs

    def sample_next_word(self, word1, word2, n=10):
        """
        Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

        """
        next_words = []
        probs = []
        bigram = (word1, word2)
        
        if bigram not in self.bigram_prefix_to_trigram:
            return [], [] 
        
        words = self.bigram_prefix_to_trigram[bigram]
        counts = self.bigram_prefix_to_trigram_weights[bigram]
        
        total_count = sum(counts)
        probabilities = [count / total_count for count in counts]
        indices = np.random.choice(len(words), size=min(n, len(words)), replace=False, p=probabilities)
        next_words = [words[i] for i in indices]
        
        probs = [probabilities[i] for i in indices]
        
        return next_words, probs


    def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
        """
        Generate sentences using beam search.

        """
        sentences = []
        probs = []
        beam_queue = [(prefix.split(), 1.0)]
        
        for _ in range(max_len - len(prefix.split())):
            new_beam_queue = []
            
            for sentence, prob in beam_queue:
                if sentence[-1] == "<EOS>":  
                    new_beam_queue.append((sentence, prob))
                    continue  
                
                if len(sentence) < 2:
                    continue  
                word1, word2 = sentence[-2], sentence[-1]
                next_words, next_probs = sampler(word1, word2, n=beam)  
                
                for next_word, next_prob in zip(next_words, next_probs):
                    new_sentence = sentence + [next_word]
                    new_prob = prob * next_prob  
                    new_beam_queue.append((new_sentence, new_prob))
            
            if not new_beam_queue:
                break  
            
            beam_queue = sorted(new_beam_queue, key=lambda x: x[1], reverse=True)[:beam]
        
        for sentence, prob in beam_queue:
            if sentence[-1] != "<EOS>":
                sentence.append("<EOS>")  
            sentences.append(" ".join(sentence))
            probs.append(prob)
        
        return sentences, probs


print("======================================================================")
print("Checking Language Model")
print("======================================================================")

# Defines language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()

print("------------- Evaluating top next word prediction -------------")
next_words, probs = language_model.top_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)

print("------------- Evaluating sample next word prediction -------------")
next_words, probs = language_model.sample_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)

print("------------- Evaluating beam search -------------")
sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")


sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
    print(sent, prob)
