import numpy as np
import math
import string


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
#        pass    # REMOVE THIS
        documentNum = 0
        doc = self.documents
        with open(self.documents_path) as file:
            for lines in file.readlines():
                curr = ''.join([x for x in lines if x in string.ascii_letters + '\'- '])
                curr = curr.rstrip('\n ')
                doc = curr.split(' ')
                documentNum += 1
                self.doc.append(doc)
                    
            self.number_of_documents = documentNum
            
        
        

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        #        pass    # REMOVE THIS

        for i in range(len(self.documents)):
            for j in range(i, len(self.documents[i])):
                self.vocabulary.append(self.documents[i][j])

        self.vocabulary = set(self.vocabulary)
        self.vocabulary = sorted(self.vocabulary)
        self.vocabulary_size = len(self.vocabulary)
        #print(self.vocabulary)
        

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################

        self.term_doc_matrix = np.zeros((self.number_of_documents,self.vocabulary_size))

        for i in range(self.number_of_documents):
            for n, item in enumerate(self.vocabulary):
                counter = 0
                for j in range(0, len(self.documents[i])):
                     if item == self.documents[i][j]:
                            counter += 1
                self.term_doc_matrix[i][n] = counter

        


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        self.document_topic_prob = np.random.rand(self.number_of_documents, number_of_topics)
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.rand(number_of_topics, len(self.vocabulary))
        self.topic_word_prob = normalize(self.topic_word_prob)

        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        #print("E step:")
        
        # ############################
        # your code here
        # ############################

        number_of_topics = len(self.document_topic_prob[0])
        print("number_of_topics: " + str(number_of_topics))
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                probabilityTotal = 0
                for k in range(number_of_topics):
                    self.topic_prob[i][k][j] = self.document_topic_prob[i][k] * self.topic_word_prob[k][j]
                    probabilityTotal += self.topic_prob[i][k][j]

                for k in range(number_of_topics):
                    self.topic_prob[i][k][j] /= probabilityTotal


            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        #print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################

        for i in range(number_of_topics):
            for j in range(self.vocabulary_size):
                curr = 0
                for k in range(self.number_of_documents):
                    curr += self.term_doc_matrix[k][j] * self.topic_prob[k][i][j]

                self.topic_word_prob[i][j] = curr

        self.topic_word_prob = normalize(self.topic_word_prob)
        
        # update P(z | d)

        # ############################
        # your code here
        # ############################

        for i in range(self.number_of_documents):
            for j in range(number_of_topics):
                self.document_topic_prob[i][j] = 0
                for k in range(self.vocabulary_size):
                    self.document_topic_prob[i][j] += self.term_doc_matrix[i][k] * self.topic_prob[i][j][k]

        self.document_topic_prob = normalize(self.document_topic_prob)
        
        


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################

        total = 0
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                curr = 0
                for k in range(number_of_topics):
                    curr += self.document_topic_prob[i][k] * self.topic_word_prob[k][j]
                if curr > 0:
                    total += np.log(curr) * self.term_doc_matrix[i][j]
        self.likelihoods.append(total)
        
        return

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################

            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            if (self.likelihoods[iteration] - self.likelihoods[iteration-1] < epsilon ):
                if(iteration >= 1):
                    break




def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 200
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
