from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
import argparse
from time import gmtime, strftime

import operator
import os
import re

app = Flask(__name__)
api = Api(app)


@app.route('/')
def welcome_message():
    return 'Welcome to the keyphrase extraction backend providing RAKE.'


# http://localhost:4124/keyphrase/ping
# http://localhost:4124/keyphrase/extract?plaintext=...
class Ping(Resource):
    def get(self):
        return strftime("%Y-%m-%d %H:%M:%S", gmtime())


# cmp. https://github.com/aneesha/RAKE - put in class and ported to Python 3
class RAKEPy3(Resource):
    debug = False
    test = True

    def is_number(self,s):
        try:
            float(s) if '.' in s else int(s)
            return True
        except ValueError:
            return False

    def load_stop_words(self,stop_word_file):
        """
        Utility function to load stop words from a file and return as a list of words
        @param stop_word_file Path and file name of a file containing stop words.
        @return list A list of stop words.
        """
        stop_words = []
        for line in open(stop_word_file):
            if line.strip()[0:1] != "#":
                for word in line.split():  # in case more than one per line
                    stop_words.append(word)
        return stop_words

    def separate_words(self,text, min_word_return_size):
        """
        Utility function to return a list of all words that are have a length greater than a specified number of characters.
        @param text The text that must be split in to words.
        @param min_word_return_size The minimum no of characters a word must have to be included.
        """
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for single_word in splitter.split(text):
            current_word = single_word.strip().lower()
            #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(current_word) > min_word_return_size and current_word != '' and not self.is_number(current_word):
                words.append(current_word)
        return words

    def split_sentences(self,text):
        """
        Utility function to return a list of sentences.
        @param text The text that must be split in to sentences.
        """
        sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
        sentences = sentence_delimiters.split(text)
        return sentences

    def build_stop_word_regex(self,stop_word_file_path):
        stop_word_list = self.load_stop_words(stop_word_file_path)
        stop_word_regex_list = []
        for word in stop_word_list:
            word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
            stop_word_regex_list.append(word_regex)
        stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
        return stop_word_pattern

    def generate_candidate_keywords(self,sentence_list, stopword_pattern):
        phrase_list = []
        for s in sentence_list:
            tmp = re.sub(stopword_pattern, '|', s.strip())
            phrases = tmp.split("|")
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != "":
                    phrase_list.append(phrase)
        return phrase_list

    def calculate_word_scores(self,phraseList):
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = self.separate_words(phrase, 0)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            #if word_list_degree > 3: word_list_degree = 3 #exp.
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree  #orig.
                #word_degree[word] += 1/(word_list_length*1.0) #exp.
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
        #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return word_score

    def generate_candidate_keyword_scores(self,phrase_list, word_score):
        keyword_candidates = {}
        for phrase in phrase_list:
            keyword_candidates.setdefault(phrase, 0)
            word_list = self.separate_words(phrase, 0)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def __init__(self):
        self.stop_words_path = 'SmartStoplist.txt' # stop_words_path
        self.__stop_words_pattern = self.build_stop_word_regex(self.stop_words_path)

    def run(self, text):
        sentence_list = self.split_sentences(text)

        phrase_list = self.generate_candidate_keywords(sentence_list, self.__stop_words_pattern)

        word_scores = self.calculate_word_scores(phrase_list)

        keyword_candidates = self.generate_candidate_keyword_scores(phrase_list, word_scores)

        #sorted_keywords = sorted(keyword_candidates.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_keywords = sorted(iter(keyword_candidates.items()), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('plaintext', type=str, required=True, help="Text for extraction", action='append')

        text = re.sub('\s+', ' ', parser.parse_args()['plaintext'][0].replace('\n', '').strip())
        sentence_list = self.split_sentences(text)
        # stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1
        stop_path = 'SmartStoplist.txt'  # SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
        stop_word_pattern = self.build_stop_word_regex(stop_path)

        # Generate candidate keywords
        phrase_list = self.generate_candidate_keywords(sentence_list, stop_word_pattern)

        # Calculate individual word scores
        word_scores = self.calculate_word_scores(phrase_list)

        # Generate candidate keyword scores
        keyword_candidates = self.generate_candidate_keyword_scores(phrase_list, word_scores)
        if self.debug: print(keyword_candidates)

        sorted_keywords = sorted(iter(keyword_candidates.items()), key=operator.itemgetter(1), reverse=True)
        if self.debug: print(sorted_keywords)

        total_keywords = len(sorted_keywords)
        # Only take top 1/3.5 of list via range
        filtered_keywords = sorted_keywords[0:((int)(total_keywords / 3.5))]
        # if debug: print totalKeywords
        # print (sorted_keywords[0:((int)(total_keywords / 4))])

        # Remove keyphrases longer than 4 words (usually meaningless set of words)
        result_list = [k for k in filtered_keywords if len(k[0].split(' ')) <= 4]  # -> List comprehension (look it up)
        json_answer = '['
        for keyphrase in result_list:
            # Split keyphrase into words and only merge those back together that contain no (special characters or 0-9)
            temp = ' '.join([k for k in keyphrase[0].split(' ') if len(re.findall("[^A-Za-z-]+", k)) == 0])
            if temp != '':  # In case we removed every word, don't write cleaned keyphrase to file, otherwise do
                # keyphrase = tuple([temp,keyphrase[1]])
                json_answer = json_answer + '"' + str(keyphrase[0]) + '":' + str(keyphrase[1]) + ','
        if json_answer == '[':
            return '[]'
        else:
            return re.sub(',$', ']', json_answer)

# if __name__ == '__main__':
#     print('Detecting key phrases...')
#
#     rake = RAKEPy3('SmartStoplist.txt')
#
#     # Traverse corpus directors
#     for root, dirs, files in os.walk("./txten"):
#         for file in files:
#             #Open all keyfiles
#             if file.endswith('.txt'):
#                 print(' Analyzing ' + os.path.join(root, file))
#
#                 # Read in contents of text file into one line
#                 with open(os.path.join(root, file), encoding='utf-8') as a_file:
#                     # First replace all line breaks with spaces and strip space in front or back, then replace multiple whitespace chars by one space
#
#                     # NOTE: There are two variants: Replace \n with space or with nothing - the first might split words, the seconds might melt words together
#                     # Hence, apply both and then take common keyphrases
#                     text = re.sub('\s+',' ',a_file.read().replace('\n','').strip())
#                     #text = re.sub('\s+',' ',a_file.read().replace('\n',' ').strip())
#
#                 # Afterwards, apply phrase extraction
#
#                 # print('Approach 1:')
#                 # keywords = rake.run(text)
#                 # print(keywords)
#                 # print('\nApproach 2:')
#
#                 sentence_list = rake.split_sentences(text)
#                 #stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1
#                 stop_path = 'SmartStoplist.txt'  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
#                 stop_word_pattern = rake.build_stop_word_regex(stop_path)
#
#                 # Generate candidate keywords
#                 phrase_list = rake.generate_candidate_keywords(sentence_list, stop_word_pattern)
#
#                 # Calculate individual word scores
#                 word_scores = rake.calculate_word_scores(phrase_list)
#
#                 # Generate candidate keyword scores
#                 keyword_candidates = rake.generate_candidate_keyword_scores(phrase_list, word_scores)
#                 if rake.debug: print(keyword_candidates)
#
#                 sorted_keywords = sorted(iter(keyword_candidates.items()), key=operator.itemgetter(1), reverse=True)
#                 if rake.debug: print(sorted_keywords)
#
#                 total_keywords = len(sorted_keywords)
#                 # Only take top 1/3.5 of list via range
#                 filtered_keywords = sorted_keywords[0:((int)(total_keywords / 3.5))]
#                 #if debug: print totalKeywords
#                 #print (sorted_keywords[0:((int)(total_keywords / 4))])
#
#                 # Remove keyphrases longer than 4 words (usually meaningless set of words)
#                 result_list = [k for k in filtered_keywords if len(k[0].split(' ')) <= 4] # -> List comprehension (look it up)
#                 with open(os.path.join(root, file)+'_keyphrases.txt', mode='a', encoding='utf-8') as a_file:
#                     for keyphrase in result_list:
#                         # Split keyphrase into words and only merge those back together that contain no special characters or 0-9
#                         temp = ' '.join([k for k in keyphrase[0].split(' ') if len(re.findall("[^A-Za-z-]+", k)) == 0])
#                         if temp != '': # In case we removed every word, don't write cleaned keyphrase to file, otherwise do
#                             #keyphrase = tuple([temp,keyphrase[1]])
#                             a_file.write(temp + ',' + (str)(keyphrase[1]) + '\n')
#                             #print(temp) # Debug print statement
#
#                 # We could produce duplicates by removing words with special characters
#                 # (example: "concise oxford dictionary,8.5")
#                 # WONTFIX for us, because duplicates are removed at a later stage when keyphrases are merged
#
#                 # for keyphrase in filtered_keywords:
#                 #     if len(re.findall("[^A-Za-z0-9]+", keyphrase[0].split(' '))) > 0:
#                 #         print('yeah')
#
#                 # print(result_list)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', help='Port, default is 4124')
    args = p.parse_args()

    # Setup server
    host = '0.0.0.0'
    port = int(args.port) if args.port else 4124
    path = '/keyphrase'

    api.add_resource(Ping, path + '/ping')
    api.add_resource(RAKEPy3, path + '/extract')

    app.run(host=host, port=port)
