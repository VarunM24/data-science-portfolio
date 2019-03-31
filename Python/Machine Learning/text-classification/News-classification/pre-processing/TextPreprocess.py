
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet
import re
from nltk.tokenize import MWETokenizer
# nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from itertools import chain
from nltk.probability import FreqDist
# nltk.download('stopwords')
import pickle
import time
import itertools
import random
import traceback
import numpy as np
import pandas as pd
import numpy as np

class Preprocess:

    def get_in_paragraph(self,raw_data):
        paragraphs = []
        for i in range(len(raw_data)):
            if i%3 == 1:
                paragraphs.append(raw_data[i].strip('\n')[5:]) #.lower()   - removed lowering the case right now as POS tagging may use it
        return paragraphs

    def tokenize_paragraph(self,data):
        return [nltk.word_tokenize(each) for each in data]

    def too_short_item(self,tokens,threshold,threshold2,lim):
        a_list = []
        max_no = 0
        print_lim = 10
        cnt = 0
        for each in range(len(tokens)):
        
            if max_no==lim:                
                    break        
            if (len(tokens[each]) < threshold) & (len(tokens[each]) > threshold2):
                if cnt < print_lim :
                    print('length of token,',str(len(tokens[each])))
                    print('index -',str(each))
                   # print('token =',tokens[each],labels[each])
                    cnt +=1
                a_list.append(each)
                max_no =max_no + 1        
        print('total number ',str(max_no))    
        return a_list


    def add_POS(self,list_document_sentences):
        # generate POS tag for each token
        for i in range(len(list_document_sentences)):
            tagged_sents = []
            try:
                for j in range(len(list_document_sentences[i])):
                    uni_sent = nltk.word_tokenize(list_document_sentences[i][j])
                    tagged_sent = nltk.tag.pos_tag(uni_sent)
                    tagged_sents.append(tagged_sent)
                list_document_sentences[i] =  tagged_sents   
            except Exception:
                traceback.print_exc()
                print("This is an error message!")
                print(list_document_sentences[i])
                print(i)
                break
            
    def get_wordnet_pos(self,treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    # Removes specified POS from sentences
    def filter_POS(self,POSed_doc_list,POS_tags_to_discard_list):
    
        final_tokens_dict=[[[tagged_set for tagged_set in sents if tagged_set[1] not in POS_tags_to_discard_list] for sents in doc_i]
                          for doc_i in POSed_doc_list]
    
        return final_tokens_dict 

    def lemmatization(self,list_tokens): # modified for sentence based lemmatization
        lemmatizer = nltk.stem.WordNetLemmatizer()
        final_tokens_dict =[0]*len(list_tokens)
        for i in range(len(list_tokens)):
    #         list_tokens[i] = [lemmatizer.lemmatize(each[0], get_wordnet_pos(each[1])) for each in list_tokens[i]]
            final_tokens =[]
            for tagged_set in list_tokens[i]:
                final_tokens = final_tokens + [lemmatizer.lemmatize(w[0], self.get_wordnet_pos(w[1])) for w in tagged_set ]
            final_tokens_dict[i] = final_tokens
        return final_tokens_dict

    # replace 1 word doc to specified word
    def change_oneworddoc_to(self,all_docs,word_to_change_to):
        # index for words less than 2 and more than 0
        idxs = self.too_short_item(all_docs, 2,0,1000000)
        print('no of one word docs',str(len(idxs)))
        print('percentage of one word docs',str(len(idxs)/len(all_docs)))
        print('some of the indexes of 1 word docs \n',idxs[0:10])
    
        lst=list(all_docs)
        # Replacing
        target = [word_to_change_to]*len(idxs)
        pos = idxs
        for x,y in zip(pos,target):
            lst[x] = [y]
        #print those one word documents
        # print( [all_docs[i] for i in idxs])
        return lst

    def convert_lowercase(self,doclist):
    
        return [[word.lower() for word in doc ] for doc in doclist]


    def get_all_tokens(self,list_tokens):

        return list(itertools.chain.from_iterable(list_tokens))

    # DIDNT USE FOR CNN
    def get_n_grams(self,tokens, N, number_of_most_common):
        # generte bigrams
        bigrams = nltk.ngrams(tokens, n = N)
        bigrams_fd = nltk.FreqDist(bigrams)
        # have a look at the top 200 bigrams
        bigrams_common = [each[0] for each in bigrams_fd.most_common(number_of_most_common)]
        return bigrams_common

    def get_bi_tri_collocations_filtered_freq_bmi_perc(self,tokens,trigram=False, N_best_bigram=1000,N_best_trigram=1000, min_corpus_bigram_freq=5,
                                                   min_corpus_trigram_freq=5,
                                           remove_stopwords=True,remove_symbols=True):
    
        print('starting to get all tokens')
        # combine all tokens from all documents
        lfinal_tokens_combined = self.get_all_tokens(tokens)
        print('Got all tokens')
        # generate bigrams
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        print('getting all bigram collocations')
        finder2 = nltk.collocations.BigramCollocationFinder.from_words(lfinal_tokens_combined)
        print('got all bigram collocations')
        #applying filter to remove collocations that occur less than 5 times in the corpus
        finder2.apply_freq_filter(min_corpus_bigram_freq)
        bigrams = finder2.nbest(bigram_measures.pmi, N_best_bigram)
        bigramlist = bigrams
        if(trigram):
            # generate bigrams
            trigram_measures = nltk.collocations.TrigramAssocMeasures()
            print('getting all trigram collocations')
            finder3 = nltk.collocations.TrigramCollocationFinder.from_words(lfinal_tokens_combined)
            print('got all trigram collocations')
            #applying filter to remove collocations that occur less than 5 times in the corpus
            finder3.apply_freq_filter(min_corpus_trigram_freq)
            trigrams = finder3.nbest(trigram_measures.pmi, N_best_trigram)
            trigramlist = trigrams
        if(remove_stopwords):
        
            stopwords_list1 = set(nltk.corpus.stopwords.words('english'))
            print('remving stopwords from all bigram collocations')
            bigramlist = [bigrampair for bigrampair in bigrams if (bigrampair[0] not in stopwords_list1  ) and (bigrampair[1] not in stopwords_list1)]    
        
            print('remved stopwords from all bigram collocations')
            if(trigram):
                print('remving stopwords from all trigram collocations')
                trigramlist = [trigrampair for trigrampair in trigrams if (trigrampair[0] not in stopwords_list1  ) and (trigrampair[1] not in stopwords_list1) 
                              and (trigrampair[2] not in stopwords_list1)]    
                print('remved stopwords from all trigram collocations')
            
        if(remove_symbols):
            print('remving symbols from all collocations')
            bigramlist = [bigrampair for bigrampair in bigrams if ( ((bigrampair[0].isalpha() and not re.search('\W', bigrampair[0]) and not re.search('^.$', bigrampair[0])) or 
            ('-' in bigrampair[0] and not re.search('[^a-zA-Z-]', bigrampair[0]) and not re.search('^-.+|.+-$|^-+$|^-$', bigrampair[0]) and not re.search('^.$', bigrampair[0])) )) 
            
            and  ((bigrampair[1].isalpha() and not re.search('\W', bigrampair[1]) and not re.search('^.$', bigrampair[1])) or ('-' in bigrampair[1] and not re.search('[^a-zA-Z-]', bigrampair[1]) and not re.search('^-.+|.+-$|^-+$|^-$', bigrampair[1]) and not re.search('^.$', bigrampair[1])) )] 
            
            if(trigram):
                trigramlist = [trigrampair for trigrampair in trigrams if ( ((trigrampair[0].isalpha() and not re.search('\W', trigrampair[0]) and not re.search('^.$', trigrampair[0])) or ('-' in trigrampair[0] and not re.search('[^a-zA-Z-]', trigrampair[0]) and not re.search('^-.+|.+-$|^-+$|^-$', trigrampair[0]) and not re.search('^.$', trigrampair[0])) )) 
                
                and  ((trigrampair[1].isalpha() and not re.search('\W', trigrampair[1]) and not re.search('^.$', trigrampair[1])) or ('-' in trigrampair[1] and not re.search('[^a-zA-Z-]', trigrampair[1]) and not re.search('^-.+|.+-$|^-+$|^-$', trigrampair[1]) and not re.search('^.$', trigrampair[1])) )
                
                and ((trigrampair[2].isalpha() and not re.search('\W', trigrampair[2]) and not re.search('^.$', trigrampair[2])) or ('-' in trigrampair[2] and not re.search('[^a-zA-Z-]', trigrampair[2]) and not re.search('^-.+|.+-$|^-+$|^-$', trigrampair[2]) and not re.search('^.$', trigrampair[2])) )]    
            print('remved symbols from all collocations')
        collocations_common = bigramlist
    
        if(trigram):
            collocations_common = trigramlist + collocations_common 
        print(collocations_common[1:20])
        print(collocations_common[-20:])
        print('All done - total collocations = ')
        print(len(collocations_common))
        return collocations_common

    def remove_stopwords(self,list_tokens):
        #introduce the nltk stopwords
        stopwords_list = nltk.corpus.stopwords.words('english')
        stopwords_set = set(stopwords_list) #set is fast in searching

        #remove stopwords from tokens
        for i in range(len(list_tokens)):
            list_tokens[i] = [each for each in list_tokens[i] if each not in stopwords_set]

    # Default version is also going to remove words with hyphenation so I modified it also will remove single letter words
    def remove_punctuation(self,list_tokens):
        for i in range(len(list_tokens)):
    #         list_tokens[i] = [each for each in list_tokens[i] if each.isalpha()]   
            list_tokens[i] = [tokens for tokens in list_tokens[i] if ((tokens.isalpha() and not re.search('\W', tokens) and not re.search('^.$', tokens)) or 
                                                              ('-' in tokens and not re.search('[^a-zA-Z-]', tokens) and not re.search('^-.+|.+-$|^-+$|^-$', tokens)
                                                              and not re.search('^.$', tokens)) )]

    #Re-tokenizing tokens with Multiple Words Expression Tokenizer
    def introduce_n_grams_in_docs(self,list_tokens, new_N_grams):
        mwe_tokenizer = nltk.tokenize.MWETokenizer(new_N_grams)
        for i in range(len(list_tokens)):
            list_tokens[i] = mwe_tokenizer.tokenize(list_tokens[i])

    # filters if word occurs in more than eg. 20% of number of documents also returns the words filtered
    def filtered_words_by_perc_occur_corpus_more_less_than(self,list_tokens,perc_value,percent=True,more = True):
        num_of_doc_threshold = round((perc_value*len(list_tokens))/100)
        
        if(not percent):
            num_of_doc_threshold = perc_value
        print('num_of_doc_threshold',str(num_of_doc_threshold))

        unibi_conv_tokens_dict = {}
      
        list_tokens1 =[ [ tuple(token.split('_')) if '_' in token else token for token in doc ] for doc in list_tokens]

        words_2 = list(chain.from_iterable([set(value) for value in list_tokens1]))
        fd_2 = FreqDist(words_2)

        print(fd_2)

        if(more):
            certainFreqWords = set([k for k, v in fd_2.items() if v > num_of_doc_threshold])
        else:
            certainFreqWords = set([k for k, v in fd_2.items() if v < num_of_doc_threshold])

        return certainFreqWords


    def remove_words_from_ngramlist(self,ngramlist,filter_words):
        # removes list of words from ngramlist and docs which may contain tuples representing ngrams and returns filtered list of docs
        # removing required tokens  from bigram list
        filter_words1 = set(filter_words )
        ngramlist1 = [ngram for ngram in ngramlist if ngram not in filter_words]   
        return ngramlist1
                   
    def remove_words_tuples_corpus(self,doclist,updated_ngramlist,filter_words):
        # accepts doclist which hasnt been updated with mwetokenizer yet              
    
        #removing required tokens  from bigram list
        print('total lenth of doclist = ',len(doclist))
        print('total words in filter list',len(filter_words))
        start_time = time.time()
        filter_words = set(filter_words)        
        doclist1 = [[z for z in doc if z not in filter_words] for doc in doclist]  
        print("--- %s seconds for removal ---" % (time.time() - start_time))
    
        #combining the filtered bigram list and doclist using mwetokenizer 
        start_time = time.time()
        mwe_tokenizer = MWETokenizer(updated_ngramlist)
        doclist1 = [mwe_tokenizer.tokenize(doc) for doc in doclist1]
        print("--- %s seconds for MWE ---" % (time.time() - start_time))
        return doclist1
  

     # filters if word occurs in more than % eg. 20% of class
    def words_to_filter_by_perc_occur_labels_more_than(self,list_tokens,labels,perc):
        no_class= len(set(labels))
        comb_features = [[]]*no_class
    
        for i in range(len(list_tokens)):
            comb_features[int(labels[i][1:])-1] = comb_features[int(labels[i][1:])-1] +  list_tokens[i]
        wrds_to_filter = self.filtered_words_by_perc_occur_corpus_more_less_than(list_tokens=comb_features,perc_value=perc,percent=True,more = True)
        return wrds_to_filter
    
    
    
    def create_comb_features(self,list_tokens,labels):
        no_class= len(set(labels))
        comb_features = [[]]*no_class
       
        for i in range(len(list_tokens)):
            comb_features[int(labels[i][1:])-1] = comb_features[int(labels[i][1:])-1] +  list_tokens[i]
        
        return comb_features
        
    def mutual_class_filter(self,list_tokens,labels,filter_labels):
        no_class= len(set(labels))
        comb_features = [[]]*no_class
       
        for i in range(len(list_tokens)):
            comb_features[int(labels[i][1:])-1] = comb_features[int(labels[i][1:])-1] +  list_tokens[i]
        new_comb_feature =[]
        for i in filter_labels:
            new_comb_feature.append(comb_features[int(i[1:])-1])
        
        wrds_to_filter = self.filtered_words_by_perc_occur_corpus_more_less_than(list_tokens=new_comb_feature,perc_value=1,percent=False,more = True)
        list_tokens1 = list(list_tokens)
        
        print('removing words...')
        cnt = 0
        for wrds in wrds_to_filter:
            if cnt <10:
                print(wrds)
                cnt = cnt+1
            else:
                break
     
        # get index of all docs which have given filter labels
        idx_arr = [idx for idx, e in enumerate(labels) if e in filter_labels]
        print('affected indexes...')
        cnt = 0
        for idxs in idx_arr:
            if cnt <10:
                print(idxs)
                cnt = cnt+1
            else:
                break
     
        idx_set = set(idx_arr)
        filter_words = set(wrds_to_filter)    
        for doc_i in idx_arr:
            list_tokens1[doc_i]    =    [word for word in list_tokens1[doc_i] if word not in filter_words]    
        
        return list_tokens1
     

    def create_vocab_revocab_dict(self,doclist):
        all_tokens = self.get_all_tokens(doclist)
        all_token_set = set(all_tokens)
        vocab_dict = dict(enumerate(all_token_set))
        rev_vocab_dict = {v:k for k, v in vocab_dict.items()}
        return vocab_dict,rev_vocab_dict

    def find_empty_or_1specific_word_token_idx(self,doclist,empty=True,wrd=None):
        emptytokens_idx =[]
        for token_idx in range(len(doclist)):
            if(empty):
                if doclist[token_idx] == []:
                    emptytokens_idx.append(token_idx)
            else:
                if doclist[token_idx] == [wrd]:
                    emptytokens_idx.append(token_idx)
        perc = len(emptytokens_idx)/len(doclist)
        print('percentage of such words in doclist:',perc)
        return emptytokens_idx

    def remove_less_frequent_words(self,list_tokens, number_of_occurence):
        list_token_set = [set(each) for each in list_tokens] #change each list to set, just because set s faster
        counted_set ={}
        for i in range(len(list_tokens)):
            for each_token in list_tokens[i]:
                if each_token not in counted_set:
                    #initial count
                    count = 0
                    for each_token_set in list_token_set:
                        if each_token in each_token_set:
                            count += 1
                    if count <= number_of_occurence: # 'count is 1' means that it only appear in one abstract
                        list_tokens[i].remove(each_token)
                    else:
                        counted_set.add(each_token)

    def replace_idx_word(self,doclist, idxs , word):
        idxs = idxs
        print('no of ',word,' in docs',str(len(idxs)))
        print('percentage of ',word,' word docs',str(len(idxs)/len(doclist)))
        print('some of the indexes of ',word,' word docs \n',idxs[0:10])
    
        lst=list(doclist)
        # Replacing
        target = [word]*len(idxs)
        pos = idxs
        for x,y in zip(pos,target):
    #         print(x)
            lst[x] = [y]
        return lst

    # Joining the documents into string for TFIDF Vectorizing 
    def finalizing_document_list(self,doclist):
        newdoclist = list(doclist)
        for i in range(len(newdoclist)):
            newdoclist[i] = ' '.join(newdoclist[i])
        
        return newdoclist
