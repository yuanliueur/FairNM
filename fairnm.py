import pandas as pd
import itertools
import torch
import pickle
import collections.abc
import copy
import unicodedata
import re
import numpy as np
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

class FairNM:
    def __init__(self, model_path, vocab_path, name_weights_path):
        self.model = torch.load(model_path, map_location='cpu')
        with open(vocab_path, "rb") as handle:
            self.train_vocab = pickle.load(handle)
        self.name_weights = pd.read_csv(name_weights_path, header=None)
        self.name_weights_dict = dict(zip(self.name_weights.iloc[:, 0], self.name_weights.iloc[:, 1]))
        self.maximum_name_weight = max(self.name_weights_dict.values())

    def get_ngrams(self, name, padded_bool, n):
        """ Takes in a name and return the padded n-grams

        Parameters
        ----------
        name (str) : The name to turn into n-grams
        padded_bool (bool) : Turn on/off padding
        n (int) : n-gram order

        Returns
        -------
        (list) : a list of n-grams
        """

        if padded_bool == True:
            padded = ' '*(n-1) + name + ' '*(n-1)

        else:
            padded = name

        ngrams = []

        for i in range(len(padded)-n+1):
            ngrams.append(padded[i:i+n])

        return list(set(ngrams))
    
    def inverse_index_encoder(self, df, columnname):
        index_dict = {}        

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Extract the encodings from the specified column
            encodings = row[columnname]

            # Iterate over each encoding
            for encoding in encodings:
                # Check if the encoding already exists as a key in the dictionary
                if encoding in index_dict:
                    # If it does, append the value to the list of values for that key
                    index_dict[encoding].append(row['db_code'])
                else:
                    # If it doesn't, create a new key-value pair in the dictionary
                    index_dict[encoding] = [row['db_code']]
                
        return index_dict
    
    def recall_filter(self, threshold, query_list, indexed_scr, scr_namelist):
        candidates = set()
        
        # query_list has {db_code : (name, person_id)}
        for k, v in query_list.items():
            # get padded tri-grams and number of tri-grams of query
            ngrams = self.get_ngrams(v[0], True, n=3)
            len_ngrams = len(ngrams)
            
            # temporary dict that counts the number of overlapping trigrams with a match
            temp_dict = {}
            
            # Loop over the grams of the name and check whether the key is in the screening list
            for gram in ngrams:
                if gram in indexed_scr.keys():
                    # Now loop over the documents with this index in the screening list and count the occurrences
                    for db_code in indexed_scr[gram]:
                        if db_code in temp_dict.keys():
                            temp_dict[db_code] += 1
                        else:
                            temp_dict[db_code] = 1
                            
            for screen_code, occurrence_count in temp_dict.items():
                shortest_gram = min(len_ngrams, scr_namelist[screen_code][0])

                score = occurrence_count / shortest_gram
                
                if score > threshold:
                    match = 0 
                    
                    if scr_namelist[screen_code][1] == v[1]:
                        match = 1
                        
                    candidates.add((k, screen_code, match, score))
        return candidates

    def generate_merged_names(self, names):
        merged_names = []
        locations = list(range(len(names)))

        for r in range(1, len(names) + 1):
            combinations = itertools.combinations(locations, r)

            for combination in combinations:
                merged_name = ''.join([names[idx] for idx in combination])
                merged_names.append((merged_name, list(combination)))

        return merged_names


    def string_split(self, x, prefix_suffix=["<", ">"]):
        """
        Tokenize the input names and add prefix and suffix
        """
        tokenized_str = []

        x_bounded = copy.deepcopy(x)

        if isinstance(prefix_suffix, collections.abc.Sequence) and len(prefix_suffix) == 2:
            prefix = prefix_suffix[0] if isinstance(prefix_suffix[0], str) else ""
            suffix = prefix_suffix[1] if isinstance(prefix_suffix[1], str) else ""
            x_bounded = prefix + x + suffix

        tokenized_str += [sub_x for sub_x in x_bounded]

        tokenized_str = [t for t in tokenized_str if t]
        return tokenized_str

    def normalizeString(self, s, uni2ascii=True, lowercase=True, strip=True, only_latin_letters=False):
        """
        Normalize the names
        """
        # Convert input string to ASCII:
        if uni2ascii:
            s = unicodedata.normalize("NFKD", str(s))
        # Convert input string to lowercase:
        if lowercase:
            s = s.lower()
        # Remove trailing whitespace:
        if strip:
            s = s.strip()
        # Remove non-latin letters:
        if only_latin_letters:
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def pad_data(self, s, maxlen):
        padded = np.zeros((maxlen,), dtype=np.int64)
        if len(s) > maxlen:
            padded[:] = s[: maxlen]
        else:
            padded[: len(s)] = s
        return padded

    def tokenize_names(self, name1, name2, missing_char_threshold=0.5, preproc_steps=(True, True, True, False), max_seq_len=10):
        """
        Embed the tokenizes input names
        """

        dataset_pd = pd.DataFrame([[name1, name2, True]], columns=["s1", "s2", "label"])

        dataset_pd["s1_unicode"] = dataset_pd["s1"].apply(
            self.normalizeString, args=preproc_steps
        )
        dataset_pd["s2_unicode"] = dataset_pd["s2"].apply(
            self.normalizeString, args=preproc_steps
        )

        dataset_pd["s1_tokenized"] = dataset_pd["s1_unicode"].apply(
            lambda x: self.string_split(
                x,
            )
        )
        dataset_pd["s2_tokenized"] = dataset_pd["s2_unicode"].apply(
            lambda x: self.string_split(
                x,
            )
        )

        s1_tokenized = dataset_pd["s1_tokenized"].to_list()
        s2_tokenized = dataset_pd["s2_tokenized"].to_list()

        s1_indx = [
            [self.train_vocab.tok2index[tok] for tok in seq if tok in self.train_vocab.tok2index]
            for seq in s1_tokenized
        ]
        s2_indx = [
            [self.train_vocab.tok2index[tok] for tok in seq if tok in self.train_vocab.tok2index]
            for seq in s2_tokenized
        ]

        to_be_removed = []
        for i in range(len(s1_indx) - 1, -1, -1):
            if (
                (1 - len(s1_indx[i]) / max(1, len(s1_tokenized[i])))
                > missing_char_threshold
                or (1 - len(s2_indx[i]) / max(1, len(s2_tokenized[i])))
                > missing_char_threshold
                or len(s1_tokenized[i]) == 0
                or len(s2_tokenized[i]) == 0
            ):
                to_be_removed.append(i)
                del s1_indx[i]
                del s2_indx[i]

        dataset_pd.reset_index(inplace=True)

        dataset_pd["s1_indx"] = s1_indx
        dataset_pd["s2_indx"] = s2_indx

        dataset_pd.reset_index(drop=True, inplace=True)

        dataset_pd["s1_indx_pad"] = dataset_pd.s1_indx.apply(self.pad_data, maxlen = max_seq_len)
        dataset_pd["s2_indx_pad"] = dataset_pd.s2_indx.apply(self.pad_data, maxlen = max_seq_len)

        dataset_pd["s1_len"] = dataset_pd.s1_indx.apply(lambda x: max_seq_len if len(x) > max_seq_len else len(x))
        dataset_pd["s2_len"] = dataset_pd.s2_indx.apply(lambda x: max_seq_len if len(x) > max_seq_len else len(x))

        x1 = dataset_pd.s1_indx.apply(self.pad_data, maxlen = max_seq_len)
        x2 = dataset_pd.s2_indx.apply(self.pad_data, maxlen = max_seq_len)

        s1_len = dataset_pd.s1_indx.apply(lambda x: max_seq_len if len(x) > max_seq_len else len(x))
        s2_len = dataset_pd.s2_indx.apply(lambda x: max_seq_len if len(x) > max_seq_len else len(x))

        return x1, s1_len, x2, s2_len, max_seq_len

    def SNM_inference(self, name1, name2):
        x1, len1, x2, len2, max_len  = self.tokenize_names(
            name1, name2,
        )
        """
        Inference of the Short Name Module
        """

        x1 = torch.tensor(np.array(x1[0]))
        x1 = torch.reshape(x1, (max_len, 1))
        len1 = np.array(len1)

        x2 = torch.tensor(np.array(x2[0]))
        x2 = torch.reshape(x2, (max_len, 1))
        len2 = np.array(len2)

        with torch.no_grad():
            pred = self.model(
                x1,
                len1,
                x2,
                len2,
                pooling_mode='hstates_layers_simple',
                device='cpu',
                output_state_vectors=False,
                evaluation=True,
            )
            pred_softmax = F.softmax(pred, dim=-1)

        return pred_softmax[0, 1].item()

    def weights_per_name(self, name):
        name_parts = name
        name_scoring = []

        for name in name_parts:
            name_scoring.append(self.name_weights_dict.get(name, self.maximum_name_weight))

        total = sum(name_scoring)
        normalized_scores = [x / total for x in name_scoring]

        return normalized_scores

    def weighted_score(self, bestMatch):
        total_weight = 0
        weight_per_match = []

        for k, v in bestMatch.items():
            weight1 = self.name_weights_dict.get(v[0], self.maximum_name_weight)
            weight2 = self.name_weights_dict.get(v[1], self.maximum_name_weight)

            total_weight += (weight1 + weight2)
            weight_per_match.append(weight1 + weight2)

        norm_weight_per_match = [x / sum(weight_per_match) for x in weight_per_match]

        weighted_score = 0
        score_count = 0

        for k, v in bestMatch.items():
            weighted_score += norm_weight_per_match[score_count] * v[4]
            score_count += 1

        return weighted_score


    def nameMatcher(self, name1, name2, weighting_bool=True, verbose=False, GRU_unit=True):
        # Split names into name parts
        name_list1 = name1.split()
        name_list2 = name2.split()

        # Ensure name1 is the shorter name
        if len(name_list1) > len(name_list2):
            name_list1, name_list2 = name_list2, name_list1

        # Generate all potential name token combination, including merged tokens
        merged1 = self.generate_merged_names(name_list1)
        merged2 = self.generate_merged_names(name_list2)

        if verbose:
            print('TOKENS')
            print(merged1, '\n', '\n', merged2)
            print()

            print('ALL COMBINATIONS')

        dict_temp = {}
        double_match = {}

        # Store all results in {index1 : [result2.1, result2.2]} format
        all_tuples = {}

        # store all similarity scores
        for combination in itertools.product(merged1, merged2):
            namepart1, locpart1 = combination[0]
            namepart2, locpart2 = combination[1]

            # Change similarity scoring here, for now: Levenshtein normalized similarity
            sim_score = Levenshtein.normalized_similarity(namepart1, namepart2)

            # Short name module
            if (len(namepart1) <= 3 and len(namepart2) <= 2) or (len(namepart2) <= 3 and len(namepart1) <= 2) and GRU_unit == True:
                sim_score = self.SNM_inference(namepart1, namepart2)

            # Append to results or create new key
            if tuple(locpart1) in all_tuples.keys():
                all_tuples[tuple(locpart1)].append((namepart1, namepart2, tuple(locpart1), tuple(locpart2), sim_score))
            else:
                all_tuples[tuple(locpart1)] = [(namepart1, namepart2, tuple(locpart1), tuple(locpart2), sim_score)]

        # Sort the lists of scores from best to worst match
        for key, value in all_tuples.items():
            value.sort(key=lambda x: x[4], reverse=True)

        # Store the best found match for index_namelist1
        bestMatch = {}

        # Store the combination in index_namelist2 : index_namelist2 format
        matchedNames = {}

        for k, v in all_tuples.items():
            # If we didn't already match with this name_index2, add it to the dictionaries
            if v[0][3] not in matchedNames.keys():
                bestMatch[k] = v[0]
                matchedNames[v[0][3]] = k

            # If we already found a match for this name_index2, resolve the colusion
            else:
                nextBestMatch = 1

                while v[nextBestMatch][3] in matchedNames.keys():
                    nextBestMatch += 1

                bestMatch[k] = v[nextBestMatch]
                matchedNames[v[nextBestMatch][3]] = k

        keys_to_remove = set()
        for key in bestMatch.keys():

            if len(key) > 1:
                individual_keys = list(key)
                individual_scores = []

                # Check if individual keys exist in the dictionary before accessing their scores
                for individual_key in individual_keys:
                    if (individual_key,) in bestMatch:
                        individual_scores.append(bestMatch[(individual_key,)][4])

                if len(individual_scores) == len(individual_keys) and bestMatch[key][4] <= max(individual_scores):
                    keys_to_remove.add(key)

        # Remove the keys we didn't match
        for key in keys_to_remove:
            bestMatch.pop(key)

        scores = [value[4] for value in bestMatch.values()]

        # Calculate the weighted score between the matched names
        if weighting_bool == True:
            final_score = self.weighted_score(bestMatch)

        else:
            final_score = sum(scores) / len(name_list1)

        if verbose:
            print(bestMatch)
            print(scores)
            print()
            print('Similarity Score NEW:', final_score)
            print()

        return final_score
