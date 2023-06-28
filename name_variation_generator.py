import random

class nameVariationGenerator:
    def __init__(self):
        self.keyboard = {
            'q': 'was', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
            'y': 'tghu', 'u': 'yjhi', 'i': 'ukjo', 'o': 'ilkp', 'p': 'o',
            'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
            'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop', 'z': 'asx',
            'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }
    
    def fat_finger_replace(self, word):
        # choose a random index in the word
        index = random.randint(0, len(word) - 1)
        
        # get the character at the chosen index
        char = word[index]
        
        no_tries = 0
        
        # repeat until an alphabetic character in the keyboard is found
        while char not in self.keyboard.keys() and no_tries < 20:
            # choose a new random index
            index = random.randint(0, len(word) - 1)
            
            # get the new character at the chosen index
            char = word[index]        
            no_tries += 1
    
        if no_tries >= 19:
            return word
        
        # get the adjacent keys for the chosen character
        adjacent_keys = self.keyboard[char]
        
        # choose a random adjacent key
        replacement = random.choice(adjacent_keys)
        
        # create a new word with the replacement character
        new_word = word[:index] + replacement + word[index + 1:]

        return new_word

    def delete_random_char(self, word):
        # choose a random index in the word
        index = random.randint(0, len(word) - 1)
        
        char = word[index]
        
        no_tries = 0 
        # repeat until an alphabetic character in the keyboard is found
        while char not in self.keyboard.keys() and no_tries < 20:
            # choose a new random index
            index = random.randint(0, len(word) - 1)
            
            # get the new character at the chosen index
            char = word[index]
            no_tries += 1
            
        if no_tries >= 19:
            return word
            
        # remove the character at the chosen index
        new_word = word[:index] + word[index + 1:]
    
        return new_word 

    
def augment_database(names_df, 
                 name_variation_generator,
                 excluded_variations = [], 
                 excluded_langs = ['VIE'], 
                 sample_size = 3000, 
                 random_seed = 123):

    names_df = names_df[~names_df['language_code'].isin(excluded_langs)]
    names_df.loc[:, 'full_name'] = names_df.full_name.replace(r'\s+', ' ', regex=True)

    sample_df = names_df.groupby('language_code').apply(lambda x: x.sample(n=sample_size, random_state = random_seed, replace=False))
    
    sample_df['person_id'] = range(0, len(sample_df))
    sample_df.reset_index(drop=True, inplace=True)
    sample_df = sample_df[['person_id', 'language_code', 'full_name', 'first_name', 'middle_name', 'last_name']]

    fat_finger, random_del, swapped_names = True, True, True
    
    if 'fat_finger' in excluded_variations:
        fat_finger = False        
    
    if 'random_del' in excluded_variations:
        random_del = False
        
    if 'swapped_names' in excluded_variations:
        swapped_names = False

    if fat_finger:
        sample_df['fat_finger'] = sample_df['full_name'].apply(lambda x: name_variation_generator.fat_finger_replace(x))

    if random_del:
        sample_df['random_del'] = sample_df['full_name'].apply(lambda x: name_variation_generator.delete_random_char(x))

    if swapped_names:
        sample_df['swapped_names'] = sample_df.apply(lambda row: ' '.join([row['last_name'], row['first_name']]), axis=1)

    ## easily add more variation
    return sample_df.loc[:, ~sample_df.columns.isin(['first_name', 'middle_name', 'last_name'])]


