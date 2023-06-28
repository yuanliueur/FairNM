import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class TestBench:
    sns.set_theme()

    def run(self, test_df, fair_nm, name_matcher, verbose=True):
        """ Function that execute the test bench

        Parameters
        ----------
        test_df (dataframe) : the sample dataframe output of the augmented_dataframe function
        fair_nm (class) : FairNM() class instance
        name_matcher (func) : the chosen similarity function

        Returns
        -------
        TP_FP (dict) : dictionary of True and False Positives for all name variations and language codes
        """
        
        thresholds = np.linspace(0, 0.99)

        # Extract utilized languages and name variations
        list_of_langs = test_df['language_code'].unique()
        included_variations = test_df.columns[2:]

        # Add 'database code' to find back the record
        db_codes = ['d{}'.format(i) for i in range(0, len(test_df))]
        test_df['db_code'] = db_codes
        test_db_id = test_df.set_index('db_code')['person_id'].to_dict()

        # Make a query database for the full sample database {db_code: [name, person_id]}
        db_dict = test_df.set_index('db_code').to_dict(orient='index')
        query_dict = {}

        for key, value in db_dict.items():
            query_dict[key] = [value['full_name'], value['person_id']]

        query_list = test_df[['db_code', 'person_id', 'full_name']]

        final_results = {variation: [] for variation in included_variations}

        for var in included_variations:
            if verbose:
                print(var)

            test_df[var + 'NG'] = test_df[var].apply(lambda x: fair_nm.get_ngrams(x, True, 3))
            results_per_lang = {language: [] for language in list_of_langs}
            
            lang_iterable = tqdm(list_of_langs, desc="Processing") if verbose else list_of_langs

            for lang in lang_iterable:
                screening_df = test_df[test_df['language_code'] == lang]
                db_codes = ['s{}'.format(i) for i in range(0, len(screening_df))]
                screening_df.loc[:, 'db_code'] = db_codes

                # Maps ids to person_id
                screen_db_id = screening_df.set_index('db_code')['person_id'].to_dict()

                # Make indexed screening df per ethn
                indexed_SL = fair_nm.inverse_index_encoder(screening_df, var + 'NG')
                namelist_SL = {row['db_code']: [len(row[var + 'NG']), row['person_id']] for _, row in screening_df.iterrows()}

                t_test = 0.50
                candidates = fair_nm.recall_filter(t_test, query_dict, indexed_SL, namelist_SL)
                for pair in candidates:
                    query_name = test_df.iloc[test_db_id[pair[0]]]['full_name']
                    screen_name = test_df.iloc[screen_db_id[pair[1]]][var]

                    sim_score = name_matcher(query_name, screen_name)

                    results_per_lang[lang].append([query_name, screen_name, sim_score, pair[2]])

            TP_per_lang = {language: [] for language in list_of_langs}
            FP_per_lang = {language: [] for language in list_of_langs}

            for lang in list_of_langs:
                for t in thresholds:
                    TP_count = 0
                    FP_count = 0

                    for res in results_per_lang[lang]:
                        if res[2] > t and res[3] == 1:
                            TP_count += 1
                        elif res[2] > t and res[3] == 0:
                            FP_count += 1

                    TP_per_lang[lang].append(TP_count)
                    FP_per_lang[lang].append(FP_count)

            final_results[var] = [TP_per_lang, FP_per_lang]

        return final_results


    def TP_FP_to_Measures(self, TP_FP_res, sample_size=3000):
        # Store total true and false positives over all name variations
        total_TPs = {}
        total_FPs = {}

        for var, res in TP_FP_res.items():
            TP_res = res[0]  # dict {'ARAB': [TPs], 'BRI' : [TPs]}
            FP_res = res[1]  # dict {'ARAB': [FPs], 'BRI' : [FPs]}

            for lang in TP_res.keys():
                if lang not in total_TPs.keys():
                    # initialize total TPs and FPs with zeros
                    total_TPs[lang] = np.zeros(len(TP_res[lang]))
                    total_FPs[lang] = np.zeros(len(TP_res[lang]))

                total_TPs[lang] = [x + y for x, y in zip(total_TPs[lang], TP_res[lang])]
                total_FPs[lang] = [x + y for x, y in zip(total_FPs[lang], FP_res[lang])]

        # Turn total TP and FP into accuracy measures
        precision_per_lang = {lang: [] for lang in total_TPs.keys()}
        recall_per_lang = {lang: [] for lang in total_TPs.keys()}
        F1_per_lang = {lang: [] for lang in total_TPs.keys()}

        for lang in total_TPs.keys():
            # precision is TP / (TP + FP)
            precision_per_lang[lang] = [x / (x + y) for x, y in zip(total_TPs[lang], total_FPs[lang])]

            # recall is TP / sample_size
            recall_per_lang[lang] = [x / (len(TP_FP_res) * sample_size) for x in total_TPs[lang]]

            # F1 is 2 * (precision * recall) / (precision + recall)
            F1_per_lang[lang] = [2 * (x * y) / (x + y) for x, y in
                                 zip(precision_per_lang[lang], recall_per_lang[lang])]

        # Calculate Fairness score
        Fairness = []

        for i in range(len(list(precision_per_lang.values())[0])):
            max_prec = 0
            min_prec = 1

            max_rec = 0
            min_rec = 1

            for lang in precision_per_lang.keys():
                if precision_per_lang[lang][i] > max_prec:
                    max_prec = precision_per_lang[lang][i]
                if precision_per_lang[lang][i] < min_prec:
                    min_prec = precision_per_lang[lang][i]

                if recall_per_lang[lang][i] > max_rec:
                    max_rec = recall_per_lang[lang][i]
                if recall_per_lang[lang][i] < min_rec:
                    min_rec = recall_per_lang[lang][i]

            Fairness.append(1 - ((max_prec - min_prec) + (max_rec - min_rec)) / 2)

        return [precision_per_lang, recall_per_lang, F1_per_lang, Fairness]


    def visualizer(self, measuresResults, MoI, sep):
        valid_MoI = ['Precision', 'Recall', 'F1', 'Overview', 'Fairness']

        if MoI not in valid_MoI:
            raise ValueError(
                'Invalid input value. Please choose one of the following options: {}'.format(valid_MoI))

        thresholds = np.linspace(0, 0.99)
        if MoI == 'Precision':
            fig, ax = plt.subplots()
            if sep:
                for code, values in measuresResults[0].items():
                    ax.plot(thresholds, values, label=code)
                ax.set_title('Showing Precision per Language Code')

            if not sep:
                mean_values = np.mean(list(measuresResults[0].values()), axis=0)
                ax.plot(thresholds, mean_values, label='Overall')
                ax.set_title('Showing Overall Precision')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 0.5), loc='center', ncols=1)

        elif MoI == 'Recall':
            fig, ax = plt.subplots()
            if sep:
                for code, values in measuresResults[1].items():
                    ax.plot(thresholds, values, label=code)
                ax.set_title('Showing Recall per Language Code')

            if not sep:
                mean_values = np.mean(list(measuresResults[1].values()), axis=0)
                ax.plot(thresholds, mean_values, label='Overall')
                ax.set_title('Showing Overall Recall')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 0.5), loc='center', ncols=1)

        elif MoI == 'F1':
            fig, ax = plt.subplots()
            if sep:
                for code, values in measuresResults[2].items():
                    ax.plot(thresholds, values, label=code)
                ax.set_title('Showing F1 per Language Code')

            if not sep:
                mean_values = np.mean(list(measuresResults[2].values()), axis=0)
                ax.plot(thresholds, mean_values, label='Overall')
                ax.set_title('Showing Overall F1')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 0.5), loc='center', ncols=1)

        elif MoI == 'Fairness':
            fig, ax = plt.subplots()
            ax.plot(thresholds, measuresResults[3], label='Fairness')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            plt.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 0.5), loc='center', ncols=1)

        elif MoI == 'Overview':
            fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(12, 3))

            if sep:
                for code, values in measuresResults[0].items():
                    ax1.plot(thresholds, values, label=code)

                for code, values in measuresResults[1].items():
                    ax2.plot(thresholds, values)

                for code, values in measuresResults[2].items():
                    ax3.plot(thresholds, values)

            if not sep:
                ax1.plot(thresholds, np.mean(list(measuresResults[0].values()), axis=0), label='Overall')
                ax2.plot(thresholds, np.mean(list(measuresResults[1].values()), axis=0))
                ax3.plot(thresholds, np.mean(list(measuresResults[2].values()), axis=0))

            ax4.plot(thresholds, measuresResults[3])

            ax1.set_xlabel('Thresholds')
            ax2.set_xlabel('Thresholds')
            ax3.set_xlabel('Thresholds')
            ax4.set_xlabel('Thresholds')

            ax1.set_ylabel('Precision')
            ax2.set_ylabel('Recall')
            ax3.set_ylabel('F1')
            ax4.set_ylabel('Fairness')

            ax1.set_xlim(0, 1)
            ax2.set_xlim(0, 1)
            ax3.set_xlim(0, 1)
            ax4.set_xlim(0, 1)

            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
            ax3.set_ylim(0, 1)
            ax4.set_ylim(0, 1)

            plt.tight_layout()
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 0.5), loc='center', ncols=1)


