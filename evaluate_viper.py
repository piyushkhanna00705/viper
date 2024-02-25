import pandas as pd
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate_viper(csv_path, csv_path_captions=None):
    # Read the data
    df = pd.read_csv(csv_path)

    if csv_path_captions is not None:
        captions_correct_aug_ans=0
        captions_strict_correct=0

        og_strict_correct = 0
        og_correct_aug_ans = 0

        df_captions = pd.read_csv(csv_path_captions)

        captions_strict_correct_df = pd.DataFrame(columns=df_captions.columns)
        captions_lenient_correct_df = pd.DataFrame(columns=df_captions.columns)

        og_strict_correct_df = pd.DataFrame(columns=df.columns)
        og_lenient_correct_df = pd.DataFrame(columns=df.columns)

        #Iterate df_captions and df together
        for i in range(len(df_captions)):
            if str(df['answer'][i]).lower()!=str(df['result'][i]).lower() and str(df_captions['answer'][i]).lower()==str(df_captions['result'][i]).lower() and str(df_captions['img_path'][i])==str(df['img_path'][i]):
                captions_strict_correct+=1
                captions_strict_correct_df = captions_strict_correct_df.append(df_captions.iloc[i])
            if str(df['answer'][i]).lower() not in str(df['result'][i]).lower() and str(df_captions['answer'][i]).lower() in str(df_captions['result'][i]).lower() and str(df_captions['img_path'][i])==str(df['img_path'][i]):
                captions_correct_aug_ans+=1
                captions_lenient_correct_df = captions_lenient_correct_df.append(df_captions.iloc[i])

            if str(df['answer'][i]).lower()==str(df['result'][i]).lower() and str(df_captions['answer'][i]).lower()!=str(df_captions['result'][i]).lower() and str(df_captions['img_path'][i])==str(df['img_path'][i]):
                og_strict_correct+=1
                og_strict_correct_df = og_strict_correct_df.append(df_captions.iloc[i])
            if str(df['answer'][i]).lower() in str(df['result'][i]).lower() and str(df_captions['answer'][i]).lower() not in str(df_captions['result'][i]).lower() and str(df_captions['img_path'][i])==str(df['img_path'][i]):
                og_correct_aug_ans+=1
                og_lenient_correct_df = og_lenient_correct_df.append(df_captions.iloc[i])

        print("Captions Correctly answered lenient: " , captions_correct_aug_ans)
        print("Captions Correctly answered strict: ", captions_strict_correct)
        print("Captions Total questions: ",len(df_captions))
        print("Captions Lenient Accuracy: ",captions_correct_aug_ans/len(df_captions))
        print("Captions Strict Accuracy: ",captions_strict_correct/len(df_captions))

        print("OG Correctly answered lenient: " , og_correct_aug_ans)
        print("OG Correctly answered strict: ", og_strict_correct)
        print("OG Total questions: ",len(df))
        print("OG Lenient Accuracy: ",og_correct_aug_ans/len(df))
        print("OG Strict Accuracy: ",og_strict_correct/len(df))

        captions_strict_correct_df.to_csv(f'results/captions_strict_correct.csv')
        captions_lenient_correct_df.to_csv(f'results/captions_lenient_correct.csv')

        og_strict_correct_df.to_csv(f'results/og_strict_correct.csv')
        og_lenient_correct_df.to_csv(f'results/og_lenient_correct.csv')
        
        return

    correct_aug_ans=0
    float_count=0
    strict_correct=0

    all_incorrect_df = pd.DataFrame(columns=df.columns)
    lenient_correct_samples_df = pd.DataFrame(columns=df.columns)

    for i in tqdm(range(len(df))):
        # if type(df['answer'][i])!=float:
        #     float_count+=1
        if str(df['answer'][i]).lower()==str(df['result'][i]).lower():
            strict_correct+=1
        if str(df['answer'][i]).lower() in str(df['result'][i]).lower():
            correct_aug_ans+=1
            if str(df['answer'][i]).lower()!=str(df['result'][i]).lower():
                lenient_correct_samples_df = lenient_correct_samples_df.append(df.iloc[i])
        else:
            all_incorrect_df = all_incorrect_df.append(df.iloc[i])


    print("Correctly answered lenient: " ,correct_aug_ans)
    print("Correctly answered strict: ",strict_correct)
    print("Total questions: ",len(df))
    print("Lenient Accuracy: ",correct_aug_ans/len(df))
    print("Strict Accuracy: ",strict_correct/len(df))
    print("Float count: ",float_count)

    filename = csv_path.split('/')[-1].split('.')[0]

    # all_incorrect_df.to_csv(f'results/all_incorrect_{filename}.csv')
    # lenient_correct_samples_df.to_csv(f'results/lenient_correct_{filename}.csv')



# print("ViperGPT without captions:")
# evaluate_viper('results/results_24.csv')

# print("ViperGPT with captions:")
# evaluate_viper('results/results_29.csv')


evaluate_viper('results/results_24.csv', 'results/results_29.csv')