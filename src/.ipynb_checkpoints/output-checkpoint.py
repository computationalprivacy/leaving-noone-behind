import pandas as pd

ARGS_TO_SAVE =  ['target_record_id', 'name_generator', 'epsilon', 'n_aux', 'n_test', 'n_original', 'n_synthetic',
                 'n_pos_train', 'n_pos_test', 'cv', 'seed', 'cols_to_select']

def prep_for_output(args: dict, all_results: list, utility_results: dict=None, have_auc=True) -> pd.DataFrame:
    n_rows = len(all_results)
    all_data = []

    for i in range(n_rows):
        col_to_add = [args[value] for value in ARGS_TO_SAVE]
        col_to_add += [utility_results]
        col_to_add += [args['models'][i]]
        col_to_add += list(all_results[i])
        all_data.append(col_to_add)

    if have_auc:
        col_names = ARGS_TO_SAVE + ['utility_results', 'model', 'train_acc', 'train_auc', 'test_acc', 'test_auc']
    else: 
        col_names = ARGS_TO_SAVE + ['utility_results', 'model', 'train_acc', 'test_acc']
        
    output_df = pd.DataFrame(all_data, columns = col_names)

    return output_df

def save_output(output_path: str, args: dict, all_results: list, utility_results: dict=None, have_auc=True) -> None:
    output_df = prep_for_output(args, all_results, utility_results, have_auc)
    output_df.to_csv(output_path)
    return None

