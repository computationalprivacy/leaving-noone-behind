import pandas as pd
import multiprocessing as mp
import os
from src.shadow_data import create_shadow_training_data_membership
from src.set_based_classifier import fit_set_based_classifier
from src.feature_extractors import fit_ohe, get_feature_extractors, apply_feature_extractor
from src.classifiers import drop_zero_cols, scale_features, select_features, fit_validate_classifiers
from src.output import save_output
from src.utils import blockPrint, enablePrint, ignore_depreciation

#For parallelization purposes, we need the following set of functions    
def run_pipeline_parallel(args):
    return run_pipeline(*args)

def run_pipeline(i, df_aux, target_record, datasets_test, labels_test, target_record_id, feat_selection,
                 models, cv, seed, continuous_cols, categorical_cols, meta_data, generator, n_original,
                 n_synthetic, n_pos_train, train_seeds, scenario, name_generator, args):
    """
    This function is call a part of the main() fonction, by each thread.
    """

    #save path
    save_dir = f'./results_experiments/synthetic_only/scenario_{scenario}/target_{target_record_id}/{name_generator}_{n_synthetic}/'
    
    #We create the directory if needed
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
      
    datasets_train, labels_train, _ = create_shadow_training_data_membership(df = df_aux, meta_data = meta_data,
                                    target_record = target_record, generator = generator, n_original = n_original,
                                    n_synth = n_synthetic, n_pos = n_pos_train, seeds = train_seeds)
    enablePrint()
    print('Finished providing training sets')
    blockPrint()
    
    # fit one hot encoding
    data_ohe = pd.concat(datasets_train,ignore_index = True)
    ohe, ohe_column_names = fit_ohe(data_ohe, categorical_cols, meta_data)
        
    ### Query based extraction
    enablePrint()
    print('Beginning queries')
    blockPrint()
    
    QUERY_FEATURE_EXTRACTORS = ['naive', ('query', range(1, len(ohe_column_names) + len(continuous_cols) + 1),
                                          10 ** 6, {'categorical':(1,), 'continuous': (3,)})]

    feature_extractors, do_ohe = get_feature_extractors(QUERY_FEATURE_EXTRACTORS)
    ignore_depreciation()
    X_train, y_train = apply_feature_extractor(datasets = datasets_train, target_record = target_record,
                                        labels = labels_train,
                                        ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names, continuous_cols = continuous_cols,
                                        feature_extractors = feature_extractors, do_ohe = do_ohe)
    X_test, y_test = apply_feature_extractor(datasets = datasets_test, target_record = target_record,
                                        labels = labels_test,
                                        ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names, continuous_cols = continuous_cols,
                                        feature_extractors = feature_extractors, do_ohe = do_ohe)
    X_train, X_test = drop_zero_cols(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)

    if feat_selection:
        valid_cols = select_features(X_train, y_train)
        X_train, X_test = X_train[valid_cols], X_test[valid_cols]

    trained_models, all_results = fit_validate_classifiers(X_train, y_train, X_test, y_test, models = models, cv = cv)
    
    #name of the saved file
    if i==-1 :
        QUERY_OUTPUT_PATH = save_dir + 'query.csv'
        save_output(QUERY_OUTPUT_PATH, vars(args), all_results)
    else :
        QUERY_OUTPUT_PATH = save_dir + f'query_{i}_seed_{seed}.csv'
        save_output(QUERY_OUTPUT_PATH, vars(args), all_results, have_auc = False)
        
    ### Classifier with TargetAttention
    enablePrint()
    print('Beginning targetattention')
    blockPrint()
    
    TARGET_ATTENTION_MODEL_PARAMS = {'num_in_features':len(ohe_column_names) + len(continuous_cols) + 2,
                                     'embedding_hidden_size':10, 'embedding_size':10,
                                    'attention_size': 10, 'n_records': 1000,
                                    'prediction_hidden_size':5, 'dropout_rate':0.15}
    
    trained_models, all_results = fit_set_based_classifier(datasets_train, datasets_test, 
                                                               labels_train, labels_test, 
                                                           TARGET_ATTENTION_MODEL_PARAMS,
                                                               target_record, 
                                                               ohe, categorical_cols, ohe_column_names,
                                                           continuous_cols=continuous_cols, meta_data=meta_data, df_aux=df_aux,
                                                   path_best_model = save_dir + f'best_model_{target_record_id}_targetattention_{mp.current_process().name}.pt', model_type = 'TargetAttention', 
                                                          top_X = 100)

    #name of the saved file
    if i==-1:
        TARGETATTENTION_OUTPUT_PATH = save_dir + 'targetattention.csv'
        save_output(TARGETATTENTION_OUTPUT_PATH, vars(args), all_results)
    else :
        TARGETATTENTION_OUTPUT_PATH = save_dir + f'targetattention_{i}_seed_{seed}.csv'
        save_output(TARGETATTENTION_OUTPUT_PATH, vars(args), all_results)
    
    print('success!')
    
    return 0