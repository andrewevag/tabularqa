import boto3
import core.loops as loops
import core.model_calls as model_calls
import core.prompt_generators as prompt_generators
import core.postprocessors as postprocessors
import core.executors as executors
import core.utils as utils
import core.eval as evalulator
import core.pipelines as pipelines
import core.cost as cost
import os
from dotenv import load_dotenv
import yaml
import argparse


def instantiate_pipeline_from_yaml(config_path, exemplar_indices, annotations_filename, client=None, debug=False, lite=False):

    loader = utils.generic_load_table if not lite else utils.generic_load_sample
    indices, exemplars = utils.annotation_reader(annotations_filename)

    exemplar_indices = list(map(lambda x: x[0], exemplar_indices))
    to_use_exemplars = [exemplars[i] for i in exemplar_indices]


    # Custom YAML constructors for placeholders
    def client_placeholder_constructor(loader, node):
        return client

    def exemplars_placeholder_constructor(loader, node):
        return to_use_exemplars

    def loader_placeholder_constructor(loader, node):
        return loader
    
    def debug_placeholder_constructor(loader, node):
        return debug
    
    def lite_placeholder_constructor(loader, node):
        return lite

    # Create a custom YAML loader
    class CustomLoader(yaml.Loader):
        pass

    # Register the custom constructors with the custom loader
    CustomLoader.add_constructor("!CLIENT_PLACEHOLDER", client_placeholder_constructor)
    CustomLoader.add_constructor("!EXEMPLARS_PLACEHOLDER", exemplars_placeholder_constructor)
    CustomLoader.add_constructor("!LOADER_PLACEHOLDER", loader_placeholder_constructor)
    CustomLoader.add_constructor("!DEBUG_PLACEHOLDER", debug_placeholder_constructor)
    CustomLoader.add_constructor("!LITE_PLACEHOLDER", lite_placeholder_constructor)

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=CustomLoader)

    complete_pipe_config = config['complete_pipe']
    complete_pipe = _instantiate_from_config(complete_pipe_config, loader)


    model = complete_pipe.main_pipeline.answerer
    return complete_pipe, model

def _instantiate_from_config(config, loader):
    """Does the recursive instantiation of the pipeline from the config file."""
    if not isinstance(config, dict) or 'class_name' not in config:
        return config  # Return as-is if it's not a class configuration


    class_name = config['class_name']

    cls = eval(class_name)

    # Step of the DFS.
    arguments = config.get('arguments', {})
    resolved_arguments = {
        key: _instantiate_from_config(value, loader)
        for key, value in arguments.items()
    }
    # Add the loader if required
    if 'loader' in resolved_arguments:
        resolved_arguments['loader'] = loader

    # Instantiate the class with the resolved arguments
    return cls(**resolved_arguments)


def instantiate_pipeline(
    client,
    lite,
    annotations_filename,
    exemplar_indices,
    debug
):
    """This is not used. The pipelines are instantiated from the configs."""
    loader = utils.generic_load_table if not lite else utils.generic_load_sample
    indices, exemplars = utils.annotation_reader(annotations_filename)

    exemplar_indices = list(map(lambda x: x[0], exemplar_indices))
    to_use_exemplars = [exemplars[i] for i in exemplar_indices]
    
    prompt_generator = prompt_generators.ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2(loader)
    exemplar_builder = prompt_generators.ExemplarBuilder(loader, prompt_generator)
    
    full_prompt_generator = prompt_generators.FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2(
    	loader,
    	prompt_generator,
    	exemplar_builder,
    	shots=to_use_exemplars
    )
    
    claudewrappedgenerator = prompt_generators.ClaudeMessageEmbeddingPromptGenerator(loader, full_prompt_generator)
    
    prompt_generator = claudewrappedgenerator
    
    model = model_calls.Claude_3_5_Sonnet_ModelAnswerer(client, temperature=0.0, top_p=0.9, max_gen_len=300)
    postprocessor = postprocessors.TillReturnLinePostProcessorMultipleIndents(loader, prefix=4, first_prefix='    # The columns used to answer the question: ')
    executor = executors.SaferMultiLineStatementExecutor(loader)
    
    pipe = pipelines.Pipeline(prompt_generator, model, postprocessor, executor, debug=debug)
    
    error_zero_prompt_generator = prompt_generators.ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsUniquesStrExamples(loader)
    
    error_prompt_generator = prompt_generators.ErrorFixingGeneratorClaudeVol2(loader, error_zero_prompt_generator, num_rows=10)
    
    error_model = model_calls.Claude_3_5_Sonnet_ModelAnswerer(client, temperature=0.0, top_p=0.9, max_gen_len=1000)
    
    error_postprocessor = postprocessors.TillReturnLinePostProcessorMultipleIndents(loader, prefix=4, first_prefix='    # The columns used to answer the question: ')
    
    error_executor = executors.SaferMultiLineStatementExecutor(loader)
    
    error_fix_pipeline = pipelines.ErrorFixPipeline(error_prompt_generator, error_model, error_postprocessor, error_executor)
    
    complete_pipe = pipelines.ErrorFixingAndTimeoutPipeline(pipe, error_fix_pipeline, 600)

    return complete_pipe, model

    

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run the Pipeline on DataBench Test Set.")
    parser.add_argument(
        "--pipeline-config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the complete pipeline."
    )

    parser.add_argument(
        "--lite",
        action="store_true",
        help="Run the pipeline in lite mode (default: False)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the pipeline in debug mode (default: False)."
    )

    args = parser.parse_args()



    load_dotenv()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    if aws_access_key_id and aws_secret_access_key:
        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name="us-west-2"
        )
    else: client = None

    semeval_test_qa = utils.load_test_data_with_answers()
    
    lite=args.lite
    debug=args.debug
    save_seq=True
    annotations_filename = "annotation/annotations_cot.json"
    exemplar_indices = [(17, 140), (0, 132), (28, 286), (31, 303), (4, 246), (24, 175), (20, 176), (8, 141), (14, 12)]
        
    n = len(exemplar_indices)
    result_path = f"results/{args.pipeline_config.split('/')[-1][:-5]}{'_lite' if lite else ''}"
    

    complete_pipe, model = instantiate_pipeline_from_yaml(
        config_path=args.pipeline_config,
        exemplar_indices=exemplar_indices,
        annotations_filename=annotations_filename,
        client=client,
        debug=debug,
        lite=lite
    )

    cost_predictor = cost.CostAccumulator(model.pit, model.pot)
    
    evaluator = evalulator.Evaluator(qa=semeval_test_qa)
    
    msg = f"""
    Running only on the Test set. {args.pipeline_config} {'lite' if lite else ''}
    	
    Uses:
    {complete_pipe}
    {cost_predictor}
    {evaluator}
    saver : SerialSaver
    
    Uses {n} exemplars from the Chain of Thought dataset.
    The exemplars are (index in {annotations_filename}, index in train[:400])
    {exemplar_indices}
    
    See the relevant commit at the appropriate datetime for the exact code used.
    """
    print(msg)
    saver = loops.SerialSaver(result_path, msg, lite=lite)
    
    looper = loops.SerialLooper(
    	semeval_test_qa,
    	complete_pipe,
    	cost_predictor,
    	evaluator,
    	saver,
        debug=debug,
    	lite=lite,
    	save_seq=save_seq
    )
    res = looper.loop()
    print(res)

if __name__ == "__main__":
    main()
