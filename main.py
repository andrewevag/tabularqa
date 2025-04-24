import boto3
import core.loops as loops
import core.model_calls as model_calls
import core.prompt_generators as prompt_generators
import core.postprocessors as postprocessors
import core.executors as executors
import core.utils as utils
import core.eval as eval
import core.pipelines as pipelines
import core.cost as cost
import os
from dotenv import load_dotenv

def instantiate_pipeline(
    client,
    lite,
    annotations_filename,
    exemplar_indices,
    debug
):
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
    load_dotenv()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    
    semeval_test_qa = utils.load_test_data_with_answers()
    
    
    lite=False
    debug=False
    save_seq=True
    annotations_filename = "annotation/annotations_cot.json"
    exemplar_indices = [(17, 140), (0, 132), (28, 286), (31, 303), (4, 246), (24, 175), (20, 176), (8, 141), (14, 12)]
        
    n = len(exemplar_indices)
    result_path = f"results/Claude_3_5_Sonnet_Answerer_ERROR_FIXING{'_lite' if lite else ''}"
    
    complete_pipe, model = instantiate_pipeline(client, lite=lite, annotations_filename=annotations_filename, exemplar_indices=exemplar_indices, debug=debug)

    cost_predictor = cost.CostAccumulator(model.pit, model.pot)
    
    evaluator = eval.Evaluator(qa=semeval_test_qa)
    
    msg = f"""
    Running only on the Test set. Claude 3.5 Sonnet Answerer. Error fixing. {'lite' if lite else ''}
    	
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
