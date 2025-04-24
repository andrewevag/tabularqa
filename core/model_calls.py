from botocore.exceptions import ClientError
import json
import ollama
from ollama import ChatResponse
from abc import ABC, abstractmethod
from typing import Any, Union, Optional

class Answerer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prompt: Any) -> Any:
        pass

class SampleAnswerer(Answerer):
    def __init__(self):
        pass

    def __call__(self, prompt: str) -> str:
        return "True"

class BedrockModelAnswerer(Answerer):
    def __init__(self, model_id, client):
        super().__init__()
        self.client = client
        self.model_id = model_id
        
    def __call__(self, prompt : Union[str, dict[str, Any]], gen_parameters : dict[str, Any]) -> tuple[str, int, int]:
        
        gen_parameters["prompt"] = prompt
        # Convert the native request to JSON.
        request = json.dumps(gen_parameters)
        try:
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId=self.model_id, body=request)

            input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
            output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
    
            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            response_text = model_response["generation"]
            return response_text, input_tokens, output_tokens
        
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return "", 0, 0
    
    def __str__(self) -> str:
        return f"BedrockModelAnswerer(Answerer)\n\tmodel_id: {self.model_id}\n\tclient: {self.client}"

class LLama3_1_405BInstructAnswerer(BedrockModelAnswerer):
    model_id = "meta.llama3-1-405b-instruct-v1:0"
    pit = 0.0024
    pot = 0.0024
    def __init__(self, client, max_gen_len: int=30, temperature: float=0.0, top_p: float=0.9):
        super().__init__(
            "meta.llama3-1-405b-instruct-v1:0",
            client
        )

        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, prompt: str, custom_gen_parameters=None):
        if custom_gen_parameters:
            gen_parameters = custom_gen_parameters
            gen_parameters["prompt"] = prompt
        else:
            gen_parameters = {
                "prompt": prompt,
                "max_gen_len": self.max_gen_len,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        
        response, input_tokens, output_tokens = super().__call__(prompt, gen_parameters)

        return response, input_tokens, output_tokens

class LLama3_3_70BInstructAnswerer(BedrockModelAnswerer):
    model_id = "us.meta.llama3-3-70b-instruct-v1:0"
    pit = 0.00072
    pot = 0.00072
    def __init__(self, client, max_gen_len: int=30, temperature: float=0.0, top_p: float=0.9):
        super().__init__(
            "us.meta.llama3-3-70b-instruct-v1:0",
            client
        )

        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, prompt: str, custom_gen_parameters : dict[str, Any]=None):
        if custom_gen_parameters:
            gen_parameters = custom_gen_parameters
            gen_parameters["prompt"] = prompt
        else:
            gen_parameters = {
                "prompt": prompt,
                "max_gen_len": self.max_gen_len,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        
        response, input_tokens, output_tokens = super().__call__(prompt, gen_parameters)

        return response, input_tokens, output_tokens

class Claude_3_5_Sonnet_ModelAnswerer(BedrockModelAnswerer):
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    pit = 0.003
    pot = 0.015

    def __init__(self, client, max_gen_len: int=30, temperature: float=0.0, top_p: float=0.9):
        super().__init__(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            client
        )
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, prompt : dict[str, Any], custom_gen_parameters=None) -> tuple[str, int, int]:
        prompt["max_tokens"] = self.max_gen_len
        prompt["anthropic_version"] = "bedrock-2023-05-31"
         
        if custom_gen_parameters:
            for key, value in custom_gen_parameters.items():
                prompt[key] = value
        else:
            prompt['temperature'] = self.temperature
            prompt['top_p'] = self.top_p
            pass
        try: 
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(prompt)
            )

            output_binary = response["body"].read()
            output_json = json.loads(output_binary)
            input_tokens = output_json['usage']['input_tokens']
            output_tokens = output_json['usage']['output_tokens']
            output = output_json["content"][0]["text"]

            return output, input_tokens, output_tokens
        except Exception as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return "", 0, 0

    def __str__(self) -> str:
        return f"Claude_3_5_Sonnet_ModelAnswerer(BedrockModelAnswerer)\n\tmodel_id: {self.model_id}\n\tclient: {self.client}\n\tmax_gen_len: {self.max_gen_len}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}"

class Llama3_8BAnswererOllama(Answerer):
    model_id = "llama3.1:8b"
    pit = 0 
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt: str):
        response = ollama.generate(
            model='llama3.1:8b', 
            prompt=prompt,  
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 32
                },
            raw=True
        )
        return response['response'], response['prompt_eval_count'], 0

    def __str__(self) -> str:
        return f"Llama3_8BAnswererOllama(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

class Llama3_8BAnswererOllamaChat(Answerer):
    model_id = "llama3.1:8b"
    pit = 0 
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt):
        response: ChatResponse = ollama.chat(
            model='llama3.1:8b',
            messages=prompt,
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 30
            })
        return response.message.content,response.prompt_eval_count, 0

    def __str__(self):
        return f"Llama3_8BAnswererOllamaChat(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

class Llama3_370BAnswererOllama(Answerer):
    model_id = "llama3.3"
    pit = 0
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt: str):
        response = ollama.generate(
            model='llama3.3', 
            prompt=prompt,  
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 32
                },
            raw=True
        )
        return response['response'], 0, 0

    def __str__(self):
        return f"Llama3_370BAnswererOllama(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

class Llama3_370BAnswererOllamaChat(Answerer):
    model_id = "llama3.3"
    pit = 0 
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt):
        print(prompt, len(prompt))
        response: ChatResponse = ollama.chat(
            model='llama3.3',
            messages=prompt,
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 16
            })
        return response.message.content, 0, 0

    def __str__(self):
        return f"Llama3_370BAnswererOllamaChat(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

class Qwen2_5_Coder7BAnswererOllama(Answerer):
    model_id = "qwen2.5-coder"
    pit = 0 
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt: str):
        response = ollama.generate(
            model='qwen2.5-coder', 
            prompt=prompt,  
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 64
                },
            raw=True
        )
        return response['response'], response['prompt_eval_count'], 0

    def __str__(self):
        return f"Qwen2_5_Coder7BAnswererOllama(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

class Qwen2_5_Coder7BAnswererOllamaChat(Answerer):
    model_id = "qwen2.5-coder"
    pit = 0 
    pot = 0
    def __init__(self, temperature=0.0, top_p=0.9, max_gen_len=300):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        

    def __call__(self, prompt: str):
        response: ChatResponse = ollama.chat(
            model='qwen2.5-coder',
            messages=prompt,
            options={
                "temperature": self.temperature, 
                "top_p": self.top_p, 
                "num_predict" : self.max_gen_len,
                "num_ctx" : 1024 * 32
            })

        return response['message']['content'], response['prompt_eval_count'], 0

    def __str__(self):
        return f"Qwen2_5_Coder7BAnswererOllamaChat(Answerer)\n\tmodel_id: {self.model_id}\n\tpit: {self.pit}\n\tpot: {self.pot}\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}\n\tmax_gen_len: {self.max_gen_len}"

