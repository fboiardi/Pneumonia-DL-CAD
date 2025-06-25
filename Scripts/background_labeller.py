import pandas as pd
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import time, os

set_seed(42)

def analyse_mimic_report(report):
    chat = [
          {"role": "user", 
           "content": f'''
           ### ROLE ###
           YOU ARE A HIGHLY ACCLAIMED RADIOLOGIST BASED IN THE UNITED KINGDOM WITH DECADES OF EXPERIENCE IN ANALYSING ELECTRONIC HEALTH RECORDS (EHRs) AND RADIOLOGICAL REPORTS. YOUR TASK IS TO ANALYSE AN EHR REPORT AND PERFORM A SERIES OF INSTRUCTIONS BASED ON THE DIAGNOSIS OF PNEUMONIA.
           
           ### TASK OVERVIEW ###
            YOUR TASK IS TO ANALYSE THE ENTIRE RADIOLOGICAL TEXT PROVIDED AND PERFORM THESE INSTRUCTIONS:
            1. DETERMINE WHETHER THE PATIENT HAS ACTIVE PNEUMONIA AT THE TIME OF THE STUDY:
               - CLASSIFY THE REPORT AS:
                 - "1" FOR PNEUMONIA-POSITIVE (PATIENT CURRENTLY HAS PNEUMONIA, OR NO OTHER CONDITIONS ARE POSSIBLE).
                 - "0" FOR PNEUMONIA-NEGATIVE (CLEARLY STATES NO PNEUMONIA OR NO INFECTION).
                 - "-1" FOR PNEUMONIA-UNCERTAIN (PNEUMONIA MENTIONED WITH UNCERTAINTY, AMBIGUITY, OR POSSIBILITY).
               - IMPORTANTLY, EVEN IF THE REPORT EXPLICITLY STATES PNEUMONIA, IT DOES NOT AUTOMATICALLY MEAN THE CLASSIFICATION IS PNEUMONIA-POSITIVE; YOU MUST CONSIDER THE CONTEXTUAL INFORMATION TO UNDERSTAND WHETHER THE MENTION IS POSITIVE, NEGATIVE, OR UNCERTAIN.
            
            2. EXTRACT THE PNEUMONIA LOCATION STRING(S):
                - IF CLASSIFIED AS "1" (PNEUMONIA-POSITIVE) OR "-1" (PNEUMONIA-UNCERTAIN), EXTRACT THE POSITIONAL INFORMATION OF THE ACTIVE PNEUMONIA AS A LIST IF PROVIDED (E.G., ["UPPER LOBES BILATERALLY", "RIGHT MIDDLE LOBE"]) EXACTLY AS WRITTEN IN THE TEXT.
                 - OUTPUT "None" IF NO LOCATION IS MENTIONED.
                - IF CLASSIFIED AS "0", OUTPUT "None" FOR THE LOCATION, REGARDLESS OF WHETHER A LOCATION IS MENTIONED.
            
            ### PNEUMONIA-POSITIVE ("1") EXAMPLES ###
            - Patient shows signs of pneumonia.
            - There is evidence of pneumonia.
            - Pneumonia has worsened since the prior radiograph.
            - Pneumonia has almost completely resolved, but is still present.
            - Pneumonia is healing, but still present.
            - Pre-existing pneumonia has improved.
            - Improved, but not yet resolved, pneumonia.
            - There has been partial clearing of the pneumonia.

            ### PNEUMONIA-NEGATIVE ("0") EXAMPLES ###
            - No pneumonia.
            - No definite pneumonia.
            - No evidence of pneumonia.
            - No evidence to suggest infection.
            - Interval resolution of patient's pneumonia.
            - Pneumonia has resolved.
            - Pneumonia is mentioned as having resolved.
            - Resolution of pneumonia.
            - There has been complete clearing of the pneumonia.

            ### PNEUMONIA-UNCERTAIN ("-1") EXAMPLES ###
            - <symptom> may reflect <other condition>, less likely pneumonia.
            - <symptom> may reflect <other condition>, though infection is not excluded in the correct clinical setting.
            - <symptom> more typical of <other condition> than pneumonia.
            - <symptom> could reflect <other condition> or secondary process such as pneumonia.
            - <other condition> more likely than pneumonia.
            - Differential includes pneumonia.
            - Could be <other condition>, but should be followed closely to exclude pneumonia.
            - Pneumonia is difficult to exclude.
            - Pneumonia could also be considered.

            ### OUTPUT FORMAT ###
            GENERATE THE OUTPUT ACCORDING TO THE FOLLOWING XML FORMAT: OPENING TAG, RESULT, CLOSING TAG.

            INPUT EXAMPLE 1:
            "Patient shows signs of pneumonia localized to the right lower lobe."
            
            EXPECTED OUTPUT EXAMPLE 1:
            <EVALUATION>1</EVALUATION>
            <LOCATION>["right lower lobe"]</LOCATION>

            INPUT EXAMPLE 2:
            "The known pre-existing left upper lobe pneumonia has substantially improved."

            EXPECTED OUTPUT EXAMPLE 2:
            <EVALUATION>1</EVALUATION>
            <LOCATION>["left upper lobe"]</LOCATION>

            INPUT EXAMPLE 3:
            "No vascular congestion or acute focal pneumonia."

            EXPECTED OUTPUT EXAMPLE 3:
            <EVALUATION>0</EVALUATION>
            <LOCATION>None</LOCATION>

            INPUT EXAMPLE 4:
            "Resolution of left lower lobe pneumonia."

            EXPECTED OUTPUT EXAMPLE 4:
            <EVALUATION>0</EVALUATION>
            <LOCATION>None</LOCATION>

            INPUT EXAMPLE 5:
            "Subtle increase in lower lung opacities may reflect atelectasis, less likely early pneumonia."

            EXPECTED OUTPUT EXAMPLE 5:
            <EVALUATION>-1</EVALUATION>
            <LOCATION>["lower lung"]</LOCATION>

            INPUT EXAMPLE 6:
            "Retrocardiac patchy opacity may reflect atelectasis though infection is not excluded in the correct clinical setting."

            EXPECTED OUTPUT EXAMPLE 6:
            <EVALUATION>-1</EVALUATION>
            <LOCATION>["Retrocardiac"]</LOCATION>

            ### WHAT NOT TO DO ###
            - DO NOT ASSUME OR INTERPRET ANY INFORMATION NOT EXPLICITLY PRESENT IN THE TEXT.
            - DO NOT MODIFY THE STRING OF THE LOCATION; IT HAS TO BE THE SAME CHARACTERS.
            - DO NOT INCLUDE FURTHER INFORMATION OTHER THAN THE POSITIONAL INFORMATION FOR THE LOCATION.
            - DO NOT OUTPUT LOCATIONS THAT ARE NOT EXPLICITLY MENTIONED IN THE TEXT IN REFERENCE TO PNEUMONIA.
            - DO NOT INCLUDE INFORMATION ABOUT THE PNEUMONIA OUTSIDE OF ITS POSITIONAL INFORMATION.
            - DO NOT PRODUCE OUTPUT WITHOUT STRICTLY FOLLOWING THE SPECIFIED XML FORMAT: OPENING TAG, RESULT, CLOSING TAG.
            
            ### FINAL NOTE ###
            ENSURE THE OUTPUT FORMAT IS CONSISTENT AND EXACTLY MATCHES THE EXAMPLES ABOVE. DO NOT OUTPUT ANY TEXT OTHER THAN THE PROVIDED FORMATED OUTPUT.

            ### INPUT TEXT ###
            Here is the radiological text to classify:
            {report}'''
          }
    ]
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    
    data = tokenizer.decode(outputs[0])
    response = data.split('</think>')[-1].replace('<｜end▁of▁sentence｜>', '')
    
    return response

def extract_pneu_info(mimic_row):
    print(f'Report #{mimic_row.name + 1}')
    if mimic_row['Pneumonia_Relevant'] == 1:
        return analyse_mimic_report(mimic_row['Report'])
    return 'N/A'

if __name__ == '__main__':
    login(token="XXXXX") # replace with huggingface token
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id,token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU device - for parallel use if available
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        temperature=.6,
        top_p=.6,
        torch_dtype=dtype,
        token=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    mimic_sample = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-pneumonia-full-wreports_0.csv')

    start = time.time()
    mimic_sample['LLM_Pneumonia'] = mimic_sample.apply(extract_pneu_info, axis=1)
    end = time.time()
    print(f'Time elapsed: {(end-start)/3600:.2f}h')

    mimic_sample.to_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-pneumonia-full-wreports_0_llm.csv', index=False)
