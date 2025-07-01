import google.generativeai as genai
from PIL import Image

model_available=False
model_name='gemini-1.5-flash-latest'

async def ai_text_analysis(prompt_type,df_context):
    if not model_available: return f"Model not available"

    prompt={
        'plan':f'You are data anaylsis expert. Suggest concise data analysis plan:\n{df_context}',
        'final':f'Summarize the insights from the following it should be on point dataset:\n{df_context}'
    }

    try:
        model=genai.GenerativeModel(model_name) # type: ignore
        res= await model.generate_content_async(prompt.get(prompt_type),generation_config=genai.types.GenerationConfig(max_output_tokens=500,temperature=0.4)) # type: ignore
        return res.text if res.parts else '❌ Gemini Response Blocked'
    except Exception as e:
        return f'Gemini Error {e}'
    


async def ai_image_analysis(img_path):
    if not model_available: return f"Model not available"

    model=genai.GenerativeModel(model_name) # type: ignore
    result=[]

    for title,path in img_path:
        try:
            img=Image.open(path)
            res= await model.generate_content_async([f'Explain this {title}',img],
                                                    generation_config=genai.types.GenerationConfig(max_output_tokens=200,temperature=0.4)) # type: ignore
            result.append((title,res.text if res.parts else 'Blocked or empty response'))

        except Exception as e:
            result.append((title,f'❌Error {e}'))  

    return result  