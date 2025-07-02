import os,traceback
import pandas as pd
import chainlit as cl
from visulations import generate_graphs
from analysis import ai_image_analysis,ai_text_analysis
from utlis import df_into_string
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg') # helps to create image in backend

model_name='gemini-1.5-flash-latest'

model_available=False


try:
    if api_key := os.environ.get('Gemini_API'):
        genai.configure(api_key=api_key)
        model=genai.GenerativeModel(model_name)
        model_available=True

except Exception as e:
    print(f'Gemini Init Failed {e}')



async def cleanup(files):
    for f in files:
        try:os.remove(f)
        except:pass



@cl.on_chat_start
async def start():
    await cl.Message(content='Upload a CSV file for AI Anaylsis').send()
    file=await cl.AskFileMessage(content='Upload CSV file',accept=['text/csv']).send()

    if not file:
        return await cl.Message(content='No file found').send()
    
    # processing
    processing_msg=cl.Message(content='Processing file....')
    await processing_msg.send()

    # analysis
    try:
        df = pd.read_csv(file[0].path)

        if df.empty:
            processing_msg.content = 'Empty dataset'
            await processing_msg.update()
            return
        
        cl.user_session.set('df',df)

        info=df_into_string(df)
        await cl.Message(content=info).send()

        # text analysis
        if model_available:
            plan= await ai_text_analysis('plan',info)
            await cl.Message(content=f'AI plan {plan}').send()

        # generating graphs
        visuals, saved_files = generate_graphs(df)
        for title,path in visuals:
            await cl.Message(content=f'{title}',elements=[cl.Image(name=title,path=path)]).send()


        # visual 
        if model_available:
            insights=await ai_image_analysis(visuals)
            for title,insight in enumerate(insights):
                await cl.Message(content=f'{title} Insight\n{insight}').send()

            final=await ai_text_analysis('final',info)
            await cl.Message(content=f'Final AI Report\n{final}').send()
 
        processing_msg.content = 'Analysis Complete'
        await processing_msg.update()
        await cleanup(saved_files)


    except Exception as e:
        traceback.print_exc()
        processing_msg.content = f'Error {e}'
        await processing_msg.update()