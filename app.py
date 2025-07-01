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



# def save_fig(fig):
#     f=tempfile.NamedTemporaryFile(delete=False,suffix='.png')
#     fig.savefig(f.name,bbox_inches='tight',dpi=100)
#     plt.close(fig)
#     return f.name

# def df_into_string(df,max_rows=5): # for text anaylsis
#     buf=io.StringIO()
#     df.info(buf)
#     schema=buf.getvalue()
#     head=df.head(max_rows).to_markdown(index=False)

#     missing=df.isnull().sum()
#     missing=missing[missing>0]
#     missing_info='No missing values.'if missing.empty else str(missing)
#     return f"""
#     Schema : \n {schema} \n Preview : \n{head}\n Missing : \n{missing_info}
#     """


# async def ai_text_analysis(prompt_type,df_context):
#     if not model_available: return f"Model not available"

#     prompt={
#         'plan':f'You are data anaylsis expert. Suggest concise data analysis plan:\n{df_context}',
#         'final':f'Summarize the insights from the following it should be on point dataset:\n{df_context}'
#     }

#     try:
#         model=genai.GenerativeModel(model_name) # type: ignore
#         res= await model.generate_content_async(prompt.get(prompt_type),generation_config=genai.types.GenerationConfig(max_output_tokens=500,temperature=0.4))
#         return res.text if res.parts else '❌ Gemini Response Blocked'
#     except Exception as e:
#         return f'Gemini Error {e}'
    

# image analysis
# async def ai_image_analysis(img_path):
#     if not model_available: return f"Model not available"

#     model=genai.GenerativeModel(model_name)
#     result=[]

#     for title,path in img_path:
#         try:
#             img=Image.open(path)
#             res= await model.generate_content_async([f'Explain this {title}',img],
#                                                     generation_config=genai.types.GenerationConfig(max_output_tokens=200,temperature=0.4))
#             result.append((title,res.text if res.parts else 'Blocked or empty response'))

#         except Exception as e:
#             result.append((title,f'❌Error {e}'))  

#     return result      


            
# generate graphs

# def generate_graphs(df:pd.DataFrame):
#     visualizations = []
#     saved_files = []

#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     cat_cols = [col for col in df.select_dtypes(include='object') if 1 < df[col].nunique() < 30]

#     try:
#         # 1. Histograms for numeric columns
#         if numeric_cols:
#             fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(8, 4*len(numeric_cols)))
#             if len(numeric_cols) == 1:
#                 axes = [axes]
#             for ax, col in zip(axes, numeric_cols):
#                 sns.histplot(df[col], kde=True, ax=ax) # type: ignore
#                 ax.set_title(f'Histogram of {col}')
#             plt.tight_layout()
#             hist_path = save_fig(fig)
#             saved_files.append(hist_path)
#             visualizations.append(('Histograms of Numeric Columns', hist_path))

#         # 2. Boxplots for numeric columns
#         if numeric_cols:
#             fig, axes = plt.subplots(nrows=1, ncols=len(numeric_cols), figsize=(4*len(numeric_cols), 6))
#             if len(numeric_cols) == 1:
#                 axes = [axes]
#             for ax, col in zip(axes, numeric_cols):
#                 sns.boxplot(y=df[col], ax=ax)
#                 ax.set_title(f'Boxplot of {col}')
#             plt.tight_layout()
#             box_path = save_fig(fig)
#             saved_files.append(box_path)
#             visualizations.append(('Boxplots of Numeric Columns', box_path))

#         # 3. Countplots for categorical columns
#         if cat_cols:
#             fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(8, 4*len(cat_cols)))
#             if len(cat_cols) == 1:
#                 axes = [axes]
#             for ax, col in zip(axes, cat_cols):
#                 sns.countplot(x=df[col], ax=ax)
#                 ax.set_title(f'Countplot of {col}')
#                 ax.tick_params(axis='x', rotation=45)
#             plt.tight_layout()
#             count_path = save_fig(fig)
#             saved_files.append(count_path)
#             visualizations.append(('Countplots of Categorical Columns', count_path))

#         # 4. Pairplot (if reasonable number of numeric columns)
#         if len(numeric_cols) > 1 and len(numeric_cols) <= 5:
#             pairplot = sns.pairplot(df[numeric_cols])
#             pair_path = save_fig(pairplot.fig)
#             saved_files.append(pair_path)
#             visualizations.append(('Pairplot of Numeric Columns', pair_path))

#         # 5. Violin plots (numeric vs categorical if available)
#         if numeric_cols and cat_cols:
#             # Use first numeric and first categorical column for example
#             num_col = numeric_cols[0]
#             cat_col = cat_cols[0]
#             fig = plt.figure(figsize=(10, 6))
#             sns.violinplot(x=df[cat_col], y=df[num_col])
#             plt.title(f'Violin Plot of {num_col} by {cat_col}')
#             plt.xticks(rotation=45)
#             violin_path = save_fig(fig)
#             saved_files.append(violin_path)
#             visualizations.append((f'Violin Plot of {num_col} by {cat_col}', violin_path))

#         # 6. Correlation heatmap
#         if len(numeric_cols) > 1:
#             fig = plt.figure(figsize=(10, 8))
#             corr = df[numeric_cols].corr()
#             sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
#             plt.title('Correlation Heatmap')
#             heatmap_path = save_fig(fig)
#             saved_files.append(heatmap_path)
#             visualizations.append(('Correlation Heatmap', heatmap_path))

#     except Exception as e:
#         print(f"Error generating graphs: {e}")
#         traceback.print_exc()

#     return visualizations, saved_files


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
            await cl.Message(content=f'### AI plan {plan}').send()

        # generating graphs
        visuals, saved_files = generate_graphs(df)
        for title,path in visuals:
            await cl.Message(content=f'### {title}',elements=[cl.Image(name=title,path=path)]).send()


        # visual 
        if model_available:
            insights=await ai_image_analysis(visuals)
            for title,insight in enumerate(insights):
                await cl.Message(content=f'###{title} Insight\n{insight}').send()

            final=await ai_text_analysis('final',info)
            await cl.Message(content=f'### Final AI Report\n{final}').send()
 
        processing_msg.content = 'Analysis Complete'
        await processing_msg.update()
        await cleanup(saved_files)


    except Exception as e:
        traceback.print_exc()
        processing_msg.content = f'Error {e}'
        await processing_msg.update()