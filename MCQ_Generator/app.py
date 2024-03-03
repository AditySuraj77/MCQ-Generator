from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os
import json
from dotenv import load_dotenv
import PyPDF2
import pandas as pd
import fitz
import streamlit as st

load_dotenv()
api = os.getenv("GOOGLE_API_KEY")

Gemini = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=api)

with open('C:\\Users\\user.LAPTOP\\Desktop\\MCQ Generator\\Response.json', 'r') as f:
    response_json = json.load(f)

# QUIZ TEMPLATE
TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{RESPONSE_JSON}

"""
quiz_template = PromptTemplate(
    input_variables=['text', 'number', 'subject', 'tone', 'number', 'RESPONSE_JSON'],
    template=TEMPLATE
)
quiz_chain = LLMChain(llm=Gemini, prompt=quiz_template, output_key='quiz', verbose=True)

# SPELL EVALUTION TEMPLATE
TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}
Check from an expert English Writer of the above quiz:

"""
evalution_template = PromptTemplate(
    input_variables=['subject', 'quiz'],
    template=TEMPLATE2
)
evalution_chain = LLMChain(llm=Gemini, prompt=evalution_template, output_key='review', verbose=True)

#
final_chain = SequentialChain(chains=[quiz_chain, evalution_chain],
                              input_variables=['text', 'number', 'subject', 'tone', 'RESPONSE_JSON'],
                              output_variables=['review', 'quiz'],
                              verbose=True)






st.set_page_config('MCQ Generator')
st.header('MCQ Generator')

file = st.file_uploader("Upload PDF File",type=['pdf'] )
doc = fitz.open(file,filetype='txt')


number_of_Q = st.text_input('Number of Questions')
subject_name = st.text_input('Subject')
difficulty_level = st.selectbox('Difficulty Level', ('Beginner', 'Intermediate', 'Advance'))

btn = st.button('Generate')

if btn:
    if file and number_of_Q and subject_name and difficulty_level:
        response = final_chain(
            {
                "text": doc,
                "number": number_of_Q,
                "subject": subject_name,
                "tone": difficulty_level,
                "RESPONSE_JSON": json.dumps(response_json)
            })
        quiz = response.get('quiz')
        data_quiz = pd.DataFrame(eval(quiz))
        st.write(data_quiz)
    else:
        st.error('Enter Input !')
