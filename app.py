import os
import re
from flask import Flask, render_template, request, jsonify, send_file
import google.generativeai as genai
import pandas as pd
from io import StringIO
from google.cloud import storage,texttospeech
from google.cloud import speech
import logging
from flask import current_app


app = Flask(__name__)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else:
    app.logger.setLevel(logging.DEBUG)
    
# Set up Gemini Pro
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Set up Google Cloud Speech-to-Text client
speech_client = speech.SpeechClient()

def load_data_from_gcs(gcs_path):
    bucket_name = gcs_path.split('/')[2]
    blob_name = '/'.join(gcs_path.split('/')[3:])
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    csv_file = StringIO(data)
    df = pd.read_csv(csv_file)
    return df

# Load the dataset
gcs_path = 'gs://fake_patient_data/healthcare_dataset.csv'
df = load_data_from_gcs(gcs_path)

def get_available_conditions():
    conditions = df['Medical Condition'].unique()
    return sorted(conditions)

def text_to_speech(text, output_path):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

# def generate_concise_scenario_and_question(condition):
#     condition_df = df[df['Medical Condition'] == condition]

#     if condition_df.empty:
#         return "No patients found with this condition. Please try another."

#     patient = condition_df.sample(1).iloc[0]
#     name = patient['Name'].title()
#     prompt = f"""
#     Create a concise summary and question for a patient with {condition}.

#     Patient Details:
#     Name: {name}
#     Age: {patient['Age']}
#     Gender: {patient['Gender']}
#     Medical Condition: {patient['Medical Condition']}
#     Date of Admission: {patient['Date of Admission']}
#     """

#     # Add additional examples
#     examples = """
#     The output should be formatted as below examples, without any stars or extra symbols:
#     1. Tina Miller is a 78-year-old female with a medical history of Arthritis. They were admitted to the hospital on 2022-12-22.
#     Which of the following is a common long-term complication of Arthritis?
#     (A) Heart disease
#     (B) Retinal damage
#     (C) Kidney failure
#     (D) All of the above

#     2. Michael Mooney, a 68-year-old male, was admitted to the hospital on 2020-07-22 with a diagnosis of cancer.
#     What is the current medical condition of the patient?
#     (A) Heart disease
#     (B) Cancer
#     (C) Diabetes
#     (D) Asthma

#     3. Sarah Johnson is a 45-year-old female with a medical history of Hypertension. She was admitted to the hospital on 2023-01-10.
#     Which of the following is a common long-term complication of Hypertension?
#     (A) Stroke
#     (B) Hearing loss
#     (C) Chronic kidney disease
#     (D) Osteoporosis

#     4. David Lee, a 55-year-old male, was admitted to the hospital on 2021-05-15 with a diagnosis of Type 2 Diabetes.
#     Which of the following is a common complication associated with Type 2 Diabetes?
#     (A) Liver failure
#     (B) Diabetic retinopathy
#     (C) Chronic obstructive pulmonary disease
#     (D) Multiple sclerosis

#     5. Linda Martinez is a 62-year-old female, was admitted to the hospital on 2022-08-30 with a diagnosis of Chronic Obstructive Pulmonary Disease (COPD).
#     What is a common symptom of Chronic Obstructive Pulmonary Disease?
#     (A) Abdominal pain
#     (B) Shortness of breath
#     (C) Headaches
#     (D) Joint pain

#     6. John Smith, a 70-year-old male, was admitted to the hospital on 2019-11-05 with a diagnosis of Parkinson’s Disease.
#     Which of the following is a common symptom of Parkinson’s Disease?
#     (A) Tremors
#     (B) Increased appetite
#     (C) High blood pressure
#     (D) Skin rash
#     """

#     prompt += examples
#     response = model.generate_content(prompt)
#     return response.text
def generate_concise_scenario_and_question(condition):
    condition_df = df[df['Medical Condition'] == condition]
    if condition_df.empty:
        return "No patients found with this condition. Please try another."
    patient = condition_df.sample(1).iloc[0]
    prompt = f"""
    Create a concise 2-3 line summary of a patient with {condition} based on:
    Name: {patient['Name']}
    Age: {patient['Age']}
    Gender: {patient['Gender']}
    Medical Condition: {patient['Medical Condition']}
    Date of Admission: {patient['Date of Admission']}

    Then, generate a single, concise exam-style question about this patient's condition.
    The question should be multiple choice with 4 options (A, B, C, D).
    """
    response = model.generate_content(prompt)
    return response.text


def evaluate_answer(scenario_and_question, user_answer):
    prompt = f"""
    Given the following scenario and question:

    {scenario_and_question}

    User's Answer: {user_answer}

    Provide a brief evaluation of the user's answer, including:
    Whether it's correct or incorrect. A short explanation of the correct answer.
    Format the evaluation as follows:

    selected Answer: [Correctness]

    Explanation: [Explanation of the correct answer, starting on a new line]
    """
    response = model.generate_content(prompt)
    return response.text


def interpret_user_input(user_input, available_conditions):
    if re.search(r'quit|exit', user_input, re.IGNORECASE):
        return 'quit'
    for condition in available_conditions:
        if re.search(re.escape(condition), user_input, re.IGNORECASE):
            return condition
    return 'unknown'

@app.route('/')
def home():
    conditions = get_available_conditions()
    return render_template('index.html', conditions=conditions)

@app.route('/process_input', methods=['POST'])
def process_input():
    app.logger.info('Received request to /process_input')
    app.logger.debug('Request JSON: %s', request.json)

    user_input = request.json['user_input']
    current_condition = request.json.get('current_condition')
    scenario_and_question = request.json.get('scenario_and_question')

    app.logger.info('User input: %s', user_input)
    app.logger.info('Current condition: %s', current_condition)
    app.logger.info('Scenario and question: %s', scenario_and_question)

    available_conditions = get_available_conditions()

    try:
        if not current_condition:
            # User is selecting a condition
            interpreted_input = interpret_user_input(user_input, available_conditions)
            if interpreted_input == 'quit':
                return jsonify({'type': 'quit', 'message': 'Thank you for using the Medical Q&A Session. Goodbye!'})
            elif interpreted_input in available_conditions:
                scenario_and_question = generate_concise_scenario_and_question(interpreted_input)
                audio_path = os.path.join(UPLOAD_FOLDER, 'scenario.mp3')
                text_to_speech(scenario_and_question, audio_path)
                return jsonify({
                    'type': 'question',
                    'condition': interpreted_input,
                    'scenario_and_question': scenario_and_question,
                    'audio_path': f'/audio/scenario.mp3'

                })
            else:
                return jsonify({'error': 'Condition not recognized. Please try again.'})
        else:
            # User is answering the question
            evaluation = evaluate_answer(scenario_and_question, user_input)
            audio_path = os.path.join(UPLOAD_FOLDER, 'evaluation.mp3')
            text_to_speech(evaluation, audio_path)
            return jsonify({
                'type': 'evaluation',
                'evaluation': evaluation,
                'audio_path': f'/audio/evaluation.mp3'
            })
    except Exception as e:
        app.logger.error('Error in process_input: %s', str(e), exc_info=True)
        return jsonify({'error': str(e)}), 400
        
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_content = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)

    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return jsonify({'transcription': transcription})

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)