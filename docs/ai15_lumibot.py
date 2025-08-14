import textwrap
import time
import re
from datetime import datetime
import requests
from io import BytesIO
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
import os
import openai

out_path = os._fspath("./Open-AI-Assistants/")
api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key

client = openai.OpenAI(api_key=api_key)

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
}

client2 = {
    "openai": openai.OpenAI(api_key=api_keys["openai"])
}

def openai_version(user_input, system_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    chat_completion = client2["openai"].chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        temperature=0.6,
    )
    response_content = chat_completion.choices[0].message.content
    return response_content

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(filepath, content):
    with open(os.path.join(os.getcwd(), filepath), 'w', encoding='utf-8') as f:
        f.write(content)

def chat_gpt3(userinput, chatbot, temperature=0.7, frequency_penalty=0, presence_penalty=0, api_key='your-api-key'):
    messages = [
        {"role": "system", "content": chatbot},
        {"role": "user", "content": userinput}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        max_tokens=150,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages
    )
    text = response.choices[0].message.content
    return text

def save_assistant_id(assistant_id, filename):
    filepath = f'ids/{filename}'
    with open(filepath, 'w') as file:
        file.write(assistant_id)

def check_existing_assistant_id(filename):
    filepath = f'ids/{filename}'
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            assistant_id = file.read().strip()
            if assistant_id:
                return assistant_id
    return None

def create_and_run_assistant(name, instructions, model, content, filename):
    assistant_id = check_existing_assistant_id(filename)
    if assistant_id:
        print(f"Using existing assistant with ID: {assistant_id}")
    else:
        try:
            assistant = client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=[{"type": "code_interpreter"}],
                model=model
            )
            print(f'{name} created....')
            assistant_id = assistant.id
            save_assistant_id(assistant_id, filename)
        except Exception as e:
            print(f"An error occurred while creating the assistant: {e}")
            return None

    try:
        thread = client.beta.threads.create()
        print(f'Thread for {name} created...{thread.id}')
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role='user',
            content=content
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status in ['completed', 'failed', 'cancelled']:
                print(f'Run completed with status: {run_status.status}')
                break
            else:
                print(f'{name} Developing Strategy Then Backtesting...')
                time.sleep(5)
        print(f'Run for {name} finished, fetching messages...')
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        return extract_assistant_output(messages.data)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def generate_filename(strategy_description, extension):
    completion = openai_version(
        f"Suggest a potential filename for a trading strategy based on this description: {strategy_description}. The filename should conform to Windows standards and not include a file extension. Your response should only be one string.",
        "You are an experienced software engineer."
    )
    strategy_name = completion.strip()
    strategy_name = strategy_name.replace('"', '')
    strategy_name = re.sub(r'[<>:"/\\|?*]', '_', strategy_name)
    timestamp = datetime.now().strftime("%m_%d_%y_%H_%M")
    return f"{strategy_name}_{timestamp}.{extension}"

def save_output_to_file(output, idea, directory, extension):
    filename = generate_filename(idea, extension)
    filepath = f'{directory}/{filename}'
    try:
        with open(filepath, 'w') as file:
            file.write(output)
        print(output)
        time.sleep(5)
        print(f"Output saved to {filepath}")
    except FileNotFoundError:
        print(f"File not found: {filepath}, moving on to the next one.")

def extract_assistant_output(messages):
    output = ""
    for message in messages:
        if message.role == 'assistant' and hasattr(message.content[0], 'text'):
            output += message.content[0].text.value + "\n"
    return output.strip()

def create_and_run_data_analysis(trading_idea):
    filename = 'strategy_assistant.txt'
    data_analysis_output = create_and_run_assistant(
        name='Strategy Creator AI',
        instructions='Create a trading strategy based on the given trading idea.',
        model='gpt-4o-2024-05-13',
        content=f"Create a trading strategy using {trading_idea}. The strategy should be detailed enough for another AI to code a backtest. Output the instructions for the strategy, assuming that another AI will then code the backtest. THE ONLY OUTPUT YOU WILL MAKE IS THE STRATEGY INSTRUCTIONS FOR THE OTHER AI WHO WILL CODE THE BACKTEST. DO NOT OUTPUT ANYTHING ELSE. DO NOT CODE",
        filename=filename
    )
    if data_analysis_output:
        filename_base = generate_filename(data_analysis_output, 'txt').split('.')[0]
        save_output_to_file(data_analysis_output, data_analysis_output, os.path.join(out_path, 'strategies_event'), 'txt')
        return data_analysis_output, filename_base
    else:
        print(f"No strategy output received for {trading_idea}.")
        return None, None

def extract_trading_strategies(text):
    """
    Use OpenAI to extract trading strategies, rules, or actionable trading ideas from the given text.
    Returns a list of strategies (as strings).
    """
    prompt = (
        "Extract all trading strategies, rules, or actionable trading ideas from the following text. "
        "Do not summarize or analyze sentiment. Output only the strategies in a clear, step-by-step format. "
        f"Text: {text}"
    )
    response = chat_gpt3(prompt, 'You are a trading strategy extraction assistant. Output only actionable trading strategies.')
    return response

def create_and_run_backtest_lumibot(strategy_output, trading_idea, filename_base):
    filename = 'backtest_assistant.txt'
    lumibot_instructions = (
        "Write a complete Python backtest for the provided trading strategy using the Lumibot library. "
        "Use the latest Lumibot API (as of June 2025). "
        "Your code must:\n"
        "- Import from lumibot.strategies, lumibot.backtesting, and datetime.\n"
        "- Define a Strategy subclass with the required methods: initialize() and on_trading_iteration().\n"
        "- Use self.get_last_price(symbol) to get prices, and self.create_order()/self.submit_order() for trades.\n"
        "- Use a parameters dict or class attribute for strategy parameters if needed.\n"
        "- Include a main block that runs the backtest using YahooDataBacktesting or PolygonDataBacktesting.\n"
        "- The backtest should run for a reasonable date range (e.g., last 1-2 years).\n"
        "- Output ONLY the code, nothing else.\n"
        "Do not use backtesting.py. Do not output explanations, only code."
    )
    backtest_output = create_and_run_assistant(
        name='Backtest Coder AI',
        instructions=lumibot_instructions,
        model='gpt-4o-2024-05-13',
        content=f"Strategy Output: {strategy_output}\n\nPlease write the full Lumibot backtest code for this strategy.",
        filename=filename
    )
    if backtest_output and 'lumibot' in backtest_output:
        save_output_to_file(backtest_output, strategy_output, os.path.join(out_path, 'bt_code_event'), 'py')
    else:
        print("Backtest output did not contain Lumibot code. Logging for review.")
        with open("failed_lumibot_generations.txt", "a") as f:
            f.write(f"{trading_idea}\n")

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])
        return ' '.join([t['text'] for t in transcript.fetch()])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        with open("failed_transcripts.txt", "a") as f:
            f.write(f"{video_id}\n")
        return None

def get_pdf_text(url):
    try:
        response = requests.get(url)
        pdf = PyPDF2.PdfReader(BytesIO(response.content))
        text = ""
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text() + "\n"
        return text
    except PyPDF2.errors.PdfReadError:
        print(f"Error reading PDF from {url}")
        return None

def process_trading_ideas(ideas_list):
    for idea in ideas_list:
        print(f"Processing trading idea: {idea}")
        # If the idea is a transcript or long text, extract strategies
        if len(idea) > 200:
            strategies = extract_trading_strategies(idea)
            if not strategies or not strategies.strip():
                print("No strategies extracted from transcript. Skipping.")
                continue
            # If multiple strategies, split and process each
            for strat in strategies.split('\n\n'):
                if strat.strip():
                    strategy_output, filename_base = create_and_run_data_analysis(strat.strip())
                    if strategy_output:
                        create_and_run_backtest_lumibot(strategy_output, strat.strip(), filename_base)
        else:
            strategy_output, filename_base = create_and_run_data_analysis(idea)
            if strategy_output:
                create_and_run_backtest_lumibot(strategy_output, idea, filename_base)

def read_trading_ideas_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def classify_and_process_idea(idea):
    youtube_pattern = r"(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/(watch\?v=)?([a-zA-Z0-9\-_]+)"
    pdf_pattern = r".*\.pdf$"
    youtube_match = re.match(youtube_pattern, idea)
    pdf_match = re.match(pdf_pattern, idea)
    if youtube_match:
        video_id = youtube_match.groups()[-1]
        transcript = get_youtube_transcript(video_id)
        if transcript:
            transcript = chunk_it(transcript)
            process_trading_ideas([transcript])
    elif pdf_match:
        if idea.startswith("http"):
            pdf_text = get_pdf_text(idea)
        else:
            if os.path.isfile(idea):
                pdf_text = get_local_pdf_text(idea)
            else:
                print(f"Invalid file path: {idea}")
                return
        if pdf_text:
            pdf_text = chunk_it(pdf_text)
            process_trading_ideas([pdf_text])
    else:
        process_trading_ideas([idea])

def get_local_pdf_text(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(pdf.pages)):
                text += pdf.pages[page].extract_text() + "\n"
            return text
    except PyPDF2.errors.PdfReadError:
        print(f"Error reading PDF file: {file_path}")
        return None

def chunk_it(idea):
    # Instead of summarizing, extract strategies from each chunk
    all_text = idea
    chunks = textwrap.wrap(all_text, 2500)
    result = []
    for i, chunk in enumerate(chunks, start=1):
        strategies = extract_trading_strategies(chunk)
        print(f'{i} out of {len(chunks)} Strategy Extractions: {strategies}')
        result.append(strategies)
        save_file('output.txt', '\n\n'.join(result))
    return result

def main_idea_processor(file_path):
    global run_counter
    with open(file_path, 'r') as file:
        ideas = [line.strip() for line in file.readlines() if line.strip()]
    for idea in ideas:
        run_counter += 1
        print(f"Run #{run_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Processing trading idea: {idea}")
        classify_and_process_idea(idea)

run_counter = 0

if __name__ == "__main__":
    main_idea_processor('strat_ideas.txt')
