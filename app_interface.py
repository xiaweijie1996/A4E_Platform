import gradio as gr
import matplotlib.pyplot as plt
import tool as tl
import pandas as pd
import openai
print(openai.__version__)

import numpy as np

def generate_greeting(building_type, building_age, family, annual_electricity_consumption, data_mean, data_peak):
    openai.api_key = 'put your openai api key here'
    
    prompt = (
        f"Create a warm and engaging greeting for a user interested in predicting their home's energy consumption. "
        f"The home's building type is {building_type}, its age is {building_age}, the family type is {family}, "
        f"and their annual electricity consumption is {annual_electricity_consumption} kWh. "
        f"Can use icons like 🌟🏡💡✨ to be more expressive and adorable "
        f"Also descire the data mean and peak as {data_mean} and {data_peak} respectively in Wh."
    )

    response = openai.Completion.create(
      engine="gpt-3.5-turbo-instruct",
      prompt=prompt,
      temperature=0.7,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    
    greeting = response.choices[0].text.strip()
    return greeting


def greet_and_plot(building_type, building_age, family, annual_electricity_consumption, number_data, GPT=False):
    # Greeting based on time_of_day
  
    text_input = [annual_electricity_consumption, building_type, building_age, family]
    input_x = tl.text_to_input(text_input)
    give_x = tl.input_encode(input_x)
    
    # Load the model and generate samples
    fig, samples = tl.sample_and_plot(give_x, number_samples=number_data)
    
    # data information
    data_mean = np.mean(samples)
    data_peak = np.max(samples)
    # round to 2 decimal places
    data_mean = round(data_mean, 2)
    data_peak = round(data_peak, 2)
    
    if GPT == False:
        greeting = f"🌟 Hello and a warm welcome! 🏡 We're thrilled to help you explore the energy consumption patterns of your lovely home. With your building type as {building_type}, a rich history dating back to {building_age}, the warmth of a {family} family setup, and an annual electricity footprint of {annual_electricity_consumption} kWh, we're all set to unveil a bespoke energy consumption forecast just for you. Dive in to discover how your sanctuary stands in terms of energy use and ways to optimize it. Let's embark on this enlightening journey together! 💡✨"
    else:
        greeting = generate_greeting(building_type, building_age, family, annual_electricity_consumption, data_mean, data_peak)
    
    fig.tight_layout(pad=3.0) 
    
    return greeting, fig # , csv_string

# Define Gradio interface with a Radio component for time of day selection
demo = gr.Interface(
    fn=greet_and_plot,
    inputs=[
        gr.Radio(choices=["2 onder 1 kap", "Appartement", "Overige", "Rijtjeswoning", "Vrijstaande woning"], label="Building type", info="Please select the building type"),
        gr.Radio(choices=["1940-1979", "1980-heden", "voor 1940"], label="Building Age", info="Please select the building age"),
        gr.Radio(choices=["1 Alleenstaande", "2 Gezin met kinderen", "3 Paar zonder kinderen"], label="Family", info="Please select the family type"),
        gr.Slider(0, 15000, label="Annual electricity consumption", info="Move the slider to set the annual electricity consumption in kWh"),
        gr.Slider(0, 500, label="Amout of data", info="Move the slider to set the number of data points to be generated for the plot")
        # gr.Textbox(label="Name", info="If you do not mind we know who you are :)")
    ],
    outputs=[
        gr.Text(label="Greeting Message"),
        gr.Plot(label="Energy Consumption Plot")
        # gr.File(label="Download Generated Data")  # Add this line for data download
    ],
)

# Launch the application
demo.launch(share=True)
