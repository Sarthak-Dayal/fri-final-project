from openai import OpenAI
import base64
import requests
client = OpenAI()

# OpenAI API Key
api_key = "sk-proj-TBXs37oBZ11YGa0TIMKGT3BlbkFJ1Ful5iX1TCpYNNQFkHuu"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Create the base64 encoded images from the image paths
image_paths = ["images/all_books.png", "images/all_books_2.jpeg", "images/book_3.png"]
base64_images = [encode_image(image_path) for image_path in image_paths]

# Create the headers and payload for the API request

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "These are multiple images of a number of books. Can you list all of \
                            #the book titles in each of these images from left to right, separated \
                            #by commas? List each of the titles separately, separated \
                            #by a new line, following the format as given:\n \
                            #Image 1 Titles:\n\t'Title 1'\n\t'Title 2'\n\t'Title 3', ...\n\n \
                            #Image 2 Titles:\n\t'Title 1'\n\t'Title 2'\n\t'Title 3', ...\n\n"
                }
            ]
        }
    ],
    "temperature": 0.2
}

# Add each of the images from base64_images to the payload in the format 
# required by the API in the content field of the messages list
for base64_image in base64_images:
    payload["messages"][0]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })

response = requests.post("https://api.openai.com/v1/chat/completions", \
    headers=headers, json=payload)
print("Status Code:", response.status_code)
print("Response Content:", response.text)  # Print the raw response text

# Proceed only if the response is successful
if response.status_code == 200:
    try:
        print(response.json().get("choices")[0].get("message").get("content"))
    except Exception as e:
        print("Error parsing JSON:", e)
else:
    print("Failed to fetch data:", response.text)
