from openai import OpenAI

client = OpenAI(api_key='enter your api key here')

def chat(user_input):
    # Create a chat completion using the user's input, with a maximum of 200 tokens
    completion = client.chat.completions.create(
    model='gpt-4',
    messages=[
      {'role': 'system', 'content': 'You are a financial advisor helping a customer to invest in the stock market, and you provide your own suggestions.'},
      {'role': 'user', 'content': user_input}
    ],
    max_tokens=2000  # Setting the maximum number of tokens
    )
    return completion

def main():
    # Print the response, removing newline
    print('Ask a question regarding a sector in the stock market')
    print('')
    user_input = input('Enter a message: ')
    completion = chat(user_input)
    print('')
    print(completion.choices[0].message.content.replace('\n', ' '))


if __name__ == '__main__':
    main()
